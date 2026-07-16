"""Unit tests for rooflang.runtime.simulator (Simulator DES)."""

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import AllReduce
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Bound, OOMError, Simulator


# ── Test helpers ─────────────────────────────────────────────────────


class SyntheticKernel(Kernel):
    """Kernel with controllable flops for testing."""

    def __init__(self, flops_val=0.0, **kwargs):
        super().__init__(**kwargs)
        self._flops_val = flops_val

    @property
    def flops(self):
        return self._flops_val


def _hw(read_bw=1.0, write_bw=1.0, tflops=1.0):
    """Single GPU + HBM. Bandwidths in GB/s, compute in TFLOPS bf16."""
    gpu = Compute(name="gpu0", tflops={"bf16": tflops})
    hbm = Memory(name="hbm", capacity_gb=80.0)
    hw = HardwareGraph()
    hw.add_node(gpu)
    hw.add_node(hbm)
    hw.add_edge(FabricEdge(name="hbm_link", src=gpu, dst=hbm,
                           src_to_dst_bandwidth_gbs=write_bw,
                           dst_to_src_bandwidth_gbs=read_bw,
                           is_full_duplex=False))
    return hw, gpu, hbm


def _sim(graph, placement, hardware):
    return Simulator(graph, placement, hardware).run()


# ── Single kernel timing ─────────────────────────────────────────────


class TestSingleKernel:
    def test_compute_bound(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # peak = 1e6 FLOPS/us; flops = 2e6 → ct = 2 us
        # input 4 bytes → read = 4 / (1000*1e3) ≈ 0 us
        t_in = Tensor("bf16", (1,))
        k = SyntheticKernel(flops_val=2e6, inputs={"x": t_in})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = _sim(g, p, hw)
        assert len(result.trace) == 1
        assert result.total_time_us == pytest.approx(2.0, rel=1e-6)
        assert result.trace[0].bound == Bound.COMPUTE

    def test_memory_bound(self):
        hw, gpu, hbm = _hw(read_bw=1.0, write_bw=1.0, tflops=1000.0)
        # peak = 1e9 FLOPS/us; flops = 1e6 → ct ≈ 0.001 us
        # input 2000 bytes → read = 2000 / (1.0*1e3) = 2 us
        t_in = Tensor("bf16", (1000,))
        k = SyntheticKernel(flops_val=1e6, inputs={"x": t_in})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(2.0, rel=1e-6)
        assert result.trace[0].bound == Bound.MEMORY


# ── Stream serialization ─────────────────────────────────────────────


class TestStreamSerial:
    def test_same_stream_serial(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # Two independent kernels on same stream, each 1 us compute
        t1 = Tensor("bf16", (1,))
        t2 = Tensor("bf16", (1,))
        k1 = SyntheticKernel(flops_val=1e6, inputs={"x": t1})
        k2 = SyntheticKernel(flops_val=1e6, inputs={"x": t2})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        p = Placement(hardware=hw)
        p.set_kernel_device(k1, gpu, stream=0)
        p.set_kernel_device(k2, gpu, stream=0)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(2.0, rel=1e-6)

    def test_different_streams_parallel(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # Two kernels on different streams, each 1 us compute
        # They share the device → each gets dev_share = 0.5 → each takes 2 us
        # Total = 2 us (parallel)
        t1 = Tensor("bf16", (1,))
        t2 = Tensor("bf16", (1,))
        k1 = SyntheticKernel(flops_val=1e6, inputs={"x": t1})
        k2 = SyntheticKernel(flops_val=1e6, inputs={"x": t2})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        p = Placement(hardware=hw)
        p.set_kernel_device(k1, gpu, stream=0)
        p.set_kernel_device(k2, gpu, stream=1)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(2.0, rel=1e-6)

    def test_three_on_same_stream(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        kernels = []
        for _ in range(3):
            t = Tensor("bf16", (1,))
            k = SyntheticKernel(flops_val=1e6, inputs={"x": t})
            kernels.append(k)
        g = ComputeGraph()
        p = Placement(hardware=hw)
        for k in kernels:
            g.add_kernel(k)
            p.set_kernel_device(k, gpu, stream=0)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(3.0, rel=1e-6)


# ── Resource sharing ─────────────────────────────────────────────────


class TestResourceSharing:
    def test_weighted_cap(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # k1: cap=0.75, k2: cap=0.25, on different streams
        # total_cap = 1.0 → each gets its own cap fraction as share
        # k1: ct = 1e6/(1e6 * 0.75) = 1.333 us
        # k2: ct = 1e6/(1e6 * 0.25) = 4.0 us
        # They run in parallel; total = max(1.333, 4.0) = 4.0 us
        t1 = Tensor("bf16", (1,))
        t2 = Tensor("bf16", (1,))
        k1 = SyntheticKernel(flops_val=1e6, inputs={"x": t1})
        k2 = SyntheticKernel(flops_val=1e6, inputs={"x": t2})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        p = Placement(hardware=hw)
        p.set_kernel_device(k1, gpu, stream=0, resource_cap=0.75)
        p.set_kernel_device(k2, gpu, stream=1, resource_cap=0.25)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(4.0, rel=1e-6)


# ── Read/write bandwidth direction ──────────────────────────────────


class TestBandwidthDirection:
    def test_asymmetric_read_write(self):
        # read_bw = 2.0 GB/s, write_bw = 1.0 GB/s
        hw, gpu, hbm = _hw(read_bw=2.0, write_bw=1.0, tflops=1000.0)
        # input: 2000 bytes → read = 2000 / (2.0*1e3) = 1 us
        # output: 2000 bytes → write = 2000 / (1.0*1e3) = 2 us
        # total mt = 3 us (compute negligible)
        t_in = Tensor("bf16", (1000,))
        t_out = Tensor("bf16", (1000,))
        k = SyntheticKernel(flops_val=0.0, inputs={"x": t_in}, outputs={"y": t_out})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(3.0, rel=1e-6)

    def test_symmetric_same_result(self):
        hw, gpu, hbm = _hw(read_bw=2.0, write_bw=2.0, tflops=1000.0)
        # input + output each 2000 bytes → each 1 us → total 2 us
        t_in = Tensor("bf16", (1000,))
        t_out = Tensor("bf16", (1000,))
        k = SyntheticKernel(flops_val=0.0, inputs={"x": t_in}, outputs={"y": t_out})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(2.0, rel=1e-6)


# ── Zero-value errors ────────────────────────────────────────────────


class TestZeroValueErrors:
    def test_zero_compute_raises(self):
        gpu = Compute(name="gpu0", tflops={"fp32": 1.0})
        hbm = Memory(name="hbm", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu)
        hw.add_node(hbm)
        hw.add_edge(FabricEdge(name="link", src=gpu, dst=hbm,
                               src_to_dst_bandwidth_gbs=1.0,
                               dst_to_src_bandwidth_gbs=1.0,
                               is_full_duplex=False))
        t_in = Tensor("bf16", (1,))
        k = SyntheticKernel(flops_val=1e6, inputs={"x": t_in})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        with pytest.raises(ValueError, match="no compute for dtype"):
            _sim(g, p, hw)

    def test_zero_bandwidth_raises(self):
        gpu = Compute(name="gpu0", tflops={"bf16": 1.0})
        hbm = Memory(name="hbm", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu)
        hw.add_node(hbm)
        hw.add_edge(FabricEdge(name="link", src=gpu, dst=hbm,
                               src_to_dst_bandwidth_gbs=1.0,
                               dst_to_src_bandwidth_gbs=0.0,
                               is_full_duplex=False))
        t_in = Tensor("bf16", (100,))
        k = SyntheticKernel(flops_val=0.0, inputs={"x": t_in})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        with pytest.raises(ValueError, match="Zero read bandwidth"):
            _sim(g, p, hw)


# ── Alpha not scaled by contention ───────────────────────────────────


class TestAlphaSeparation:
    def test_alpha_fixed_time(self):
        # Two GPUs connected by NVLink with alpha=5 us
        gpu0 = Compute(name="gpu0", tflops={"bf16": 1000.0})
        gpu1 = Compute(name="gpu1", tflops={"bf16": 1000.0})
        hbm0 = Memory(name="hbm0", capacity_gb=80.0)
        hbm1 = Memory(name="hbm1", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu0)
        hw.add_node(gpu1)
        hw.add_node(hbm0)
        hw.add_node(hbm1)
        hw.add_edge(FabricEdge(name="hbm0_link", src=gpu0, dst=hbm0,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="hbm1_link", src=gpu1, dst=hbm1,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        nvlink = FabricEdge(name="nvlink", src=gpu0, dst=gpu1,
                            src_to_dst_bandwidth_gbs=100.0,
                            dst_to_src_bandwidth_gbs=100.0,
                            is_full_duplex=True, alpha_us=5.0)
        hw.add_edge(nvlink)

        # AllReduce: 100000 bytes, world=2
        # transferred = 2*(1/2)*100000 = 100000 bytes
        # eff_bw via find_aggregate_bandwidth — for 2 GPUs it should be ~100 GB/s
        # xfer = 100000 / (100 * 1e3) = 1 us
        # alpha = 5 us
        # total network = 5 + 1 = 6 us
        t_out = Tensor("bf16", (1,))
        k_pred = SyntheticKernel(flops_val=0.0, outputs={"y": t_out})
        t_out2 = Tensor("bf16", (1,))
        k_succ = SyntheticKernel(flops_val=0.0, inputs={"a": Tensor("bf16", (1,))})

        ar = AllReduce(total_bytes=100000.0, world=2, dtype="bf16")
        ar.inputs = {"x": Tensor("bf16", (1,))}
        ar.outputs = {"y": Tensor("bf16", (1,))}

        g = ComputeGraph()
        g.add_kernel(k_pred)
        g.add_kernel(ar)
        g.add_kernel(k_succ)
        g.add_data_edge(k_pred, ar, {"y": "x"})
        g.add_data_edge(ar, k_succ, {"y": "a"})

        p = Placement(hardware=hw)
        p.set_kernel_device(k_pred, gpu0)
        p.set_kernel_device(k_succ, gpu1)

        result = _sim(g, p, hw)
        # Find the AllReduce trace entry
        ar_entry = [e for e in result.trace if e.kernel is ar]
        assert len(ar_entry) == 1
        ar_time = ar_entry[0].end_us - ar_entry[0].start_us
        # alpha=5, xfer≈1 → total ≈ 6 us
        assert ar_time == pytest.approx(6.0, rel=0.1)


# ── DAG ordering ─────────────────────────────────────────────────────


class TestDAGOrdering:
    def test_sequential_chain(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # k1 → k2 → k3, each 1 us compute
        t1_out = Tensor("bf16", (1,))
        t2_in = Tensor("bf16", (1,))
        t2_out = Tensor("bf16", (1,))
        t3_in = Tensor("bf16", (1,))
        k1 = SyntheticKernel(flops_val=1e6, outputs={"y": t1_out})
        k2 = SyntheticKernel(flops_val=1e6, inputs={"x": t2_in}, outputs={"y": t2_out})
        k3 = SyntheticKernel(flops_val=1e6, inputs={"x": t3_in})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_kernel(k3)
        g.add_data_edge(k1, k2, {"y": "x"})
        g.add_data_edge(k2, k3, {"y": "x"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpu)
        p.set_kernel_device(k2, gpu)
        p.set_kernel_device(k3, gpu)
        result = _sim(g, p, hw)
        assert result.total_time_us == pytest.approx(3.0, rel=1e-6)
        # Verify ordering
        entries = {e.kernel: e for e in result.trace}
        assert entries[k1].end_us <= entries[k2].start_us
        assert entries[k2].end_us <= entries[k3].start_us

    def test_fork_join(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # k1 → k2a (stream 0), k1 → k2b (stream 1), both → k3
        # k2a and k2b parallel (different streams, share device)
        t1_out = Tensor("bf16", (1,))
        t2a_in = Tensor("bf16", (1,))
        t2a_out = Tensor("bf16", (1,))
        t2b_in = Tensor("bf16", (1,))
        t2b_out = Tensor("bf16", (1,))
        t3_in1 = Tensor("bf16", (1,))
        t3_in2 = Tensor("bf16", (1,))
        k1 = SyntheticKernel(flops_val=1e6, outputs={"y": t1_out})
        k2a = SyntheticKernel(flops_val=1e6, inputs={"x": t2a_in}, outputs={"y": t2a_out})
        k2b = SyntheticKernel(flops_val=1e6, inputs={"x": t2b_in}, outputs={"y": t2b_out})
        k3 = SyntheticKernel(flops_val=1e6, inputs={"a": t3_in1, "b": t3_in2})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2a)
        g.add_kernel(k2b)
        g.add_kernel(k3)
        g.add_data_edge(k1, k2a, {"y": "x"})
        g.add_data_edge(k1, k2b, {"y": "x"})
        g.add_data_edge(k2a, k3, {"y": "a"})
        g.add_data_edge(k2b, k3, {"y": "b"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpu, stream=0)
        p.set_kernel_device(k2a, gpu, stream=0)
        p.set_kernel_device(k2b, gpu, stream=1)
        p.set_kernel_device(k3, gpu, stream=0)
        result = _sim(g, p, hw)
        # k1: 1us (alone). Then k2a+k2b parallel with share 0.5 each → 2us.
        # k3: 1us (alone). Total = 1 + 2 + 1 = 4us
        assert result.total_time_us == pytest.approx(4.0, rel=1e-6)


# ── Empty graph ──────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_graph(self):
        hw, gpu, hbm = _hw()
        g = ComputeGraph()
        p = Placement(hardware=hw)
        result = _sim(g, p, hw)
        assert result.total_time_us == 0.0
        assert result.trace == []


# ── OOM detection ───────────────────────────────────────────────────


def _hw_small(capacity_gb=0.001, read_bw=1000.0, write_bw=1000.0, tflops=1.0):
    """Single GPU + small HBM for OOM tests."""
    gpu = Compute(name="gpu0", tflops={"bf16": tflops})
    hbm = Memory(name="hbm", capacity_gb=capacity_gb)
    hw = HardwareGraph()
    hw.add_node(gpu)
    hw.add_node(hbm)
    hw.add_edge(FabricEdge(name="hbm_link", src=gpu, dst=hbm,
                           src_to_dst_bandwidth_gbs=write_bw,
                           dst_to_src_bandwidth_gbs=read_bw,
                           is_full_duplex=False))
    return hw, gpu, hbm


class TestOOM:
    def test_no_overflow(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        t_in = Tensor("bf16", (4,))
        k = SyntheticKernel(flops_val=0.0, inputs={"x": t_in})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = _sim(g, p, hw)
        assert hbm in result.peak_memory
        assert result.peak_memory[hbm] == t_in.size_bytes

    def test_oom_raises(self):
        hw, gpu, hbm = _hw_small(capacity_gb=0.000001)  # 1000 bytes
        # output = 10000 bf16 elements = 20000 bytes > 1000
        t_in = Tensor("bf16", (1,))
        t_out = Tensor("bf16", (10000,))
        k = SyntheticKernel(flops_val=0.0, inputs={"x": t_in},
                            outputs={"y": t_out})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        with pytest.raises(OOMError) as exc_info:
            _sim(g, p, hw)
        err = exc_info.value
        assert err.memory is hbm
        assert err.used_bytes > err.capacity_bytes
        assert err.trigger_kernel is k
        assert len(err.alive_tensors) > 0

    def test_weight_counted_in_peak(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        t_w = Tensor("fp32", (1000,))  # 4000 bytes
        t_in = Tensor("bf16", (1,))
        k = SyntheticKernel(flops_val=0.0, inputs={"x": t_in},
                            weights={"W": t_w})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = _sim(g, p, hw)
        assert result.peak_memory[hbm] >= t_w.size_bytes

    def test_output_freed_after_consumer(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # k1 → k2 → k3: k1's output freed after k2 completes
        t1_out = Tensor("bf16", (500,))  # 1000 bytes
        t2_in = Tensor("bf16", (500,))
        t2_out = Tensor("bf16", (500,))
        t3_in = Tensor("bf16", (500,))
        k1 = SyntheticKernel(flops_val=1e6, outputs={"y": t1_out})
        k2 = SyntheticKernel(flops_val=1e6, inputs={"x": t2_in},
                             outputs={"y": t2_out})
        k3 = SyntheticKernel(flops_val=1e6, inputs={"x": t3_in})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_kernel(k3)
        g.add_data_edge(k1, k2, {"y": "x"})
        g.add_data_edge(k2, k3, {"y": "x"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpu)
        p.set_kernel_device(k2, gpu)
        p.set_kernel_device(k3, gpu)
        result = _sim(g, p, hw)
        # Peak should be 2 * 1000 (k1.out + k2.out alive simultaneously)
        # not 3 * 1000 (all three alive at once)
        assert result.peak_memory[hbm] == pytest.approx(2000.0)

    def test_root_input_freed_after_kernel(self):
        hw, gpu, hbm = _hw(read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        # k1 (root, has input) → k2: k1's input freed after k1 completes
        t1_in = Tensor("bf16", (2000,))  # 4000 bytes
        t1_out = Tensor("bf16", (1,))    # 2 bytes
        t2_in = Tensor("bf16", (1,))
        k1 = SyntheticKernel(flops_val=1e6, inputs={"x": t1_in},
                             outputs={"y": t1_out})
        k2 = SyntheticKernel(flops_val=1e6, inputs={"x": t2_in})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_data_edge(k1, k2, {"y": "x"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpu)
        p.set_kernel_device(k2, gpu)
        result = _sim(g, p, hw)
        # Peak = t1_in (4000) + t1_out (2) at the start of k1
        # After k1 completes: t1_in freed, only t1_out (2) alive
        assert result.peak_memory[hbm] == pytest.approx(4002.0)
