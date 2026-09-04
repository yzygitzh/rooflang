# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Unit tests for rooflang.runtime.simulator (Simulator DES)."""

from fractions import Fraction

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import AllReduce, Gather, Scatter
from rooflang.language.kernels.forward import Nop, Slice
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.optimization.comm import optimize_comms
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


class MixedDtypeKernel(Kernel):
    """Kernel with serial compute segments at different dtype peaks."""

    def __init__(self, flops_by_dtype, **kwargs):
        super().__init__(**kwargs)
        self._flops_by_dtype = flops_by_dtype

    @property
    def flops(self):
        return sum(self._flops_by_dtype.values())

    @property
    def flops_by_dtype(self):
        return self._flops_by_dtype


def _hw(read_bw=1.0, write_bw=1.0, tflops=1.0,
        tflops_by_dtype=None):
    """Single GPU + HBM. Bandwidths in GB/s, compute in TFLOPS bf16."""
    gpu = Compute(
        name="gpu0",
        tflops=tflops_by_dtype or {"bf16": tflops},
    )
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


def _place_comm_tensors(placement, kernel, memories):
    """Assign comm ports round-robin to their participant memories."""
    tensors = list(kernel.inputs.values()) + list(kernel.outputs.values())
    for index, tensor in enumerate(tensors):
        placement.set_tensor_memory(tensor, memories[index % len(memories)])


# ── Single kernel timing ─────────────────────────────────────────────


class TestSingleKernel:
    def test_mixed_dtype_compute_segments_are_summed(self):
        hw, gpu, hbm = _hw(
            read_bw=1000.0,
            write_bw=1000.0,
            tflops_by_dtype={"fp8": 2.0, "fp4": 4.0},
        )
        kernel = MixedDtypeKernel(
            {"fp8": 2e6, "fp4": 4e6},
            inputs={"x": Tensor("bf16", (1,))},
        )
        graph = ComputeGraph()
        graph.add_kernel(kernel)
        placement = Placement(hardware=hw)
        placement.set_kernel_device(kernel, gpu)

        result = _sim(graph, placement, hw)

        # 2e6 / 2 TFLOPS = 1 us; 4e6 / 4 TFLOPS = 1 us.
        assert result.total_time_us == pytest.approx(2.0)
        assert result.trace[0].compute_time_us == pytest.approx(2.0)

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
        entry = result.trace[0]
        assert entry.bound == Bound.COMPUTE
        assert entry.compute_time_us == pytest.approx(2.0)
        assert entry.memory_time_us == pytest.approx(2e-6)
        assert entry.network_time_us == 0.0
        assert entry.local_elapsed_time_us == pytest.approx(2.0)
        assert entry.network_elapsed_time_us == 0.0

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
        entry = result.trace[0]
        assert entry.bound == Bound.MEMORY
        assert entry.compute_time_us == pytest.approx(0.001)
        assert entry.memory_time_us == pytest.approx(2.0)
        assert entry.network_time_us == 0.0
        assert entry.local_elapsed_time_us == pytest.approx(2.0)
        assert entry.network_elapsed_time_us == 0.0

    def test_fractional_weight_read_keeps_full_resident_capacity(self):
        hw, gpu, hbm = _hw(read_bw=1.0, write_bw=1.0, tflops=1.0)
        weight = Tensor("bf16", (1000,))
        kernel = SyntheticKernel(weights={"w": weight})
        kernel.weight_read_fraction = Fraction(1, 4)
        graph = ComputeGraph()
        graph.add_kernel(kernel)
        placement = Placement(hardware=hw)
        placement.set_kernel_device(kernel, gpu)

        result = _sim(graph, placement, hw)

        assert result.trace[0].memory_time_us == pytest.approx(0.5)
        assert result.peak_memory[hbm] == 2000.0

    def test_fractional_input_read_keeps_full_resident_capacity(self):
        class FractionalInputKernel(SyntheticKernel):
            def input_read_fraction(self, port):
                return Fraction(1, 4) if port == "x" else 1

        hw, gpu, hbm = _hw(read_bw=1.0, write_bw=1.0, tflops=1.0)
        tensor = Tensor("bf16", (1000,))
        kernel = FractionalInputKernel(inputs={"x": tensor})
        graph = ComputeGraph()
        graph.add_kernel(kernel)
        placement = Placement(hardware=hw)
        placement.set_kernel_device(kernel, gpu)

        graph.validate()
        result = _sim(graph, placement, hw)

        assert kernel.input_tensor_bytes == 2000.0
        assert kernel.input_bytes == 500.0
        assert result.trace[0].memory_time_us == pytest.approx(0.5)
        assert result.peak_memory[hbm] == 2000.0


# ── Stream serialization ─────────────────────────────────────────────


class TestStreamSerial:
    def test_measurement_starts_after_control_dependency(self):
        hw, gpu, hbm = _hw(
            read_bw=1000.0, write_bw=1000.0, tflops=1.0)
        preload = SyntheticKernel(flops_val=2e6)
        measured = SyntheticKernel(flops_val=3e6)
        graph = ComputeGraph()
        graph.add_kernel(preload)
        graph.add_kernel(measured)
        graph.add_control_edge(preload, measured)
        placement = Placement(hardware=hw)
        placement.set_kernel_device(preload, gpu)
        placement.set_kernel_device(measured, gpu)

        result = Simulator(
            graph, placement, hw, measurement_start=measured).run()

        assert result.total_time_us == pytest.approx(5.0)
        assert result.measurement_start_us == pytest.approx(2.0)
        assert result.measured_time_us == pytest.approx(3.0)

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
        for entry in result.trace:
            assert entry.local_elapsed_time_us == pytest.approx(2.0)
            assert entry.network_elapsed_time_us == 0.0

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
        elapsed_by_kernel = {
            entry.kernel: entry.local_elapsed_time_us
            for entry in result.trace
        }
        assert elapsed_by_kernel[k1] == pytest.approx(4 / 3)
        assert elapsed_by_kernel[k2] == pytest.approx(4.0)


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
        _place_comm_tensors(p, ar, [hbm0, hbm1])

        result = _sim(g, p, hw)
        # Find the AllReduce trace entries (one per participant)
        ar_entry = [e for e in result.trace if e.kernel is ar]
        assert len(ar_entry) == 2
        ar_time = ar_entry[0].end_us - ar_entry[0].start_us
        # alpha=5, xfer≈1 → total ≈ 6 us
        assert ar_time == pytest.approx(6.0, rel=0.1)


def test_collective_topology_info_is_cached_by_device_set():
    hw, gpus, _ = _hw_multi_gpu(n_gpus=2)
    simulator = Simulator(ComputeGraph(), Placement(), hw)

    first = simulator._collective_fabric_info(gpus, False)
    second = simulator._collective_fabric_info(list(reversed(gpus)), False)

    assert first is second
    assert len(simulator._collective_info_cache) == 1


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


class TestMaterializedSlice:
    def test_allocates_compact_output_and_charges_memory_io(self):
        hw, gpu, hbm = _hw(read_bw=1.0, write_bw=1.0, tflops=1.0)
        materialize = Slice()
        materialize.inputs = {"x": Tensor("bf16", (1000,))}
        materialize.outputs = {"y": Tensor("bf16", (100,))}
        g = ComputeGraph()
        g.add_kernel(materialize)
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(materialize, gpu)

        result = _sim(g, p, hw)

        assert result.total_time_us == pytest.approx(2.2)
        assert result.peak_memory[hbm] == 2200.0


class TestNop:
    def test_preserves_dependencies_without_timing_cost(self):
        hw, gpu, hbm = _hw(read_bw=1.0, write_bw=1.0, tflops=1.0)
        source = SyntheticKernel(outputs={"y": Tensor("bf16", (1000,))})
        nop = Nop(
            inputs={"payload": Tensor("bf16", (1000,))},
            outputs={"done": Tensor("int32", (7,))},
        )
        sink = SyntheticKernel(inputs={"x": Tensor("int32", (7,))})
        g = ComputeGraph()
        for kernel in (source, nop, sink):
            g.add_kernel(kernel)
        g.add_data_edge(source, nop, {"y": "payload"})
        g.add_data_edge(nop, sink, {"done": "x"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(source, gpu)
        p.set_kernel_device(sink, gpu)
        p.set_tensor_memory(nop.inputs["payload"], hbm)
        p.set_tensor_memory(nop.outputs["done"], hbm)
        g.validate()
        p.validate(g)

        result = _sim(g, p, hw)

        entries = {entry.kernel: entry for entry in result.trace}
        assert entries[source].end_us <= entries[nop].start_us
        assert entries[nop].start_us == entries[nop].end_us
        assert entries[nop].end_us <= entries[sink].start_us


class TestPassthroughResolution:
    def test_transitive_aliases_resolve_to_storage_source(self):
        source = SyntheticKernel(outputs={"y": Tensor("bf16", (4,))})
        fan1 = Spawn(world=1)
        fan1.inputs = {"x": Tensor("bf16", (4,))}
        fan1.outputs = {"y": Tensor("bf16", (4,))}
        fan2 = Spawn(world=1)
        fan2.inputs = {"x": Tensor("bf16", (4,))}
        fan2.outputs = {"y": Tensor("bf16", (4,))}
        g = ComputeGraph()
        for kernel in (source, fan1, fan2):
            g.add_kernel(kernel)
        g.add_data_edge(source, fan1, {"y": "x"})
        g.add_data_edge(fan1, fan2, {"y": "x"})

        passthrough = Simulator(
            g, Placement(), HardwareGraph())._build_passthrough()

        assert passthrough[fan1.outputs["y"]] == [source.outputs["y"]]
        assert passthrough[fan2.outputs["y"]] == [source.outputs["y"]]

    def test_converging_alias_paths_do_not_duplicate_storage_sources(self):
        source = SyntheticKernel(outputs={"y": Tensor("bf16", (4,))})
        fan = Nop(
            inputs={"x": Tensor("bf16", (4,))},
            outputs={
                "a": Tensor("bf16", (4,)),
                "b": Tensor("bf16", (4,)),
            },
        )
        merge = Nop(
            inputs={
                "a": Tensor("bf16", (4,)),
                "b": Tensor("bf16", (4,)),
            },
            outputs={"y": Tensor("bf16", (4,))},
        )
        graph = ComputeGraph()
        for kernel in (source, fan, merge):
            graph.add_kernel(kernel)
        graph.add_data_edge(source, fan, {"y": "x"})
        graph.add_data_edge(fan, merge, {"a": "a", "b": "b"})

        passthrough = Simulator(
            graph, Placement(), HardwareGraph())._build_passthrough()

        assert passthrough[merge.outputs["y"]] == [source.outputs["y"]]


class TestPeerUpdateDeduplication:
    def test_each_running_kernel_recomputes_once_per_start_and_finish(self):
        switch = Compute(name="switch", kind="switch")
        gpus = [
            Compute(name=f"gpu{i}", tflops={"bf16": 1000.0})
            for i in range(3)
        ]
        hbms = [Memory(name=f"hbm{i}", capacity_gb=80.0)
                for i in range(3)]
        hardware = HardwareGraph()
        hardware.add_node(switch)
        for gpu, hbm in zip(gpus, hbms):
            hardware.add_node(gpu)
            hardware.add_node(hbm)
            hardware.add_edge(FabricEdge(
                name="hbm", src=gpu, dst=hbm,
                src_to_dst_bandwidth_gbs=1000.0,
                dst_to_src_bandwidth_gbs=1000.0,
                is_full_duplex=False,
            ))
            hardware.add_edge(FabricEdge(
                name="fabric", src=gpu, dst=switch,
                src_to_dst_bandwidth_gbs=100.0,
                dst_to_src_bandwidth_gbs=100.0,
                is_full_duplex=True,
            ))

        graph = ComputeGraph()
        placement = Placement(hardware=hardware, graph=graph)
        collectives = []
        for stream in range(2):
            predecessor = SyntheticKernel(
                outputs={"y": Tensor("bf16", (1,))})
            collective = AllReduce(
                total_bytes=200000.0, world=3, dtype="bf16")
            collective.inputs = {
                f"i{i}": Tensor("bf16", (1,)) for i in range(3)
            }
            collective.outputs = {
                f"o{i}": Tensor("bf16", (1,)) for i in range(3)
            }
            graph.add_kernel(predecessor)
            graph.add_kernel(collective)
            graph.add_data_edge(predecessor, collective, {"y": "i0"})
            placement.set_kernel_device(
                predecessor, gpus[0], stream=stream)
            _place_comm_tensors(placement, collective, hbms)
            collectives.append(collective)

        simulator = Simulator(graph, placement, hardware)
        original = simulator._recompute_shares
        calls = []

        def record(running):
            calls.append(running.kernel)
            original(running)

        simulator._recompute_shares = record
        simulator.run()

        assert all(calls.count(collective) == 2
                   for collective in collectives)


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

    def test_memory_footprint_is_metadata_not_simulated_usage(self):
        hw, gpu, hbm = _hw_small(capacity_gb=0.000001)  # 1000 bytes
        kernel = SyntheticKernel(flops_val=1e6)
        graph = ComputeGraph()
        graph.add_kernel(kernel)
        placement = Placement(hardware=hw)
        placement.set_kernel_device(kernel, gpu)
        placement.record_memory_footprint(hbm, 1200.0, "kv_cache")

        result = _sim(graph, placement, hw)

        assert result.peak_memory.get(hbm, 0.0) == 0.0
        footprint, = result.memory_footprints
        assert footprint.memory is hbm
        assert footprint.size_bytes == 1200.0
        assert footprint.role == "kv_cache"

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


# ── Cross-device fabric sharing ──────────────────────────────────────


def _hw_multi_gpu(n_gpus=2, hbm_bw=1000.0, link_bw=100.0, tflops=1000.0):
    """N GPUs connected through a switch. HBM local BW >> link BW."""
    hw = HardwareGraph()
    switch = Compute(name="switch")
    hw.add_node(switch)
    gpus = []
    hbms = []
    for i in range(n_gpus):
        gpu = Compute(name=f"gpu{i}", tflops={"bf16": tflops})
        hbm = Memory(name=f"hbm{i}", capacity_gb=80.0)
        hw.add_node(gpu)
        hw.add_node(hbm)
        hw.add_edge(FabricEdge(name=f"hbm{i}", src=gpu, dst=hbm,
                               src_to_dst_bandwidth_gbs=hbm_bw,
                               dst_to_src_bandwidth_gbs=hbm_bw,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name=f"link{i}", src=gpu, dst=switch,
                               src_to_dst_bandwidth_gbs=link_bw,
                               dst_to_src_bandwidth_gbs=link_bw,
                               is_full_duplex=True))
        gpus.append(gpu)
        hbms.append(hbm)
    return hw, gpus, hbms


class TestExplicitIdentityPlacement:
    def test_simulator_uses_explicit_comm_tensor_placement(self):
        """Comm participants come from explicitly placed tensor ports."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2)

        source = Concat()
        source.inputs = {"x": Tensor("bf16", (2,))}
        source.outputs = {"y": Tensor("bf16", (2,))}

        scatter = Scatter(total_bytes=4.0, world=2)
        scatter.inputs = {"x": Tensor("bf16", (2,))}
        scatter.outputs = {
            "o0": Tensor("bf16", (1,)),
            "o1": Tensor("bf16", (1,)),
        }

        sinks = []
        for _ in range(2):
            sink = Concat()
            sink.inputs = {"x": Tensor("bf16", (1,))}
            sink.outputs = {"y": Tensor("bf16", (1,))}
            sinks.append(sink)

        g = ComputeGraph()
        g.add_kernel(source)
        g.add_kernel(scatter)
        for sink in sinks:
            g.add_kernel(sink)
        g.add_data_edge(source, scatter, {"y": "x"})
        g.add_data_edge(scatter, sinks[0], {"o0": "x"})
        g.add_data_edge(scatter, sinks[1], {"o1": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(source, gpus[0])
        for rank, sink in enumerate(sinks):
            p.set_kernel_device(sink, gpus[rank])
        p.set_tensor_memory(scatter.inputs["x"], hbms[0])
        for rank in range(2):
            p.set_tensor_memory(scatter.outputs[f"o{rank}"], hbms[rank])
        optimize_comms(g)
        p.validate(g)

        result = _sim(g, p, hw)
        scatter_entries = [e for e in result.trace if e.kernel is scatter]
        assert {e.device for e in scatter_entries} == set(gpus)


class TestCrossDeviceFabric:
    def test_remote_read_uses_link_bandwidth(self):
        """Kernel on GPU1 reading tensor from GPU0 uses link BW, not HBM BW."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)
        # k1 on GPU0: produces 100KB output (trivial compute)
        t1_out = Tensor("bf16", (50000,))  # 100,000 bytes
        k1 = SyntheticKernel(flops_val=0.0, outputs={"y": t1_out})
        # k2 on GPU1: reads k1's output (remote), no local I/O
        t2_in = Tensor("bf16", (50000,))
        k2 = SyntheticKernel(flops_val=0.0, inputs={"x": t2_in})

        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_data_edge(k1, k2, {"y": "x"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpus[0])
        p.set_kernel_device(k2, gpus[1])
        result = _sim(g, p, hw)

        # k1: write 100KB at 1000 GB/s = 100000/(1000*1e3) = 0.1 us
        # k2: remote read 100KB at link 100 GB/s = 100000/(100*1e3) = 1.0 us
        # Total = k1 + k2 = 1.1 us
        assert result.total_time_us == pytest.approx(1.1, rel=1e-3)

    def test_parallel_remote_reads_share_fabric(self):
        """Two kernels on different GPUs reading same remote share link BW."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=3, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)
        # k1 on GPU0 produces two outputs
        t1_a = Tensor("bf16", (50000,))  # 100KB
        t1_b = Tensor("bf16", (50000,))  # 100KB
        k1 = SyntheticKernel(flops_val=0.0,
                             outputs={"a": t1_a, "b": t1_b})
        # k2 on GPU1, k3 on GPU2: each reads one output from GPU0
        t2_in = Tensor("bf16", (50000,))
        t3_in = Tensor("bf16", (50000,))
        k2 = SyntheticKernel(flops_val=0.0, inputs={"x": t2_in})
        k3 = SyntheticKernel(flops_val=0.0, inputs={"x": t3_in})

        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_kernel(k3)
        g.add_data_edge(k1, k2, {"a": "x"})
        g.add_data_edge(k1, k3, {"b": "x"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpus[0])
        p.set_kernel_device(k2, gpus[1], stream=0)
        p.set_kernel_device(k3, gpus[2], stream=0)
        result = _sim(g, p, hw)

        # k1: write 200KB at 1000 GB/s = 0.2 us
        # k2 and k3 run in parallel, both read through GPU0↔switch (shared)
        # Without sharing: each = 100KB / (100*1e3) = 1.0 us
        # With sharing (GPU0↔switch has 2 users): net_share = 0.5 → 2.0 us each
        # Total = k1(0.2) + max(k2, k3)(2.0) = 2.2 us
        assert result.total_time_us == pytest.approx(2.2, rel=1e-2)
        for entry in result.trace:
            if entry.kernel in (k2, k3):
                assert entry.network_elapsed_time_us == pytest.approx(2.0)

    def test_remote_write_uses_link_bandwidth(self):
        """A kernel writing output to remote memory uses link bandwidth."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)
        # Kernel on GPU0 writes output to HBM1 (remote).
        t_src = Tensor("bf16", (50000,))  # 100KB
        kernel = Kernel(
            inputs={"src0": t_src},
            outputs={"dst0": Tensor(t_src.dtype, t_src.shape)},
        )
        g = ComputeGraph()
        g.add_kernel(kernel)
        p = Placement(hardware=hw, graph=g)
        p.set_tensor_memory(kernel.outputs["dst0"], hbms[1])
        p.set_kernel_device(kernel, gpus[0])
        result = _sim(g, p, hw)

        # Read input locally: 100KB / (1000*1e3) = 0.1 us (mt)
        # Write output remotely: 100KB / (100*1e3) = 1.0 us (xfer)
        # Total = max(mt=0.1, xfer=1.0) = 1.0 us
        assert result.total_time_us == pytest.approx(1.0, rel=1e-3)

    def test_parallel_remote_writes_share_fabric(self):
        """Two kernels writing to the same remote HBM share link bandwidth."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=3, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)
        # k1 on GPU0 produces input for both moves
        t1_a = Tensor("bf16", (50000,))  # 100KB
        t1_b = Tensor("bf16", (50000,))  # 100KB
        k1 = SyntheticKernel(flops_val=0.0,
                             outputs={"a": t1_a, "b": t1_b})
        # k2 on GPU1 and k3 on GPU2 both write to HBM0 (remote).
        t_src1 = Tensor("bf16", (50000,))
        k2 = Kernel(
            inputs={"src0": t_src1},
            outputs={"dst0": Tensor(t_src1.dtype, t_src1.shape)},
        )
        t_src2 = Tensor("bf16", (50000,))
        k3 = Kernel(
            inputs={"src0": t_src2},
            outputs={"dst0": Tensor(t_src2.dtype, t_src2.shape)},
        )

        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_kernel(k3)
        g.add_data_edge(k1, k2, {"a": "src0"})
        g.add_data_edge(k1, k3, {"b": "src0"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpus[0])
        p.set_tensor_memory(k2.outputs["dst0"], hbms[0])
        p.set_tensor_memory(k3.outputs["dst0"], hbms[0])
        p.set_kernel_device(k2, gpus[1])
        p.set_kernel_device(k3, gpus[2])
        result = _sim(g, p, hw)

        # k1: write 200KB locally at 1000 GB/s = 0.2 us
        # move1/move2 run in parallel on different GPUs:
        #   Each reads 100KB via link0 'fwd' and writes 100KB via link0 'rev'
        #   Full-duplex: read/write parallel → base xfer = 1.0 us
        #   Both share link0 'fwd' (2 users) and link0 'rev' (2 users)
        #   worst_time = 2.0 us → net_share = 0.5 → effective = 2.0 us
        # Total = k1(0.2) + max(k2, k3)(2.0) = 2.2 us
        assert result.total_time_us == pytest.approx(2.2, rel=1e-2)


class TestCollectiveFabricSharing:
    def test_parallel_collectives_share_fabric(self):
        """Concurrent collectives on the same links divide fabric bandwidth."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)
        graph = ComputeGraph()
        placement = Placement(hardware=hw, graph=graph)
        scatters = []

        for stream in range(2):
            pred = SyntheticKernel(
                flops_val=0.0, outputs={"y": Tensor("bf16", (1,))})
            scatter = Scatter(total_bytes=200000.0, world=2)
            scatter.inputs = {"x": Tensor("bf16", (1,))}
            scatter.outputs = {
                "o0": Tensor("bf16", (1,)),
                "o1": Tensor("bf16", (1,)),
            }
            succ = SyntheticKernel(
                flops_val=0.0, inputs={"x": Tensor("bf16", (1,))})

            graph.add_kernel(pred)
            graph.add_kernel(scatter)
            graph.add_kernel(succ)
            graph.add_data_edge(pred, scatter, {"y": "x"})
            graph.add_data_edge(scatter, succ, {"o1": "x"})
            placement.set_kernel_device(pred, gpus[0], stream=stream)
            placement.set_kernel_device(succ, gpus[1], stream=stream)
            _place_comm_tensors(placement, scatter, hbms)
            scatters.append(scatter)

        result = _sim(graph, placement, hw)

        # Each scatter transfers (W-1)/W * 200 KB = 100 KB.  Its isolated
        # duration is 1 us at 100 GB/s; two concurrent scatters take 2 us.
        for scatter in scatters:
            entries = [e for e in result.trace if e.kernel is scatter]
            assert len(entries) == 2
            duration = entries[0].end_us - entries[0].start_us
            assert duration == pytest.approx(2.0, rel=1e-3)


class TestMultiStreamComm:
    """Tests for multi-stream collective comm blocking and retry."""

    def test_blocked_collective_does_not_stall_pending_compute(self):
        """A waiting collective must not head-of-line block its stream."""
        hw, gpus, hbms = _hw_multi_gpu(
            n_gpus=2, hbm_bw=1000.0, link_bw=100.0,
            tflops=1000.0)

        pred = SyntheticKernel(
            flops_val=0.0, outputs={"y": Tensor("bf16", (1,))})
        occupying_gpu0 = SyntheticKernel(flops_val=1e9)
        occupying_gpu1 = SyntheticKernel(flops_val=10e9)
        useful = SyntheticKernel(
            flops_val=2e9, inputs={"x": Tensor("bf16", (1,))})

        collective = AllReduce(
            total_bytes=100000.0, world=2, dtype="bf16")
        collective.inputs = {
            "i0": Tensor("bf16", (1,)),
            "i1": Tensor("bf16", (1,)),
        }
        collective.outputs = {
            "o0": Tensor("bf16", (1,)),
            "o1": Tensor("bf16", (1,)),
        }

        graph = ComputeGraph()
        for kernel in (
            pred, occupying_gpu0, occupying_gpu1, collective, useful,
        ):
            graph.add_kernel(kernel)
        # Add the collective edge first so it is ahead of useful in GPU0's
        # pending FIFO while occupying_gpu0 is running.
        graph.add_data_edge(pred, collective, {"y": "i0"})
        graph.add_data_edge(pred, useful, {"y": "x"})

        placement = Placement(hardware=hw, graph=graph)
        placement.set_kernel_device(pred, gpus[0])
        placement.set_kernel_device(occupying_gpu0, gpus[0])
        placement.set_kernel_device(occupying_gpu1, gpus[1])
        placement.set_kernel_device(useful, gpus[0])
        _place_comm_tensors(placement, collective, hbms)

        result = _sim(graph, placement, hw)

        useful_entry = next(
            entry for entry in result.trace if entry.kernel is useful)
        collective_entry = next(
            entry for entry in result.trace
            if entry.kernel is collective and entry.device is gpus[0])
        assert useful_entry.start_us == pytest.approx(1.0, abs=1e-4)
        assert useful_entry.end_us == pytest.approx(3.0, abs=1e-4)
        assert collective_entry.start_us == pytest.approx(10.0, abs=1e-4)

    def test_allreduce_blocks_all_participant_streams(self):
        """AllReduce waits when a participant stream is already active."""
        gpu0 = Compute(name="gpu0", tflops={"bf16": 1000.0})
        gpu1 = Compute(name="gpu1", tflops={"bf16": 1000.0})
        hbm0 = Memory(name="hbm0", capacity_gb=80.0)
        hbm1 = Memory(name="hbm1", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu0)
        hw.add_node(gpu1)
        hw.add_node(hbm0)
        hw.add_node(hbm1)
        hw.add_edge(FabricEdge(name="hbm0", src=gpu0, dst=hbm0,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="hbm1", src=gpu1, dst=hbm1,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="nvlink", src=gpu0, dst=gpu1,
                               src_to_dst_bandwidth_gbs=100.0,
                               dst_to_src_bandwidth_gbs=100.0,
                               is_full_duplex=True))

        # k_long on GPU1 (root, no connection to AR): 10 us compute
        # k_pred on GPU0 → AR → k_succ on GPU1
        # AR participants = [GPU0, GPU1]. k_pred finishes instantly.
        # AR tries to start at t≈0 but GPU1/stream0 is occupied → waits.
        # At t=10, k_long finishes → AR retries and starts.
        t_long_out = Tensor("bf16", (1,))
        k_long = SyntheticKernel(flops_val=10e9, outputs={"y": t_long_out})

        t0_out = Tensor("bf16", (1,))
        k_pred = SyntheticKernel(flops_val=0.0, outputs={"y": t0_out})

        ar = AllReduce(total_bytes=100000.0, world=2, dtype="bf16")
        ar.inputs = {"i0": Tensor("bf16", (1,)), "i1": Tensor("bf16", (1,))}
        ar.outputs = {"o0": Tensor("bf16", (1,)), "o1": Tensor("bf16", (1,))}

        t_succ_in = Tensor("bf16", (1,))
        k_succ = SyntheticKernel(flops_val=0.0, inputs={"x": t_succ_in})

        g = ComputeGraph()
        g.add_kernel(k_long)
        g.add_kernel(k_pred)
        g.add_kernel(ar)
        g.add_kernel(k_succ)
        g.add_data_edge(k_pred, ar, {"y": "i0"})
        g.add_data_edge(ar, k_succ, {"o1": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k_long, gpu1)
        p.set_kernel_device(k_pred, gpu0)
        p.set_kernel_device(k_succ, gpu1)
        _place_comm_tensors(p, ar, [hbm0, hbm1])

        result = _sim(g, p, hw)
        # AR starts at t≈10 (after k_long frees GPU1/stream0)
        # AR xfer = 100000/(100*1e3) = 1.0 us
        # Total ≈ 10 + 1 = 11 us
        assert result.total_time_us == pytest.approx(11.0, rel=0.1)
        ar_entries = [e for e in result.trace if e.kernel is ar]
        assert len(ar_entries) == 2
        assert ar_entries[0].start_us >= 9.9

    def test_pending_on_extra_participant_stream(self):
        """After multi-stream comm, pending kernels on participant streams resume."""
        gpu0 = Compute(name="gpu0", tflops={"bf16": 1000.0})
        gpu1 = Compute(name="gpu1", tflops={"bf16": 1000.0})
        hbm0 = Memory(name="hbm0", capacity_gb=80.0)
        hbm1 = Memory(name="hbm1", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu0)
        hw.add_node(gpu1)
        hw.add_node(hbm0)
        hw.add_node(hbm1)
        hw.add_edge(FabricEdge(name="hbm0", src=gpu0, dst=hbm0,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="hbm1", src=gpu1, dst=hbm1,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="nvlink", src=gpu0, dst=gpu1,
                               src_to_dst_bandwidth_gbs=100.0,
                               dst_to_src_bandwidth_gbs=100.0,
                               is_full_duplex=True))

        # k0(GPU0) → AR(GPU0,GPU1) → k_after(GPU1)
        # k_after should run on GPU1 after AR finishes (pending on stream)
        t0_out = Tensor("bf16", (1,))
        k0 = SyntheticKernel(flops_val=0.0, outputs={"y": t0_out})

        ar = AllReduce(total_bytes=100000.0, world=2, dtype="bf16")
        ar.inputs = {"i0": Tensor("bf16", (1,)), "i1": Tensor("bf16", (1,))}
        ar.outputs = {"o0": Tensor("bf16", (1,)), "o1": Tensor("bf16", (1,))}

        t_after = Tensor("bf16", (1,))
        k_after = SyntheticKernel(flops_val=1e9,
                                  inputs={"x": t_after})

        g = ComputeGraph()
        g.add_kernel(k0)
        g.add_kernel(ar)
        g.add_kernel(k_after)
        g.add_data_edge(k0, ar, {"y": "i0"})
        g.add_data_edge(ar, k_after, {"o1": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k0, gpu0)
        p.set_kernel_device(k_after, gpu1)
        _place_comm_tensors(p, ar, [hbm0, hbm1])

        result = _sim(g, p, hw)
        # AR: xfer = 100000/(100*1e3) = 1.0 us
        # k_after: 1e6 / (1000*1e6) = 1.0 us
        # Total ≈ 1.0 (AR) + 1.0 (k_after) = 2.0 us
        assert result.total_time_us == pytest.approx(2.0, rel=0.1)
        k_after_entry = [e for e in result.trace if e.kernel is k_after]
        assert len(k_after_entry) == 1
        assert k_after_entry[0].start_us >= 1.0 - 0.01


class TestLocalComm:
    """Tests for local single-participant communication kernels."""

    def test_local_comm_uses_memory_bandwidth(self):
        """CommKernel with all data in same memory uses local BW."""
        hw, gpu, hbm = _hw(read_bw=500.0, write_bw=500.0, tflops=1000.0)
        from rooflang.language.kernels.comm import Gather

        # k_pred produces 50KB output, gather collects it
        t_pred_out = Tensor("bf16", (25000,))  # 50KB
        k_pred = SyntheticKernel(flops_val=0.0, outputs={"y": t_pred_out})

        gather = Gather(total_bytes=100000.0, world=2)
        gather.inputs = {"i0": Tensor("bf16", (25000,)),
                         "i1": Tensor("bf16", (25000,))}
        gather.outputs = {"y": Tensor("bf16", (50000,))}

        t_succ_in = Tensor("bf16", (50000,))
        k_succ = SyntheticKernel(flops_val=0.0, inputs={"x": t_succ_in})

        g = ComputeGraph()
        g.add_kernel(k_pred)
        g.add_kernel(gather)
        g.add_kernel(k_succ)
        g.add_data_edge(k_pred, gather, {"y": "i0"})
        g.add_data_edge(gather, k_succ, {"y": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k_pred, gpu)
        p.set_kernel_device(k_succ, gpu)
        p.set_tensor_memory(k_pred.outputs["y"], hbm)
        p.set_tensor_memory(gather.outputs["y"], hbm)
        p.set_tensor_memory(k_succ.inputs["x"], hbm)
        _place_comm_tensors(p, gather, [hbm])

        result = _sim(g, p, hw)
        # Gather resolved to local: mt = max(total_bytes/read_bw, total_bytes/write_bw)
        # = max(100000/(500*1e3), 100000/(500*1e3)) = 0.2 us
        gather_entry = [e for e in result.trace if e.kernel is gather]
        assert len(gather_entry) == 1
        gather_time = gather_entry[0].end_us - gather_entry[0].start_us
        assert gather_time == pytest.approx(0.2, rel=0.1)


class TestNetSharePureCompute:
    """Tests for net_share behavior with compute-only kernels."""

    def test_compute_kernel_net_share_unaffected(self):
        """A pure-compute kernel gets net_share=1.0 even when fabric is busy."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)

        # k_net on GPU0 stream 0: reads 1MB from HBM1 (remote) → 10 us xfer
        # k_comp on GPU0 stream 1: 10 us pure compute, local-only data
        # They overlap and share dev_share=0.5, but k_comp's net_share stays 1.0
        t_remote = Tensor("bf16", (500000,))  # 1MB
        k_src = SyntheticKernel(flops_val=0.0, outputs={"y": t_remote})
        t_in = Tensor("bf16", (500000,))
        k_net = SyntheticKernel(flops_val=0.0, inputs={"x": t_in})

        t_local_in = Tensor("bf16", (500,))
        t_local_out = Tensor("bf16", (500,))
        k_comp = SyntheticKernel(flops_val=10e9,
                                 inputs={"x": t_local_in},
                                 outputs={"y": t_local_out})

        g = ComputeGraph()
        g.add_kernel(k_src)
        g.add_kernel(k_net)
        g.add_kernel(k_comp)
        g.add_data_edge(k_src, k_net, {"y": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k_src, gpus[1])
        p.set_kernel_device(k_net, gpus[0], stream=0)
        p.set_kernel_device(k_comp, gpus[0], stream=1)
        p.set_tensor_memory(t_remote, hbms[1])

        result = _sim(g, p, hw)
        # k_net: xfer=10 us (1MB at 100 GB/s). k_comp: ct=10 us.
        # Phase 1 (0-10 us): both overlap, dev_share=0.5. k_comp does 5/10 progress.
        # Phase 2 (10-15 us): k_net done, k_comp alone, dev_share=1.0 → 5 us remaining.
        # k_comp total = 15 us. k_comp has net_share=1.0 throughout (no fabric_keys).
        comp_entry = [e for e in result.trace if e.kernel is k_comp]
        assert len(comp_entry) == 1
        comp_time = comp_entry[0].end_us - comp_entry[0].start_us
        assert comp_time == pytest.approx(15.0, rel=0.01)
        assert comp_entry[0].local_elapsed_time_us == pytest.approx(
            15.0, rel=0.01)


class TestMultipleRemoteReadsAccumulate:
    """Tests that multiple remote reads through same link accumulate bytes."""

    def test_two_inputs_from_same_remote_share_link(self):
        """Reading two tensors from the same remote memory accumulates on one key."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)

        # k1 on GPU0 produces two outputs placed on HBM0
        t1 = Tensor("bf16", (50000,))  # 100KB
        t2 = Tensor("bf16", (50000,))  # 100KB
        k1 = SyntheticKernel(flops_val=0.0, outputs={"a": t1, "b": t2})

        # k2 on GPU1 reads both from HBM0
        t2_a = Tensor("bf16", (50000,))
        t2_b = Tensor("bf16", (50000,))
        k2 = SyntheticKernel(flops_val=0.0,
                             inputs={"x": t2_a, "y": t2_b})

        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_data_edge(k1, k2, {"a": "x", "b": "y"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpus[0])
        p.set_kernel_device(k2, gpus[1])

        result = _sim(g, p, hw)
        # k2 reads 200KB total through link0 (both tensors accumulate on same key)
        # xfer = max(per_link_time) = 200KB / (100*1e3) = 2.0 us
        k2_entry = [e for e in result.trace if e.kernel is k2]
        k2_time = k2_entry[0].end_us - k2_entry[0].start_us
        assert k2_time == pytest.approx(2.0, rel=0.01)


class TestInferDtype:
    """Test _infer_dtype static method coverage."""

    def test_sparse_attention_uses_main_compute_dtype(self):
        from rooflang.language.kernels.forward import DpskV4SparseAttn
        kernel = DpskV4SparseAttn(
            B=1, H=1, H_kv=1, S_q=1, k_sel=1, S_kv=1, Hd=1,
            dtype="fp8", q_dtype="bf16", kv_dtype="fp8",
            out_dtype="bf16")

        assert Simulator._infer_dtype(kernel) == "fp8"

    def test_infer_w_dtype(self):
        """Kernel with w_dtype uses it."""
        from rooflang.language.kernels.forward import Gemm
        gpu = Compute(name="gpu0", tflops={"bf16": 1000.0, "fp8e4m3": 2000.0})
        hbm = Memory(name="hbm", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu)
        hw.add_node(hbm)
        hw.add_edge(FabricEdge(name="hbm_link", src=gpu, dst=hbm,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        k = Gemm(M=128, N=64, K=32, w_dtype="fp8e4m3",
                 a_dtype="bf16", out_dtype="bf16")
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k, gpu)
        result = _sim(g, p, hw)
        assert len(result.trace) == 1


class TestInferCommDevices:
    """Tests for standard comm-device inference from tensor memory."""

    def test_infers_without_neighbor_kernel_placement(self):
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2)
        comm = AllReduce(total_bytes=2.0, world=2, dtype="bf16")
        comm.inputs = {"x": Tensor("bf16", (1,))}
        comm.outputs = {"y": Tensor("bf16", (1,))}
        graph = ComputeGraph()
        graph.add_kernel(comm)
        placement = Placement(hardware=hw, graph=graph)
        placement.set_tensor_memory(comm.inputs["x"], hbms[0])
        placement.set_tensor_memory(comm.outputs["y"], hbms[1])

        devices = Simulator(
            graph, placement, hw)._infer_comm_devices(comm)

        assert devices == gpus

    def test_comm_resolved_to_local_when_all_data_in_same_memory(self):
        """CommKernel between 2 GPUs resolves to local when all tensors in one memory."""
        gpu0 = Compute(name="gpu0", tflops={"bf16": 1000.0})
        gpu1 = Compute(name="gpu1", tflops={"bf16": 1000.0})
        hbm = Memory(name="hbm_shared", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu0)
        hw.add_node(gpu1)
        hw.add_node(hbm)
        hw.add_edge(FabricEdge(name="link0", src=gpu0, dst=hbm,
                               src_to_dst_bandwidth_gbs=500.0,
                               dst_to_src_bandwidth_gbs=500.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="link1", src=gpu1, dst=hbm,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))

        t_pred_out = Tensor("bf16", (50000,))  # 100KB
        k_pred = SyntheticKernel(flops_val=0.0, outputs={"y": t_pred_out})

        ar = AllReduce(total_bytes=200000.0, world=2, dtype="bf16")
        ar.inputs = {"i0": Tensor("bf16", (50000,)), "i1": Tensor("bf16", (50000,))}
        ar.outputs = {"o0": Tensor("bf16", (50000,)), "o1": Tensor("bf16", (50000,))}

        t_succ_in = Tensor("bf16", (50000,))
        k_succ = SyntheticKernel(flops_val=0.0, inputs={"x": t_succ_in})

        g = ComputeGraph()
        g.add_kernel(k_pred)
        g.add_kernel(ar)
        g.add_kernel(k_succ)
        g.add_data_edge(k_pred, ar, {"y": "i0"})
        g.add_data_edge(ar, k_succ, {"o0": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k_pred, gpu0)
        p.set_kernel_device(k_succ, gpu1)
        p.set_tensor_memory(t_pred_out, hbm)
        p.set_tensor_memory(k_succ.inputs["x"], hbm)
        _place_comm_tensors(p, ar, [hbm])

        result = _sim(g, p, hw)
        # AR resolved to local device (gpu1 has highest BW=1000 to hbm)
        # Local comm: mt = max(200000/(1000*1e3), 200000/(1000*1e3)) = 0.2 us
        ar_entry = [e for e in result.trace if e.kernel is ar]
        assert len(ar_entry) == 1
        ar_time = ar_entry[0].end_us - ar_entry[0].start_us
        assert ar_time == pytest.approx(0.2, rel=0.1)

    def test_comm_not_local_when_data_in_different_memories(self):
        """CommKernel stays remote when tensors are in different memories."""
        hw, gpus, hbms = _hw_multi_gpu(n_gpus=2, hbm_bw=1000.0,
                                       link_bw=100.0, tflops=1000.0)

        t_pred_out = Tensor("bf16", (50000,))
        k_pred = SyntheticKernel(flops_val=0.0, outputs={"y": t_pred_out})

        ar = AllReduce(total_bytes=200000.0, world=2, dtype="bf16")
        ar.inputs = {"i0": Tensor("bf16", (50000,)), "i1": Tensor("bf16", (50000,))}
        ar.outputs = {"o0": Tensor("bf16", (50000,)), "o1": Tensor("bf16", (50000,))}

        t_succ_in = Tensor("bf16", (50000,))
        k_succ = SyntheticKernel(flops_val=0.0, inputs={"x": t_succ_in})

        g = ComputeGraph()
        g.add_kernel(k_pred)
        g.add_kernel(ar)
        g.add_kernel(k_succ)
        g.add_data_edge(k_pred, ar, {"y": "i0"})
        g.add_data_edge(ar, k_succ, {"o0": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k_pred, gpus[0])
        p.set_kernel_device(k_succ, gpus[1])
        p.set_tensor_memory(t_pred_out, hbms[0])
        p.set_tensor_memory(k_succ.inputs["x"], hbms[1])
        _place_comm_tensors(p, ar, hbms)

        result = _sim(g, p, hw)
        ar_entries = [e for e in result.trace if e.kernel is ar]
        assert len(ar_entries) == 2


class TestHalfDuplexCollective:
    """Tests for collective comm over half-duplex links."""

    def test_allreduce_over_half_duplex_link(self):
        """AllReduce over a half-duplex link registers single direction key."""
        gpu0 = Compute(name="gpu0", tflops={"bf16": 1000.0})
        gpu1 = Compute(name="gpu1", tflops={"bf16": 1000.0})
        hbm0 = Memory(name="hbm0", capacity_gb=80.0)
        hbm1 = Memory(name="hbm1", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu0)
        hw.add_node(gpu1)
        hw.add_node(hbm0)
        hw.add_node(hbm1)
        hw.add_edge(FabricEdge(name="hbm0", src=gpu0, dst=hbm0,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="hbm1", src=gpu1, dst=hbm1,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="pcie", src=gpu0, dst=gpu1,
                               src_to_dst_bandwidth_gbs=50.0,
                               dst_to_src_bandwidth_gbs=50.0,
                               is_full_duplex=False))

        t0_out = Tensor("bf16", (1,))
        k_pred = SyntheticKernel(flops_val=0.0, outputs={"y": t0_out})

        ar = AllReduce(total_bytes=100000.0, world=2, dtype="bf16")
        ar.inputs = {"i0": Tensor("bf16", (1,)), "i1": Tensor("bf16", (1,))}
        ar.outputs = {"o0": Tensor("bf16", (1,)), "o1": Tensor("bf16", (1,))}

        t_succ_in = Tensor("bf16", (1,))
        k_succ = SyntheticKernel(flops_val=0.0, inputs={"x": t_succ_in})

        g = ComputeGraph()
        g.add_kernel(k_pred)
        g.add_kernel(ar)
        g.add_kernel(k_succ)
        g.add_data_edge(k_pred, ar, {"y": "i0"})
        g.add_data_edge(ar, k_succ, {"o0": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k_pred, gpu0)
        p.set_kernel_device(k_succ, gpu1)
        _place_comm_tensors(p, ar, [hbm0, hbm1])

        result = _sim(g, p, hw)
        # AR xfer = 100000 / (50*1e3) = 2.0 us
        ar_entries = [e for e in result.trace if e.kernel is ar]
        assert len(ar_entries) == 2
        ar_time = ar_entries[0].end_us - ar_entries[0].start_us
        assert ar_time == pytest.approx(2.0, rel=0.1)


class TestExtraParticipantStreamPending:
    """Tests for scheduling pending kernels on extra participant streams."""

    def test_pending_kernel_on_extra_stream_starts_after_comm(self):
        """A kernel pending on a non-primary participant stream resumes after comm."""
        gpu0 = Compute(name="gpu0", tflops={"bf16": 1000.0})
        gpu1 = Compute(name="gpu1", tflops={"bf16": 1000.0})
        hbm0 = Memory(name="hbm0", capacity_gb=80.0)
        hbm1 = Memory(name="hbm1", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(gpu0)
        hw.add_node(gpu1)
        hw.add_node(hbm0)
        hw.add_node(hbm1)
        hw.add_edge(FabricEdge(name="hbm0", src=gpu0, dst=hbm0,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="hbm1", src=gpu1, dst=hbm1,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="nvlink", src=gpu0, dst=gpu1,
                               src_to_dst_bandwidth_gbs=100.0,
                               dst_to_src_bandwidth_gbs=100.0,
                               is_full_duplex=True))

        t_pred_out = Tensor("bf16", (1,))
        k_pred = SyntheticKernel(flops_val=0.0, outputs={"y": t_pred_out})

        ar = AllReduce(total_bytes=200000.0, world=2, dtype="bf16")
        ar.inputs = {"i0": Tensor("bf16", (1,)), "i1": Tensor("bf16", (1,))}
        ar.outputs = {"o0": Tensor("bf16", (1,)), "o1": Tensor("bf16", (1,))}

        # AR successor on GPU1 so AR resolves with participants=[gpu0, gpu1]
        t_ar_succ_in = Tensor("bf16", (1,))
        k_ar_succ = SyntheticKernel(flops_val=0.0, inputs={"x": t_ar_succ_in})

        t_indep_out = Tensor("bf16", (1,))
        k_indep_pred = SyntheticKernel(flops_val=1e9,
                                       outputs={"y": t_indep_out})

        t_pending_in = Tensor("bf16", (1,))
        k_pending = SyntheticKernel(flops_val=1e9,
                                    inputs={"x": t_pending_in})

        g = ComputeGraph()
        g.add_kernel(k_pred)
        g.add_kernel(ar)
        g.add_kernel(k_ar_succ)
        g.add_kernel(k_indep_pred)
        g.add_kernel(k_pending)
        g.add_data_edge(k_pred, ar, {"y": "i0"})
        g.add_data_edge(ar, k_ar_succ, {"o1": "x"})
        g.add_data_edge(k_indep_pred, k_pending, {"y": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k_pred, gpu0)
        p.set_kernel_device(k_ar_succ, gpu1)
        p.set_kernel_device(k_indep_pred, gpu1, stream=1)
        p.set_kernel_device(k_pending, gpu1, stream=0)
        _place_comm_tensors(p, ar, [hbm0, hbm1])

        result = _sim(g, p, hw)
        # AR xfer = 200000/(100*1e3) = 2 us. AR blocks GPU1/stream0.
        # k_indep_pred finishes at t=1, k_pending tries GPU1/stream0 → pending.
        # AR finishes at t=2 → extra_keys fires → k_pending starts at t=2.
        pending_entry = [e for e in result.trace if e.kernel is k_pending]
        assert len(pending_entry) == 1
        assert pending_entry[0].start_us == pytest.approx(2.0, rel=0.1)
        assert pending_entry[0].end_us == pytest.approx(3.0, rel=0.1)
