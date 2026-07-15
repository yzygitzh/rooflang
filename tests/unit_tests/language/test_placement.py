"""Unit tests for rooflang.language.placement (Placement + DeviceAssignment)."""

import pytest

from rooflang.language.placement import Placement, DeviceAssignment
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import AllReduce
from rooflang.language.kernels.identity import Move
from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.tensor import Tensor


# ── DeviceAssignment tests ───────────────────────────────────────────


class TestDeviceAssignment:
    def test_basic(self):
        gpu = Compute(name="gpu0", tflops={"bf16": 2250.0})
        da = DeviceAssignment(device=gpu, stream=1, resource_cap=0.5)
        assert da.device is gpu
        assert da.stream == 1
        assert da.resource_cap == 0.5

    def test_defaults(self):
        gpu = Compute(name="gpu0")
        da = DeviceAssignment(device=gpu)
        assert da.stream == 0
        assert da.resource_cap == 1.0


# ── Placement.set_kernel_device tests ────────────────────────────────


class TestPlacementSetKernelDevice:
    def test_basic(self):
        p = Placement()
        gpu = Compute(name="gpu0")
        k = Kernel()
        p.set_kernel_device(k, gpu, stream=2, resource_cap=0.7)
        da = p.get_kernel_device(k)
        assert da.device is gpu
        assert da.stream == 2
        assert da.resource_cap == 0.7

    def test_default_stream_and_cap(self):
        p = Placement()
        gpu = Compute(name="gpu0")
        k = Kernel()
        p.set_kernel_device(k, gpu)
        da = p.get_kernel_device(k)
        assert da.stream == 0
        assert da.resource_cap == 1.0

    def test_overwrite(self):
        p = Placement()
        gpu0 = Compute(name="gpu0")
        gpu1 = Compute(name="gpu1")
        k = Kernel()
        p.set_kernel_device(k, gpu0, stream=0)
        p.set_kernel_device(k, gpu1, stream=1)
        assert p.get_kernel_device(k).device is gpu1
        assert p.get_kernel_device(k).stream == 1

    def test_resource_cap_zero_raises(self):
        p = Placement()
        with pytest.raises(ValueError, match="resource_cap must be in"):
            p.set_kernel_device(Kernel(), Compute(name="g"), resource_cap=0.0)

    def test_resource_cap_negative_raises(self):
        p = Placement()
        with pytest.raises(ValueError, match="resource_cap must be in"):
            p.set_kernel_device(Kernel(), Compute(name="g"), resource_cap=-0.1)

    def test_resource_cap_above_one_raises(self):
        p = Placement()
        with pytest.raises(ValueError, match="resource_cap must be in"):
            p.set_kernel_device(Kernel(), Compute(name="g"), resource_cap=1.01)

    def test_resource_cap_one_ok(self):
        p = Placement()
        k = Kernel()
        p.set_kernel_device(k, Compute(name="g"), resource_cap=1.0)
        assert p.get_kernel_device(k).resource_cap == 1.0


# ── Placement.get_kernel_device tests ────────────────────────────────


class TestPlacementGetKernelDevice:
    def test_unplaced_raises(self):
        p = Placement()
        with pytest.raises(KeyError, match="Kernel not placed"):
            p.get_kernel_device(Kernel())


# ── Placement.placed_kernels tests ───────────────────────────────────


class TestPlacedKernels:
    def test_empty(self):
        assert Placement().placed_kernels == frozenset()

    def test_after_set(self):
        p = Placement()
        k1, k2 = Kernel(), Kernel()
        gpu = Compute(name="gpu0")
        p.set_kernel_device(k1, gpu)
        p.set_kernel_device(k2, gpu, stream=1)
        assert p.placed_kernels == frozenset({k1, k2})


# ── Placement.validate tests ─────────────────────────────────────────


class TestPlacementValidate:
    def test_all_placed_passes(self):
        hw, gpu, hbm = _simple_hw()
        p = Placement(hardware=hw)
        g = ComputeGraph()
        k1 = Kernel(inputs={"x": Tensor("bf16", (4,))},
                    outputs={"y": Tensor("bf16", (4,))})
        k2 = Kernel(inputs={"a": Tensor("bf16", (4,))},
                    outputs={"b": Tensor("bf16", (4,))})
        g.add_kernel(k1)
        g.add_kernel(k2)
        p.set_kernel_device(k1, gpu)
        p.set_kernel_device(k2, gpu, stream=1)
        p.validate(g)

    def test_unplaced_raises(self):
        hw, gpu, hbm = _simple_hw()
        p = Placement(hardware=hw)
        g = ComputeGraph()
        k1 = Kernel()
        k2 = Kernel()
        g.add_kernel(k1)
        g.add_kernel(k2)
        p.set_kernel_device(k1, gpu)
        with pytest.raises(ValueError, match="Unplaced kernels"):
            p.validate(g)

    def test_extraneous_raises(self):
        hw, gpu, hbm = _simple_hw()
        p = Placement(hardware=hw)
        g = ComputeGraph()
        k1 = Kernel()
        k2 = Kernel()
        g.add_kernel(k1)
        p.set_kernel_device(k1, gpu)
        p.set_kernel_device(k2, gpu)
        with pytest.raises(ValueError, match="Extraneous placements"):
            p.validate(g)

    def test_empty_graph_passes(self):
        Placement().validate(ComputeGraph())

    def test_comm_kernel_skipped(self):
        hw, gpu, hbm = _simple_hw()
        p = Placement(hardware=hw)
        g = ComputeGraph()
        k = Kernel(outputs={"y": Tensor("bf16", (4,))})
        ar = AllReduce(bytes_per_rank=1024.0, world=4, dtype="bf16")
        g.add_kernel(k)
        g.add_kernel(ar)
        p.set_kernel_device(k, gpu)
        p.validate(g)


# ── Helpers ──────────────────────────────────────────────────────────


def _simple_hw():
    """GPU + HBM connected by a FabricEdge."""
    gpu = Compute(name="gpu0", tflops={"bf16": 2250.0})
    hbm = Memory(name="hbm0", capacity_gb=288.0)
    hw = HardwareGraph()
    hw.add_node(gpu)
    hw.add_node(hbm)
    hw.add_edge(FabricEdge(name="hbm", src=gpu, dst=hbm,
                           src_to_dst_bandwidth_gbs=7750.0,
                           dst_to_src_bandwidth_gbs=7750.0,
                           is_full_duplex=False))
    return hw, gpu, hbm


# ── Placement auto-assign tensor memory tests ────────────────────────


class TestPlacementTensorMemory:
    def test_assigns_memory_on_set(self):
        hw, gpu, hbm = _simple_hw()
        t_in = Tensor("bf16", (1024,))
        t_w = Tensor("bf16", (4096, 4096))
        t_out = Tensor("bf16", (1024,))
        k = Kernel(inputs={"x": t_in}, weights={"W": t_w},
                   outputs={"y": t_out})
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        assert p.get_tensor_memory(t_in) is hbm
        assert p.get_tensor_memory(t_w) is hbm
        assert p.get_tensor_memory(t_out) is hbm

    def test_output_follows_move_dst(self):
        hw, gpu, hbm = _simple_hw()
        nvme = Memory(name="nvme", capacity_gb=3840.0)
        t_src = Tensor("bf16", (1024,))
        m = Move(t_src, dst_location=nvme)
        p = Placement(hardware=hw)
        p.set_kernel_device(m, gpu)
        assert p.get_tensor_memory(m.outputs["dst"]) is nvme
        assert p.get_tensor_memory(t_src) is hbm

    def test_input_follows_predecessor_output(self):
        hw, gpu, hbm = _simple_hw()
        nvme = Memory(name="nvme", capacity_gb=3840.0)
        hw.add_node(nvme)
        hw.add_edge(FabricEdge(name="pcie", src=gpu, dst=nvme,
                               src_to_dst_bandwidth_gbs=14.0,
                               dst_to_src_bandwidth_gbs=7.0,
                               is_full_duplex=True))

        t_mid = Tensor("bf16", (1024,))
        k1 = Kernel(outputs={"y": t_mid})
        t_in2 = Tensor("bf16", (1024,))
        k2 = Kernel(inputs={"x": t_in2}, outputs={"z": Tensor("bf16", (1024,))})

        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_data_edge(k1, k2, {"y": "x"})

        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpu)
        p.set_kernel_device(k2, gpu)
        # k2's input "x" should follow k1's output "y" memory (hbm)
        assert p.get_tensor_memory(t_in2) is hbm

    def test_set_tensor_memory_overrides(self):
        hw, gpu, hbm = _simple_hw()
        nvme = Memory(name="nvme", capacity_gb=3840.0)
        t_in = Tensor("bf16", (1024,))
        t_out = Tensor("bf16", (1024,))
        k = Kernel(inputs={"x": t_in}, outputs={"y": t_out})
        p = Placement(hardware=hw)
        p.set_tensor_memory(t_in, nvme)
        p.set_tensor_memory(t_out, nvme)
        p.set_kernel_device(k, gpu)
        assert p.get_tensor_memory(t_in) is nvme
        assert p.get_tensor_memory(t_out) is nvme

    def test_weights_always_reassigned(self):
        hw, gpu, hbm = _simple_hw()
        nvme = Memory(name="nvme", capacity_gb=3840.0)
        t_w = Tensor("bf16", (4096,))
        k = Kernel(weights={"W": t_w})
        p = Placement(hardware=hw)
        p.set_tensor_memory(t_w, nvme)
        p.set_kernel_device(k, gpu)
        assert p.get_tensor_memory(t_w) is hbm

    def test_no_hardware_skips(self):
        t_in = Tensor("bf16", (1024,))
        k = Kernel(inputs={"x": t_in})
        p = Placement()
        p.set_kernel_device(k, Compute(name="gpu0"))
        assert p.get_tensor_memory(t_in) is None

    def test_validate_tensor_completeness(self):
        hw, gpu, hbm = _simple_hw()
        g = ComputeGraph()
        t_in = Tensor("bf16", (4,))
        k = Kernel(inputs={"x": t_in})
        g.add_kernel(k)
        p = Placement()
        p.set_kernel_device(k, gpu)
        with pytest.raises(ValueError, match="has no memory"):
            p.validate(g)

    def test_validate_edge_consistency(self):
        hw, gpu, hbm = _simple_hw()
        nvme = Memory(name="nvme", capacity_gb=3840.0)
        t_out = Tensor("bf16", (4,))
        t_in2 = Tensor("bf16", (4,))
        k1 = Kernel(outputs={"y": t_out})
        k2 = Kernel(inputs={"x": t_in2})
        g = ComputeGraph()
        g.add_kernel(k1)
        g.add_kernel(k2)
        g.add_data_edge(k1, k2, {"y": "x"})
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(k1, gpu)
        # Manually override k2's input to a different memory
        p.set_tensor_memory(t_in2, nvme)
        p.set_kernel_device(k2, gpu)
        with pytest.raises(ValueError, match="Memory mismatch"):
            p.validate(g)
