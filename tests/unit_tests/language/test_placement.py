"""Unit tests for rooflang.language.placement (Placement + DeviceAssignment)."""

import pytest

from rooflang.language.placement import Placement, DeviceAssignment
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import AllReduce, Scatter
from rooflang.language.kernels.forward import Slice
from rooflang.language.kernels.identity import Concat, Spawn
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

    def test_spawn_outputs_alias_remote_predecessor_memory(self):
        hw, gpu0, hbm0 = _simple_hw()
        gpu1 = Compute(name="gpu1", tflops={"bf16": 2250.0})
        hbm1 = Memory(name="hbm1", capacity_gb=288.0)
        hw.add_node(gpu1)
        hw.add_node(hbm1)
        hw.add_edge(FabricEdge(
            name="hbm1", src=gpu1, dst=hbm1,
            src_to_dst_bandwidth_gbs=7750.0,
            dst_to_src_bandwidth_gbs=7750.0,
            is_full_duplex=False,
        ))

        pred = Kernel(outputs={"y": Tensor("bf16", (4,))})
        spawn = Spawn(world=2)
        spawn.inputs = {"x": Tensor("bf16", (4,))}
        spawn.outputs = {
            "o0": Tensor("bf16", (4,)),
            "o1": Tensor("bf16", (4,)),
        }
        graph = ComputeGraph()
        graph.add_kernel(pred)
        graph.add_kernel(spawn)
        graph.add_data_edge(pred, spawn, {"y": "x"})

        placement = Placement(hardware=hw, graph=graph)
        placement.set_kernel_device(pred, gpu0)
        placement.set_kernel_device(spawn, gpu1)

        assert all(
            placement.get_tensor_memory(tensor) is hbm0
            for tensor in (*spawn.inputs.values(), *spawn.outputs.values())
        )
        placement.validate(graph)


# ── Placement.get_kernel_device tests ────────────────────────────────


class TestPlacementGetKernelDevice:
    def test_unplaced_raises(self):
        p = Placement()
        with pytest.raises(KeyError, match="Kernel not placed"):
            p.get_kernel_device(Kernel())


class TestPlacementDeviceInference:
    def test_infer_comm_devices_requires_hardware(self):
        with pytest.raises(ValueError, match="without hardware"):
            Placement().infer_comm_devices(AllReduce(8.0, 1))

    def test_infer_comm_devices_requires_placed_ports(self):
        hw, _, _ = _simple_hw()
        comm = AllReduce(8.0, 1)
        comm.inputs = {"x": Tensor("bf16", (4,))}

        with pytest.raises(ValueError, match="tensor has no memory"):
            Placement(hardware=hw).infer_comm_devices(comm)

    def test_infer_comm_devices_requires_ports(self):
        hw, _, _ = _simple_hw()

        with pytest.raises(ValueError, match="no input/output tensors"):
            Placement(hardware=hw).infer_comm_devices(AllReduce(8.0, 1))

    def test_get_tensor_device_errors_and_success(self):
        tensor = Tensor("bf16", (4,))
        with pytest.raises(ValueError, match="without hardware"):
            Placement().get_tensor_device(tensor)

        hw, gpu, hbm = _simple_hw()
        placement = Placement(hardware=hw)
        with pytest.raises(ValueError, match="no memory placement"):
            placement.get_tensor_device(tensor)
        placement.set_tensor_memory(tensor, hbm)
        assert placement.get_tensor_device(tensor) is gpu


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


class TestMemoryFootprints:
    def test_record_memory_footprint_aggregates_by_memory_and_role(self):
        memory = Memory(name="hbm", capacity_gb=1.0)
        placement = Placement()

        placement.record_memory_footprint(memory, 1024.0, "kv_cache")
        placement.record_memory_footprint(memory, 512.0, "kv_cache")

        footprint, = placement.memory_footprints
        assert footprint.memory is memory
        assert footprint.size_bytes == 1536.0
        assert footprint.role == "kv_cache"

    def test_zero_is_ignored_and_negative_raises(self):
        memory = Memory(name="hbm", capacity_gb=1.0)
        placement = Placement()

        placement.record_memory_footprint(memory, 0.0, "unused")
        assert not placement.memory_footprints
        with pytest.raises(ValueError, match="non-negative"):
            placement.record_memory_footprint(memory, -1.0, "invalid")


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
        ar = AllReduce(total_bytes=1024.0, world=4, dtype="bf16")
        g.add_kernel(k)
        g.add_kernel(ar)
        p.set_kernel_device(k, gpu)
        p.validate(g)

    def test_unplaced_comm_kernel_tensor_memory_is_checked(self):
        comm = AllReduce(total_bytes=8.0, world=2, dtype="bf16")
        comm.inputs = {"i0": Tensor("bf16", (4,))}
        comm.outputs = {"o0": Tensor("bf16", (4,))}
        graph = ComputeGraph()
        graph.add_kernel(comm)

        with pytest.raises(ValueError, match="input.*has no memory"):
            Placement().validate(graph)

    @pytest.mark.parametrize("port", ["weight", "output"])
    def test_missing_weight_or_output_memory_raises(self, port):
        tensor = Tensor("bf16", (4,))
        kwargs = {"weights" if port == "weight" else "outputs": {"x": tensor}}
        kernel = Kernel(**kwargs)
        graph = ComputeGraph()
        graph.add_kernel(kernel)
        placement = Placement()
        placement.set_kernel_device(kernel, Compute(name="gpu0"))

        with pytest.raises(ValueError, match=rf"{port} of.*has no memory"):
            placement.validate(graph)

    @pytest.mark.parametrize(
        "kernel", [Spawn(world=1), Concat(), Slice()],
        ids=["spawn", "concat", "slice"],
    )
    def test_same_memory_kernel_tensors_in_same_memory_pass(self, kernel):
        hw, gpu, _ = _simple_hw()
        kernel.inputs = {"x": Tensor("bf16", (4,))}
        kernel.outputs = {"y": Tensor("bf16", (4,))}
        g = ComputeGraph()
        g.add_kernel(kernel)
        p = Placement(hardware=hw, graph=g)
        p.set_kernel_device(kernel, gpu)
        p.validate(g)

    def test_same_memory_kernel_tensor_without_memory_raises(self):
        kernel = Slice()
        kernel.inputs = {"x": Tensor("bf16", (4,))}
        kernel.outputs = {"y": Tensor("bf16", (4,))}
        g = ComputeGraph()
        g.add_kernel(kernel)
        p = Placement(graph=g)
        p.set_kernel_device(kernel, Compute(name="gpu0"))

        with pytest.raises(ValueError, match="Slice.*has no memory"):
            p.validate(g)

    def test_same_memory_kernel_tensors_in_different_memories_raise(self):
        hw, gpu, hbm = _simple_hw()
        other_hbm = Memory(name="hbm1", capacity_gb=288.0)
        kernel = Slice()
        kernel.inputs = {"x": Tensor("bf16", (4,))}
        kernel.outputs = {"y": Tensor("bf16", (4,))}
        g = ComputeGraph()
        g.add_kernel(kernel)
        p = Placement(hardware=hw, graph=g)
        p.set_tensor_memory(kernel.inputs["x"], hbm)
        p.set_tensor_memory(kernel.outputs["y"], other_hbm)
        p.set_kernel_device(kernel, gpu)

        with pytest.raises(ValueError, match="must share one memory"):
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

    def test_output_preserves_explicit_placement(self):
        hw, gpu, hbm = _simple_hw()
        nvme = Memory(name="nvme", capacity_gb=3840.0)
        inputs = [Tensor("bf16", (1024,)), Tensor("bf16", (2048,))]
        kernel = Kernel(
            inputs={f"src{i}": tensor
                    for i, tensor in enumerate(inputs)},
            outputs={f"dst{i}": Tensor(tensor.dtype, tensor.shape)
                     for i, tensor in enumerate(inputs)},
        )
        placement = Placement(hardware=hw)

        placement.set_tensor_memory(kernel.outputs["dst0"], hbm)
        placement.set_tensor_memory(kernel.outputs["dst1"], nvme)
        placement.set_kernel_device(kernel, gpu)

        assert placement.get_tensor_memory(kernel.outputs["dst0"]) is hbm
        assert placement.get_tensor_memory(kernel.outputs["dst1"]) is nvme

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

    def test_kernel_placement_does_not_fill_adjacent_comm_ports(self):
        hw, gpu, _ = _simple_hw()
        source = Kernel(outputs={"y": Tensor("bf16", (4,))})
        scatter = Scatter(total_bytes=8.0, world=1)
        scatter.inputs = {"x": Tensor("bf16", (4,))}
        scatter.outputs = {"o0": Tensor("bf16", (4,))}
        graph = ComputeGraph()
        graph.add_kernel(source)
        graph.add_kernel(scatter)
        graph.add_data_edge(source, scatter, {"y": "x"})
        placement = Placement(hardware=hw, graph=graph)

        placement.set_kernel_device(source, gpu)

        assert placement.get_tensor_memory(scatter.inputs["x"]) is None
        with pytest.raises(ValueError, match="input.*has no memory"):
            placement.validate(graph)

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

    def test_weights_skip_if_set(self):
        hw, gpu, hbm = _simple_hw()
        nvme = Memory(name="nvme", capacity_gb=3840.0)
        t_w = Tensor("bf16", (4096,))
        k = Kernel(weights={"W": t_w})
        p = Placement(hardware=hw)
        p.set_tensor_memory(t_w, nvme)
        p.set_kernel_device(k, gpu)
        assert p.get_tensor_memory(t_w) is nvme

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
