"""Focused tests for model-specific placement helper edge cases."""

from types import SimpleNamespace

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import AllReduce, Gather, Scatter
from rooflang.language.kernels.identity import Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.programs.models.dsv4_flash import optimization as dsv4_flash_opt
from rooflang.programs.models.glm_5_2_fp8 import optimization as glm_opt
from rooflang.programs.models import dsv4_flash, glm_5_2_fp8, kimi_k3
from rooflang.programs.experiments import visualization
from rooflang.programs.presets.b300 import B300Cluster, B300SuperChip


OPTIMIZATIONS = [
    pytest.param(dsv4_flash_opt, id="dsv4_flash"),
    pytest.param(glm_opt, id="glm_5_2_fp8"),
]


def _tensor():
    return Tensor("bf16", (4,))


@pytest.mark.parametrize("optimization", OPTIMIZATIONS)
def test_comm_memory_helper_handles_collective_chains(optimization):
    """Exercise comm-to-comm, compute-to-comm, and fan-out propagation."""
    graph = ComputeGraph()
    producer = Kernel(outputs={"y": _tensor()})
    scatter = Scatter(total_bytes=32, world=2)
    scatter.inputs = {"x": _tensor()}
    scatter.outputs = {"o0": _tensor(), "o1": _tensor()}
    gather = Gather(total_bytes=32, world=2)
    gather.inputs = {"i0": _tensor(), "i1": _tensor()}
    gather.outputs = {"y": _tensor()}
    consumer = Kernel(inputs={"x": _tensor()})
    passthrough = AllReduce(total_bytes=32, world=2)
    passthrough.inputs = {"i0": _tensor(), "i1": _tensor()}
    passthrough.outputs = {"o0": _tensor(), "o1": _tensor()}

    for kernel in (producer, scatter, gather, consumer, passthrough):
        graph.add_kernel(kernel)
    graph.add_data_edge(producer, scatter, {"y": "x"})
    graph.add_data_edge(scatter, gather, {"o0": "i0"})
    graph.add_data_edge(scatter, gather, {"o1": "i1"})
    graph.add_data_edge(gather, consumer, {"y": "x"})

    memory = Memory(name="hbm-test", capacity_gb=1.0, kind="hbm")
    placement = Placement(graph=graph)
    placement.set_tensor_memory(producer.outputs["y"], memory)
    placement.set_tensor_memory(scatter.outputs["o0"], memory)
    placement.set_tensor_memory(scatter.outputs["o1"], memory)
    placement.set_tensor_memory(consumer.inputs["x"], memory)

    optimization._place_comm_tensor_memories(graph, placement)

    assert placement.get_tensor_memory(scatter.inputs["x"]) is memory
    assert placement.get_tensor_memory(gather.inputs["i0"]) is memory
    assert placement.get_tensor_memory(gather.inputs["i1"]) is memory
    assert placement.get_tensor_memory(gather.outputs["y"]) is memory


@pytest.mark.parametrize("optimization", OPTIMIZATIONS)
def test_pp_boundary_helper_places_all_aliases(optimization):
    graph = ComputeGraph()
    predecessor = Kernel(outputs={"y": _tensor()})
    spawn = Spawn(world=2)
    spawn.inputs = {"x": _tensor()}
    spawn.outputs = {"o": _tensor()}
    successor = Kernel(inputs={"x": _tensor()})
    for kernel in (predecessor, spawn, successor):
        graph.add_kernel(kernel)
    graph.add_data_edge(predecessor, spawn, {"y": "x"})
    graph.add_data_edge(spawn, successor, {"o": "x"})

    memory = Memory(name="boundary-hbm", capacity_gb=1.0, kind="hbm")
    placement = Placement(graph=graph)
    optimization._place_pp_boundary_spawn(graph, placement, spawn, memory)

    assert all(
        placement.get_tensor_memory(tensor) is memory
        for tensor in (
            predecessor.outputs["y"], spawn.inputs["x"],
            spawn.outputs["o"], successor.inputs["x"],
        )
    )


@pytest.mark.parametrize("optimization", OPTIMIZATIONS)
def test_expert_helper_supports_replicated_experts(optimization, monkeypatch):
    monkeypatch.setattr(optimization, "N_EXPERTS", 1)
    up = Kernel(inputs={"x": _tensor()})
    down = Kernel(inputs={"x": _tensor()})
    graph = ComputeGraph()
    graph.add_kernel(up)
    graph.add_kernel(down)
    layer = SimpleNamespace(experts=[up, down])
    hardware = B300Cluster(n_nodes=1)
    gpu = next(
        component for component in hardware.nodes
        if isinstance(component, Compute) and component.kind == "gpu"
    )
    placement = Placement(hardware=hardware, graph=graph)

    optimization._place_experts_and_routes(
        graph, layer, {}, [gpu], placement, hardware, shard_experts=False)

    assert placement.get_kernel_device(up).device is gpu
    assert placement.get_kernel_device(down).device is gpu


@pytest.mark.parametrize(
    "case",
    ["partition", "cp_ep", "pp_gpus", "experts", "batch", "sequence",
     "window", "compression", "compressed_cp"],
)
def test_dsv4_flash_validation_rejects_each_constraint(case, monkeypatch):
    monkeypatch.setattr(dsv4_flash_opt, "N_EXPERTS", 8)
    monkeypatch.setattr(dsv4_flash_opt, "COMPRESS_RATIOS", (4, 128))
    monkeypatch.setattr(dsv4_flash_opt, "WINDOW", 128)
    kwargs = dict(
        layers=[object()], batch_size=64, seq_prefill=512,
        is_prefill=True, cp=2, dp=4, ep=8, pp_partition=[1], n_gpus=8,
    )
    if case == "partition":
        kwargs.update(layers=[object(), object()], pp_partition=[1])
    elif case == "cp_ep":
        kwargs.update(cp=2, dp=2, ep=8)
    elif case == "pp_gpus":
        kwargs.update(n_gpus=16)
    elif case == "experts":
        kwargs.update(cp=1, dp=3, ep=3, n_gpus=3)
    elif case == "batch":
        kwargs.update(batch_size=62)
    elif case == "sequence":
        kwargs.update(seq_prefill=511)
    elif case == "window":
        monkeypatch.setattr(dsv4_flash_opt, "WINDOW", 130)
        kwargs.update(cp=4, dp=2, ep=8)
    elif case == "compression":
        monkeypatch.setattr(dsv4_flash_opt, "COMPRESS_RATIOS", (3,))
    elif case == "compressed_cp":
        monkeypatch.setattr(dsv4_flash_opt, "COMPRESS_RATIOS", (4,))
        kwargs.update(seq_prefill=520, cp=8, dp=1, ep=8)

    with pytest.raises(ValueError):
        dsv4_flash_opt._validate_args(**kwargs)


@pytest.mark.parametrize(
    "case", ["partition", "cp_ep", "pp_gpus", "experts", "batch",
              "sequence", "index_topk"],
)
def test_glm_validation_rejects_each_constraint(case, monkeypatch):
    monkeypatch.setattr(glm_opt, "N_EXPERTS", 8)
    monkeypatch.setattr(glm_opt, "INDEX_TOPK", 2048)
    kwargs = dict(
        layers=[object()], batch_size=64, seq_prefill=512,
        is_prefill=True, cp=2, dp=4, ep=8, pp_partition=[1], n_gpus=8,
    )
    if case == "partition":
        kwargs.update(layers=[object(), object()], pp_partition=[1])
    elif case == "cp_ep":
        kwargs.update(cp=2, dp=2, ep=8)
    elif case == "pp_gpus":
        kwargs.update(n_gpus=16)
    elif case == "experts":
        kwargs.update(cp=1, dp=3, ep=3, n_gpus=3)
    elif case == "batch":
        kwargs.update(batch_size=62)
    elif case == "sequence":
        kwargs.update(seq_prefill=511)
    elif case == "index_topk":
        monkeypatch.setattr(glm_opt, "INDEX_TOPK", 130)
        kwargs.update(cp=4, dp=2, ep=8)

    with pytest.raises(ValueError):
        glm_opt._validate_args(**kwargs)


@pytest.mark.parametrize("optimization", OPTIMIZATIONS)
def test_superchip_optimizer_places_a_minimal_graph(optimization):
    graph = ComputeGraph()
    kernel = Kernel()
    graph.add_kernel(kernel)

    _, placement = optimization.optimize_model_superchip(
        graph, B300SuperChip())

    assert placement.get_kernel_device(kernel).device.kind == "gpu"


@pytest.mark.parametrize(
    "model_package", [dsv4_flash, glm_5_2_fp8, kimi_k3],
    ids=["dsv4_flash", "glm_5_2_fp8", "kimi_k3"],
)
def test_model_package_lazy_visualization_attribute(model_package):
    assert model_package.visualize_layer is visualization.visualize_layer
    with pytest.raises(AttributeError):
        getattr(model_package, "_definitely_missing")
