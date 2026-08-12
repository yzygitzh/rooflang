"""Tests for DeepSeek V4 Pro placement strategies."""

import inspect

import pytest

from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.comm import Broadcast, ReduceScatter
from rooflang.language.kernels.forward import Nop, ReadInput, Slice, SparseAttn
from rooflang.programs.dsv4_pro import optimization
from rooflang.programs.dsv4_pro import model
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_cluster_decode,
    optimize_model_cluster_prefill,
)
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.programs.presets.gb300 import GB300Cluster
from rooflang.programs.presets.h200 import H200Cluster
from rooflang.runtime.simulator import Simulator


def test_public_optimizers_are_the_supported_strategies():
    public_optimizers = {
        name for name, value in vars(optimization).items()
        if name.startswith("optimize_model_") and callable(value)
    }
    assert public_optimizers == {
        "optimize_model_superchip",
        "optimize_model_cluster_decode",
        "optimize_model_cluster_prefill",
    }


def test_declare_model_reuses_sequence_length_and_persists_kv():
    parameters = inspect.signature(model.declare_model).parameters
    assert "kv_prefill_len" not in parameters
    assert "persist_kv_cache" not in parameters
    assert not hasattr(model, "DecodeStepMeta")


def test_declare_model_exposes_only_single_step_decode():
    assert "n_decode_steps" not in inspect.signature(
        model.declare_model).parameters


def test_dynamic_optimizers_do_not_edit_declared_data_dependencies():
    forbidden = (".add_data_edge(", ".add_kernel(", ".remove_kernel(")
    for function in (
        optimize_model_cluster_prefill,
        optimize_model_cluster_decode,
    ):
        source = inspect.getsource(function)
        assert all(call not in source for call in forbidden)
        assert "_validate_args(" in source


def test_decode_finishes_dp_and_cp_transforms_before_placement():
    source = inspect.getsource(optimize_model_cluster_decode)
    dp_start = source.index("# DP comes first")
    cp_start = source.index("# Apply CP separately")
    placement_start = source.index("placement = Placement")

    assert dp_start < cp_start < placement_start
    assert source.rfind(".split_kernel(") < placement_start
    assert "batch_split_comm" not in source
    assert "local_placement" not in source
    assert source.rfind("optimize_comms(") < placement_start


def test_cluster_optimizers_require_ep_equal_cp_times_dp():
    with pytest.raises(ValueError, match=r"cp \* dp == ep"):
        optimization._validate_args(
            layers=[object()], batch_size=64, seq_prefill=512,
            is_prefill=False,
            cp=2, dp=2, ep=8, pp_partition=[1], n_gpus=8)


def test_cluster_optimizers_require_partition_to_cover_model():
    with pytest.raises(ValueError, match="sum to the model layer count"):
        optimization._validate_args(
            layers=[object(), object()], batch_size=64, seq_prefill=512,
            is_prefill=False,
            cp=2, dp=2, ep=4, pp_partition=[1, 2], n_gpus=8)


def test_declare_model_marks_kv_cache_for_persistence(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    graph, layers, _, _, _, output_head = model.declare_model(
        seq_prefill=512)
    assert len(output_head) == 4
    assert isinstance(output_head[0], Slice)
    assert output_head[0].inputs["x"].shape[1] == 512
    assert output_head[0].outputs["y"].shape[1] == 1
    assert layers[0].kv_persist_barrier.inputs[
        "prefill_output"].shape == (512, 1, 1)
    barriers = {layer.kv_persist_barrier for layer in layers}
    assert len(barriers) == 1
    barrier = next(iter(barriers))
    assert isinstance(barrier, Nop)
    assert set(barrier.inputs) == {"kv0", "kv1", "prefill_output"}
    assert barrier.outputs["done"].shape == (1,)


def test_declare_decode_uses_read_only_persistent_kv(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    graph, layers, _, _, kv_reads, _ = model.declare_model(
        batch_size=64,
        seq_prefill=512,
        decode=True,
    )

    barrier = layers[0].kv_persist_barrier
    assert isinstance(barrier, Nop)
    assert set(barrier.inputs) == {"kv0", "kv1", "decode_output"}
    assert barrier.outputs["done"].shape == (1,)
    for layer_id, layer in enumerate(layers):
        assert layer.kv_persist_barrier is barrier
        assert layer.kv_win_slice is None
        assert layer.sa.S_kv == kv_reads[layer_id].outputs["y"].shape[1]
        assert any(edge.src is layer.kv_cache_fan and edge.dst is barrier
                   for edge in graph._in_edges(barrier))


def test_cp4_dp2_ep8_pp2_prefill(monkeypatch):
    """Primary two-node configuration on a reduced two-layer model."""
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = B300Cluster(n_nodes=2)
    pp_partition = [1, 1]
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(
            batch_size=64,
            seq_prefill=512,
        )

    graph, placement = optimize_model_cluster_prefill(
        graph, layers, hw, emb, read_input, output_head,
        cp=4, dp=2, ep=8, pp_partition=pp_partition, n_gpus=16)

    barriers = [kernel for kernel in graph.kernels
                if isinstance(kernel, Nop)]
    assert len(barriers) == 8
    assert set(layers[0]._kv_persist_barrier_cp_dp_copies) == set(barriers)
    for barrier in barriers:
        assert barrier not in placement.placed_kernels
        assert barrier.outputs["done"].shape == (1,)
        assert set(barrier.inputs) == {"kv0", "kv1", "prefill_output"}
        assert {
            placement.get_tensor_memory(tensor).name.split("-", 1)[0]
            for tensor in barrier.inputs.values()
        } == {"n0", "n1"}
        for edge in graph._in_edges(barrier):
            for output_name, input_name in edge.mapping.items():
                assert placement.get_tensor_memory(
                    edge.src.outputs[output_name]) is \
                    placement.get_tensor_memory(barrier.inputs[input_name])

    for layer_id, layer in enumerate(layers):
        copies = layer._ffn_add_cp_dp_copies
        assert len(copies) == 8
        assert {
            placement.get_kernel_device(copy).device.name.split("-", 1)[0]
            for copy in copies
        } == {f"n{layer_id}"}
        expert_devices = {
            placement.get_kernel_device(expert).device
            for expert in layer.experts
        }
        assert len(expert_devices) == 8
        assert all(isinstance(device, Compute)
                   for device in expert_devices)


def test_h200_cluster_resources_use_component_kinds(monkeypatch):
    """H200 placement must not depend on a B300 model-name substring."""
    monkeypatch.setattr(model, "N_LAYERS", 1)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = H200Cluster(n_nodes=1)
    graph, layers, emb, read_input, _, output_head = model.declare_model(
        batch_size=16, seq_prefill=512)

    graph, placement = optimize_model_cluster_prefill(
        graph, layers, hw, emb, read_input, output_head,
        cp=2, dp=4, ep=8, pp_partition=[1], n_gpus=8)

    devices = {
        placement.get_kernel_device(kernel).device
        for kernel in placement.placed_kernels
    }
    assert devices
    assert all(device.kind == "gpu" for device in devices)
    assert all("nvidia-h200" in device.name for device in devices)
    assert Simulator(graph, placement, hw).run().total_time_us > 0


def test_h200_cluster_decode_uses_component_kinds(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 1)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = H200Cluster(n_nodes=1)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=32, seq_prefill=512, decode=True)

    graph, placement = optimize_model_cluster_decode(
        graph, layers, hw, emb, read_input, kv_reads, output_head,
        seq_prefill=512, cp=2, dp=4, ep=8,
        pp_partition=[1], n_gpus=8)

    devices = {
        placement.get_kernel_device(kernel).device
        for kernel in placement.placed_kernels
    }
    assert devices
    assert all(device.kind == "gpu" for device in devices)
    assert all("nvidia-h200" in device.name for device in devices)
    assert Simulator(
        graph, placement, hw, measurement_start=read_input,
    ).run().measured_time_us > 0


def test_decode_memory_mapping_uses_actual_node_width():
    hardware = GB300Cluster(nvl_scope=64, n_nodes=1)
    gpus, _, drams, ssds = optimization._cluster_resources(hardware)
    gpus_by_node = {"n0": gpus}

    mapped_drams = [optimization._nearby_memory(
        gpu, gpus_by_node, drams) for gpu in gpus]
    mapped_ssds = [optimization._nearby_memory(
        gpu, gpus_by_node, ssds) for gpu in gpus]

    assert mapped_drams[0] is drams["n0"][0]
    assert mapped_drams[1] is drams["n0"][0]
    assert mapped_drams[2] is drams["n0"][1]
    assert mapped_drams[-1] is drams["n0"][-1]
    assert all(mapped_drams.count(dram) == 2 for dram in drams["n0"])
    assert mapped_ssds == ssds["n0"]


def test_cp4_dp2_ep8_pp2_decode(monkeypatch):
    """Decode broadcasts Q and reads a disjoint KV shard on each CP rank."""
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = B300Cluster(n_nodes=2)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(
            batch_size=64,
            seq_prefill=512,
            decode=True,
        )

    graph, placement = optimize_model_cluster_decode(
        graph, layers, hw, emb, read_input, kv_reads, output_head,
        seq_prefill=512, cp=4, dp=2, ep=8,
        pp_partition=[1, 1], n_gpus=16)

    for layer_id, layer in enumerate(layers):
        copies = layer._decode_copies
        q_broadcasts = {
            graph._in_edges(copy)[0].src
            for copy in copies["cp_dp"]["wq_a"]
        }
        assert len(q_broadcasts) == 2
        assert all(isinstance(kernel, Broadcast)
                   for kernel in q_broadcasts)
        assert all(kernel.world == 4 for kernel in q_broadcasts)

        attention = copies["cp_dp"]["sa"]
        assert len(attention) == 8
        assert all(isinstance(kernel, SparseAttn)
                   for kernel in attention)
        assert sum(kernel.S_kv for kernel in attention[:4]) \
            == kv_reads[layer_id].outputs["y"].shape[1]
        assert {
            placement.get_kernel_device(kernel).device.name.split(
                "-", 1)[0]
            for kernel in attention
        } == {f"n{layer_id}"}
        assert sorted(
            int(placement.get_kernel_device(kernel).device.name.rsplit(
                "-", 1)[1])
            for kernel in copies["dp"]["wkv"]
        ) == [0, 4]

    barriers = [kernel for kernel in graph.kernels
                if isinstance(kernel, Nop)]
    assert len(barriers) == 8
    assert all(isinstance(barrier, Nop) for barrier in barriers)
    assert all(barrier not in placement.placed_kernels
               for barrier in barriers)
    assert all(set(barrier.inputs) == {"kv0", "kv1", "decode_output"}
               for barrier in barriers)
    assert all(barrier.outputs["done"].shape == (1,)
               for barrier in barriers)
    assert {
        placement.get_tensor_memory(barrier.inputs["kv0"]).name.split(
            "-", 1)[0]
        for barrier in barriers
    } == {"n0"}
    assert {
        placement.get_tensor_memory(barrier.inputs["kv1"]).name.split(
            "-", 1)[0]
        for barrier in barriers
    } == {"n1"}

    reduce_scatters = [
        kernel for kernel in graph.kernels
        if isinstance(kernel, ReduceScatter)
    ]
    assert len(reduce_scatters) == 2 * 2
    assert all(kernel.world == 4 for kernel in reduce_scatters)
    kv_preloads = {
        kernel for kernel in graph.kernels
        if isinstance(kernel, ReadInput) and "kv" in kernel.inputs
    }
    assert all(
        placement.get_tensor_memory(kernel.inputs["kv"]).kind == "ssd"
        for kernel in kv_preloads
    )
    assert placement.get_tensor_memory(
        read_input.inputs["tokens"]).kind == "dram"

    pending = list(kv_preloads)
    visited = set()
    kv_roots = set()
    while pending:
        kernel = pending.pop()
        if kernel in visited:
            continue
        visited.add(kernel)
        predecessors = [edge.src for edge in graph._in_edges(kernel)]
        if predecessors:
            pending.extend(predecessors)
        else:
            kv_roots.add(kernel)
    assert kv_roots
    assert all(
        placement.get_tensor_memory(tensor).kind == "ssd"
        for kernel in kv_roots
        for tensor in kernel.inputs.values()
    )
    control_predecessors = {
        kernel for kernel in graph._dag.predecessors(read_input)
        if not graph._dag.edges[kernel, read_input]["mapping"]
    }
    assert control_predecessors == kv_preloads

    result = Simulator(
        graph, placement, hw, measurement_start=read_input).run()
    assert result.total_time_us > 0
    assert max(
        entry.end_us for entry in result.trace
        if entry.kernel in kv_preloads
    ) <= result.measurement_start_us
    assert result.measured_time_us < result.total_time_us


def test_cp_decode_broadcasts_q_between_same_stage_layers(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = B300Cluster(n_nodes=1)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(
            batch_size=64,
            seq_prefill=512,
            decode=True,
        )
    graph = optimize_model_cluster_decode(
        graph, layers, hw, emb, read_input, kv_reads, output_head,
        seq_prefill=512, cp=4, dp=2, ep=8,
        pp_partition=[2], n_gpus=8,
    )[0]

    q_broadcasts = {
        graph._in_edges(copy)[0].src
        for layer in layers
        for copy in layer._decode_copies["cp_dp"]["wq_a"]
    }
    assert len(q_broadcasts) == 2 * 2
    assert all(isinstance(kernel, Broadcast) and kernel.world == 4
               for kernel in q_broadcasts)
