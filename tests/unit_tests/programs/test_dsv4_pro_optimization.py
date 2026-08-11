"""Tests for DeepSeek V4 Pro placement strategies."""

import inspect

import pytest

from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.comm import Broadcast, ReduceScatter
from rooflang.language.kernels.forward import Nop, Slice, SparseAttn
from rooflang.language.kernels.identity import Move
from rooflang.programs.dsv4_pro import optimization
from rooflang.programs.dsv4_pro import model
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_b300_cluster_a_cp_dp_ep_pp_decode,
    optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill,
)
from rooflang.programs.presets.b300 import B300ClusterA
from rooflang.runtime.simulator import Simulator


def test_public_optimizers_are_the_supported_strategies():
    public_optimizers = {
        name for name, value in vars(optimization).items()
        if name.startswith("optimize_model_") and callable(value)
    }
    assert public_optimizers == {
        "optimize_model_b300_superchip_a",
        "optimize_model_b300_cluster_a_cp_dp_ep_pp_decode",
        "optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill",
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
        optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill,
        optimize_model_b300_cluster_a_cp_dp_ep_pp_decode,
    ):
        source = inspect.getsource(function)
        assert all(call not in source for call in forbidden)


def test_decode_finishes_cp_and_dp_transforms_before_placement():
    source = inspect.getsource(
        optimize_model_b300_cluster_a_cp_dp_ep_pp_decode)
    cp_start = source.index("# CP comes first")
    dp_start = source.index("# DP is the second graph transform")
    placement_start = source.index("placement = Placement")

    assert cp_start < dp_start < placement_start
    assert source.rfind(".split_kernel(") < placement_start
    assert "local_placement" not in source
    assert source.index("optimize_comms(") > placement_start


def test_cp_dp_ep_pp_requires_ep_equal_cp_times_dp():
    with pytest.raises(ValueError, match=r"cp \* dp == ep"):
        optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
            g=None, layers=[object()], hw=None, emb=None,
            cp=2, dp=2, ep=8, pp_partition=[1], n_gpus=8)


def test_cp_dp_ep_pp_requires_layer_counts_to_cover_model():
    with pytest.raises(ValueError, match="sum to the model layer count"):
        optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
            g=None, layers=[object(), object()], hw=None, emb=None,
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

    hw = B300ClusterA(n_nodes=2)
    pp_partition = [1, 1]
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(
            batch_size=64,
            seq_prefill=512,
        )

    graph, placement = optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
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

    moves = [kernel for kernel in graph.kernels
             if isinstance(kernel, Move)]
    assert all(
        "nvidia-b300" in placement.get_kernel_device(move).device.name
        for move in moves
    )

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


def test_cp4_dp2_ep8_pp2_decode(monkeypatch):
    """Decode broadcasts Q and reads a disjoint KV shard on each CP rank."""
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = B300ClusterA(n_nodes=2)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(
            batch_size=64,
            seq_prefill=512,
            decode=True,
        )

    graph, placement = optimize_model_b300_cluster_a_cp_dp_ep_pp_decode(
        graph, layers, hw, emb, read_input, kv_reads, output_head,
        cp=4, dp=2, ep=8, pp_partition=[1, 1], n_gpus=16)

    for layer_id, layer in enumerate(layers):
        q_broadcasts = {
            graph._in_edges(copy)[0].src
            for copy in layer._wq_a_cp_dp_copies
        }
        assert len(q_broadcasts) == 2
        assert all(isinstance(kernel, Broadcast)
                   for kernel in q_broadcasts)
        assert all(kernel.world == 4 for kernel in q_broadcasts)

        attention = layer._sa_cp_dp_copies
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
            for kernel in layer._wkv_dp_copies
        ) == [0, 4]

    barriers = layers[0]._kv_persist_barrier_cp_dp_copies
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

    result = Simulator(graph, placement, hw).run()
    assert result.total_time_us > 0


def test_cp_decode_broadcasts_q_between_same_stage_layers(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = B300ClusterA(n_nodes=1)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(
            batch_size=64,
            seq_prefill=512,
            decode=True,
        )
    graph = optimize_model_b300_cluster_a_cp_dp_ep_pp_decode(
        graph, layers, hw, emb, read_input, kv_reads, output_head,
        cp=4, dp=2, ep=8, pp_partition=[2], n_gpus=8,
    )[0]

    q_broadcasts = {
        graph._in_edges(copy)[0].src
        for layer in layers
        for copy in layer._wq_a_cp_dp_copies
    }
    assert len(q_broadcasts) == 2 * 2
    assert all(isinstance(kernel, Broadcast) and kernel.world == 4
               for kernel in q_broadcasts)
