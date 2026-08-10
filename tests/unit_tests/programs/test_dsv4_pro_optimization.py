"""Tests for DeepSeek V4 Pro placement strategies."""

import pytest

from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.forward import Nop
from rooflang.language.kernels.identity import Move
from rooflang.programs.dsv4_pro import optimization
from rooflang.programs.dsv4_pro import model
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill,
    optimize_model_b300_cluster_a_cp8_ep8_1node,
)
from rooflang.programs.presets.b300 import B300ClusterA


def test_cp8_ep8_1node_rejects_decode():
    with pytest.raises(ValueError, match="supports prefill only"):
        optimize_model_b300_cluster_a_cp8_ep8_1node(
            g=None, layers=[], hw=None, decode_steps=[object()])


def test_dp8_ep8_requires_equal_parallel_sizes(monkeypatch):
    monkeypatch.setattr(optimization, "DP", 4)
    monkeypatch.setattr(optimization, "EP", 8)
    with pytest.raises(ValueError, match="requires DP == EP"):
        optimization.optimize_model_b300_cluster_a_dp8_ep8_1node(
            g=None, layers=[], hw=None)


def test_cp8_ep8_requires_equal_parallel_sizes(monkeypatch):
    monkeypatch.setattr(optimization, "CP", 4)
    monkeypatch.setattr(optimization, "EP", 8)
    with pytest.raises(ValueError, match="requires CP == EP"):
        optimization.optimize_model_b300_cluster_a_cp8_ep8_1node(
            g=None, layers=[], hw=None)


def test_cp_dp_ep_pp_prefill_rejects_decode():
    with pytest.raises(ValueError, match="supports prefill only"):
        optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
            g=None, layers=[], hw=None, decode_steps=[object()],
            cp=4, dp=2, ep=8, pp=[], n_gpus=16)


def test_cp_dp_ep_pp_requires_ep_equal_cp_times_dp():
    with pytest.raises(ValueError, match=r"cp \* dp == ep"):
        optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
            g=None, layers=[object()], hw=None, emb=None,
            cp=2, dp=2, ep=8, pp=[1], n_gpus=8)


def test_cp_dp_ep_pp_requires_layer_counts_to_cover_model():
    with pytest.raises(ValueError, match="sum to the model layer count"):
        optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
            g=None, layers=[object(), object()], hw=None, emb=None,
            cp=2, dp=2, ep=4, pp=[1, 2], n_gpus=8)


def test_declare_model_marks_kv_cache_for_persistence(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    graph, layers, *_ = model.declare_model(
        seq_prefill=512, persist_kv_cache=True)
    assert all(layer.persist_kv_cache for layer in layers)
    barriers = {layer.kv_persist_barrier for layer in layers}
    assert len(barriers) == 1
    barrier = next(iter(barriers))
    assert isinstance(barrier, Nop)
    assert set(barrier.inputs) == {"kv0", "kv1", "prefill_output"}
    assert barrier.outputs["done"].shape == (1,)


def test_cp4_dp2_ep8_pp2_prefill(monkeypatch):
    """Primary two-node configuration on a reduced two-layer model."""
    monkeypatch.setattr(model, "N_LAYERS", 2)
    monkeypatch.setattr(model, "N_EXPERTS", 8)
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)

    hw = B300ClusterA(n_nodes=2)
    pp = [1, 1]
    graph, layers, decode_steps, emb, read_input, kv_reads, head = \
        model.declare_model(
            batch_size=64,
            seq_prefill=512,
            persist_kv_cache=True,
        )

    graph, placement = optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill(
        graph, layers, hw, emb, read_input, decode_steps, kv_reads, head,
        cp=4, dp=2, ep=8, pp=pp, n_gpus=16)

    barriers = [kernel for kernel in graph.kernels
                if isinstance(kernel, Nop)]
    assert len(barriers) == 1
    barrier = barriers[0]
    assert barrier not in placement.placed_kernels
    assert barrier.outputs["done"].shape == (16,)
    assert len(barrier.inputs) == 2 * 8 + 1
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
