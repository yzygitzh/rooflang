# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Tests for GLM-5.2-FP8 model-specific behavior."""

import pytest

from rooflang.language.kernels.forward import Glm52SparseAttn
from rooflang.programs.experiments import find_pareto_frontier as finder
from rooflang.programs.models import glm_5_2_fp8
from rooflang.programs.models.glm_5_2_fp8 import model, optimization
from rooflang.programs.presets.b300 import B300Cluster


def test_config_matches_glm_5_2_main_model():
    assert glm_5_2_fp8.D == 6144
    assert glm_5_2_fp8.N_LAYERS == 78
    assert glm_5_2_fp8.H == 64
    assert glm_5_2_fp8.Q_LORA == 2048
    assert glm_5_2_fp8.KV_LORA == 512
    assert glm_5_2_fp8.QK_NOPE_HD == 192
    assert glm_5_2_fp8.QK_ROPE_HD == 64
    assert glm_5_2_fp8.QK_HD == 256
    assert glm_5_2_fp8.V_HD == 256
    assert glm_5_2_fp8.KV_CACHE_DIM == 576
    assert glm_5_2_fp8.INDEX_H == 32
    assert glm_5_2_fp8.INDEX_HD == 128
    assert glm_5_2_fp8.INDEX_TOPK == 2048
    assert glm_5_2_fp8.DENSE_LAYERS == 3
    assert glm_5_2_fp8.DENSE_INTER == 12288
    assert glm_5_2_fp8.N_EXPERTS == 256
    assert glm_5_2_fp8.TOPK == 8
    assert glm_5_2_fp8.MOE_INTER == 2048
    assert glm_5_2_fp8.ROUTER_SCORING_FUNC == "sigmoid"
    assert len(glm_5_2_fp8.FULL_INDEXER_LAYERS) == 21
    assert glm_5_2_fp8.N_LAYERS - len(
        glm_5_2_fp8.FULL_INDEXER_LAYERS) == 57
    assert glm_5_2_fp8.FULL_INDEXER_LAYERS[:4] == (0, 1, 2, 6)
    assert glm_5_2_fp8.FULL_INDEXER_LAYERS[-1] == 74
    assert not hasattr(glm_5_2_fp8, "WINDOW")
    assert not hasattr(glm_5_2_fp8, "COMPRESS_RATIOS")
    assert not hasattr(glm_5_2_fp8, "ROUTED_SCALING_FACTOR")
    assert not hasattr(glm_5_2_fp8, "MAX_CONTEXT")


def test_one_1m_sequence_fp8_cache_size_is_46_5_gib():
    tokens = 1_048_576
    main_bytes = 78 * tokens * 576
    index_bytes = 21 * tokens * 128

    assert main_bytes / 2**30 == pytest.approx(43.875)
    assert index_bytes / 2**30 == pytest.approx(2.625)
    assert (main_bytes + index_bytes) / 2**30 == pytest.approx(46.5)


@pytest.mark.parametrize("decode", [False, True])
def test_layer_types_indexer_ownership_and_fp8_caches(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 4)
    graph, layers, emb, _, kv_reads, output_head = model.declare_model(
        batch_size=8, seq_prefill=128, decode=decode)

    assert len(layers) == 4
    assert [layer.is_dense for layer in layers] == [True, True, True, False]
    assert [layer.has_full_indexer for layer in layers] == [
        True, True, True, False]
    assert all(isinstance(layer.sa, Glm52SparseAttn) for layer in layers)
    assert all(layer.sa.S_kv == 128 for layer in layers)
    assert all(layer.sa.k_sel == 128 for layer in layers)
    assert all(layer.sa.inputs["kv"].dtype == "fp8" for layer in layers)
    assert all(layer.sa.weights["kv_b"].dtype == "fp8" for layer in layers)
    assert all(layer.sa.indexer_compute_dtype == "fp8" for layer in layers)
    assert all(layer.sa.indexer_reduce_dtype == "fp32" for layer in layers)
    assert all(
        layer.sa.attention_flops
        == 2 * layer.sa.B * layer.sa.H * layer.sa.selected_pairs * (576 + 512)
        for layer in layers)
    assert all(type(kernel).__name__ != "RoPE" for kernel in graph.kernels)
    stage_s = 1 if decode else 128
    assert layers[0].wq_b.outputs["y"].shape == (8, stage_s, 64 * 256)
    assert layers[0].wkv.outputs["y"].shape == (8, stage_s, 576)
    assert layers[0].kv_norm.D == 576
    assert layers[0].sa.weights["kv_b"].shape == (
        512, 64 * (192 + 256))
    assert layers[0].sa.outputs["y"].shape == (8, stage_s, 64 * 256)
    assert (layers[0].wo.K, layers[0].wo.N) == (64 * 256, 6144)
    assert layers[0].dense_up.N == 2 * 12288
    assert layers[3].dispatch.scoring_func == "sigmoid"
    assert len(layers[3].experts) == 2 * glm_5_2_fp8.N_EXPERTS
    assert layers[3].experts[0].N == 2 * 2048
    assert layers[3].sw_up.N == 2 * 2048
    assert not layers[0].experts
    assert (emb.weights["emb"].weight_id
            != output_head[-2].weights["w"].weight_id)

    for layer in layers[:3]:
        assert layer.sa.inputs["index_kv"].dtype == "fp8"
        assert layer.index_wq is not None
        assert layer.index_wk is not None
        assert layer.index_weights.weights["w"].dtype == "bf16"
        assert layer.index_weights.outputs["y"].dtype == "fp32"
    assert layers[3].index_wq is None
    assert "index_kv" not in layers[3].sa.inputs
    assert layers[3].gate.weights["w"].dtype == "fp32"

    if decode:
        assert len(kv_reads) == 4
        assert all(read.outputs["y"].shape == (8, 128, 576)
                   for read in kv_reads)
        assert all(layer.kv_sink is not None for layer in layers)
        assert all(layer.index_cache_read is not None
                   for layer in layers[:3])
    else:
        assert not kv_reads
        assert all(layer.kv_cache_quant.outputs["y"].dtype == "fp8"
                   for layer in layers)
        assert all(layer.index_cache_quant.outputs["y"].dtype == "fp8"
                   for layer in layers[:3])

    graph.validate()


@pytest.mark.parametrize("decode", [False, True])
def test_cluster_optimization_with_dense_and_moe_layers(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 4)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=8, seq_prefill=128, decode=decode)
    hardware = B300Cluster(n_nodes=1)
    kwargs = dict(cp=2, dp=4, ep=8, pp_partition=[4], n_gpus=8)

    if decode:
        graph, placement = optimization.optimize_model_cluster_decode(
            graph, layers, hardware, emb, read_input, kv_reads, output_head,
            seq_prefill=128, **kwargs)
    else:
        graph, placement = optimization.optimize_model_cluster_prefill(
            graph, layers, hardware, emb, read_input, output_head, **kwargs)

    graph.validate()
    placement.validate(graph)
    assert layers[0]._expert_weight_read_fraction == 0
    assert layers[3]._expert_weight_read_fraction > 0


@pytest.mark.parametrize("decode", [False, True])
def test_two_stage_cluster_placement_materializes_pipeline_boundary(
        monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=8, seq_prefill=128, decode=decode)
    hardware = B300Cluster(n_nodes=2)
    kwargs = dict(cp=1, dp=8, ep=8, pp_partition=[1, 1], n_gpus=16)

    if decode:
        graph, placement = optimization.optimize_model_cluster_decode(
            graph, layers, hardware, emb, read_input, kv_reads, output_head,
            seq_prefill=128, **kwargs)
    else:
        graph, placement = optimization.optimize_model_cluster_prefill(
            graph, layers, hardware, emb, read_input, output_head, **kwargs)

    graph.validate()
    placement.validate(graph)


def test_pareto_model_selection_does_not_require_compression_constants():
    finder._select_model("glm_5_2_fp8")

    assert finder.N_LAYERS == 78
    assert finder.N_EXPERTS == 256
    assert finder.WINDOW is None
    assert finder.COMPRESS_RATIOS == ()
