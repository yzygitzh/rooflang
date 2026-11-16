# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Tests for the DeepSeek V2.5 model implementation."""

import pytest

from rooflang.language.kernels.forward import (
    Gemm, MlaAttn, PartialRMSNorm, StridedGemm, TokenCombine,
    TokenDispatch,
)
from rooflang.programs.experiments import find_pareto_frontier as finder
from rooflang.programs.models import dsv2_5
from rooflang.programs.models.dsv2_5 import model, optimization
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.runtime.simulator import Simulator


def test_config_matches_official_v2_5():
    assert dsv2_5.D == 5120
    assert dsv2_5.N_LAYERS == 60
    assert dsv2_5.H == 128
    assert dsv2_5.Q_LORA == 1536
    assert dsv2_5.KV_LORA == 512
    assert dsv2_5.QK_NOPE_HD == 128
    assert dsv2_5.QK_ROPE_HD == 64
    assert dsv2_5.QK_HD == 192
    assert dsv2_5.V_HD == 128
    assert dsv2_5.KV_CACHE_DIM == 576
    assert dsv2_5.DENSE_LAYERS == 1
    assert dsv2_5.DENSE_INTER == 12288
    assert dsv2_5.N_EXPERTS == 160
    assert dsv2_5.TOPK == 6
    assert dsv2_5.MOE_INTER == 1536
    assert dsv2_5.N_SHARED_EXPERTS == 2
    assert dsv2_5.SHARED_INTER == 3072
    assert dsv2_5.ROUTER_SCORING_FUNC == "softmax"
    assert dsv2_5.V == 102400
    assert not hasattr(dsv2_5, "WINDOW")
    assert not hasattr(dsv2_5, "COMPRESS_RATIOS")
    assert not hasattr(dsv2_5, "INDEX_TOPK")


def test_parameter_count_matches_official_checkpoint():
    """The graph must reproduce the checkpoint's 235,741,434,880 BF16 params."""
    graph, *_ = model.declare_model(batch_size=8, seq_prefill=128)

    weights = {}
    for kernel in graph.kernels:
        for port, tensor in kernel.weights.items():
            weights[(tensor.weight_id, port)] = tensor

    total_params = sum(t.size_bytes for t in weights.values()) / 2.0
    assert total_params == 235_741_434_880


@pytest.mark.parametrize("decode", [False, True])
def test_graph_layer_types_mla_and_bf16_caches(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 4)
    graph, layers, emb, _, kv_reads, output_head = model.declare_model(
        batch_size=8, seq_prefill=128, decode=decode)

    assert [layer.is_dense for layer in layers] == [True, False, False, False]
    assert layers[0].experts == []
    assert layers[0].gate is None
    assert layers[0].dense_up.weights["w"].shape == (5120, 2 * 12288)
    assert layers[0].dense_down.weights["w"].shape == (12288, 5120)

    for layer in layers:
        assert isinstance(layer.sa, MlaAttn)
        assert layer.sa.dtype_ == "bf16"
        assert layer.sa.kv_transform_dtype == "bf16"
        assert layer.sa.q_dtype == "bf16"
        assert layer.sa.kv_dtype == "bf16"
        assert layer.sa.out_dtype == "bf16"
        assert layer.sa.kv_cache_dim == 576
        assert layer.sa.kv_lora_rank == 512
        width = (
            576 + 512 if decode
            else layer.sa.qk_head_dim + layer.sa.v_head_dim)
        assert layer.sa.attention_flops == pytest.approx(
            2 * 8 * 128 * layer.sa.selected_pairs * width)
        assert layer.sa.weights["kv_b"].shape == (512, 128 * 256)
        assert layer.sa.weights["kv_b"].dtype == "bf16"
        assert isinstance(layer.kv_norm, PartialRMSNorm)
        assert layer.kv_norm.input_dim == 576
        assert layer.kv_norm.norm_dim == 512
        assert layer.wq_a.weights["w"].shape == (5120, 1536)
        assert layer.wq_b.weights["w"].shape == (1536, 128 * 192)
        assert layer.wkv.weights["w"].shape == (5120, 576)
        assert layer.wo.weights["w"].shape == (16384, 5120)
        assert all(
            kernel.weights["w"].dtype == "bf16"
            for kernel in (layer.wq_a, layer.wq_b, layer.wkv, layer.wo))

    for layer in layers[1:]:
        assert layer.ffn_fan.world == 3
        assert isinstance(layer.gate, Gemm)
        assert layer.gate.weights["w"].dtype == "bf16"
        assert layer.gate.outputs["y"].dtype == "fp32"
        assert isinstance(layer.dispatch, TokenDispatch)
        assert layer.dispatch.scoring_func == "softmax"
        assert layer.dispatch.topk == 6
        assert isinstance(layer.combine, TokenCombine)
        assert len(layer.experts) == 2 * dsv2_5.N_EXPERTS
        assert isinstance(layer.experts[0], StridedGemm)
        assert layer.experts[0].weights["w"].shape == (5120, 2 * 1536)
        assert layer.experts[1].weights["w"].shape == (1536, 5120)
        assert layer.sw_up.weights["w"].shape == (5120, 2 * 3072)
        assert layer.sw_down.weights["w"].shape == (3072, 5120)

    # The released checkpoint is BF16: no quantized weights or scales.
    assert all(
        tensor.dtype != "ue8m0"
        for kernel in graph.kernels
        for tensor in (*kernel.inputs.values(), *kernel.outputs.values(),
                       *kernel.weights.values()))

    assert emb.weights["emb"].weight_id != output_head[-2].weights["w"].weight_id

    if decode:
        assert len(kv_reads) == 4
        assert all(read.outputs["y"].dtype == "bf16" for read in kv_reads)
        assert all(read.outputs["y"].shape == (8, 128, 576)
                   for read in kv_reads)
        assert all(layer.kv_sink is not None for layer in layers)
    else:
        assert not kv_reads
        assert all(layer.kv_cache_fan is not None for layer in layers)
        assert all(layer.kv_sink is None for layer in layers)

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


@pytest.mark.parametrize("decode", [False, True])
def test_two_layer_cluster_graph_simulates(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=8, seq_prefill=128, decode=decode)
    hardware = B300Cluster(n_nodes=1)
    kwargs = dict(cp=1, dp=8, ep=8, pp_partition=[2], n_gpus=8)

    if decode:
        graph, placement = optimization.optimize_model_cluster_decode(
            graph, layers, hardware, emb, read_input, kv_reads, output_head,
            seq_prefill=128, **kwargs)
    else:
        graph, placement = optimization.optimize_model_cluster_prefill(
            graph, layers, hardware, emb, read_input, output_head, **kwargs)

    assert Simulator(graph, placement, hardware).run().total_time_us > 0


def test_pareto_model_selection_supports_dsv2_5():
    finder._select_model("dsv2_5")

    assert finder.N_LAYERS == 60
    assert finder.N_EXPERTS == 160
    assert finder.WINDOW is None
    assert finder.COMPRESS_RATIOS == ()


@pytest.mark.parametrize(
    "case", ["partition", "cp_ep", "pp_gpus", "experts", "batch",
             "sequence"],
)
def test_validation_rejects_each_constraint(case, monkeypatch):
    monkeypatch.setattr(optimization, "N_EXPERTS", 8)
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

    with pytest.raises(ValueError):
        optimization._validate_args(**kwargs)
