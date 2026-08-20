"""Tests for DeepSeek V4 Flash model-specific behavior."""

import pytest

from rooflang.programs.experiments import find_pareto_frontier as finder
from rooflang.programs.models import dsv4_flash
from rooflang.programs.models.dsv4_flash import model, optimization
from rooflang.programs.presets.b300 import B300Cluster


def test_config_matches_flash_main_model():
    assert dsv4_flash.D == 4096
    assert dsv4_flash.N_LAYERS == 43
    assert dsv4_flash.H == 64
    assert dsv4_flash.Q_LORA == 1024
    assert dsv4_flash.O_GROUPS == 8
    assert dsv4_flash.N_EXPERTS == 256
    assert dsv4_flash.MOE_INTER == 2048
    assert dsv4_flash.INDEX_TOPK == 512
    assert len(dsv4_flash.COMPRESS_RATIOS) == dsv4_flash.N_LAYERS
    assert dsv4_flash.COMPRESS_RATIOS.count(0) == 2
    assert dsv4_flash.COMPRESS_RATIOS.count(4) == 21
    assert dsv4_flash.COMPRESS_RATIOS.count(128) == 20
    assert dsv4_flash.COMPRESS_RATIOS[:4] == [0, 0, 4, 128]
    assert dsv4_flash.COMPRESS_RATIOS[-3:] == [4, 128, 4]


@pytest.mark.parametrize("decode", [False, True])
def test_zero_ratio_layers_use_window_only_kv(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 4)

    graph, layers, _, _, kv_reads, _ = model.declare_model(
        batch_size=8, seq_prefill=128, decode=decode)

    assert len(layers) == 4
    assert [layer.sa.S_kv for layer in layers] == [128, 128, 160, 129]
    assert [layer.sa.k_sel for layer in layers] == [128, 128, 640, 129]

    for layer in layers[:2]:
        assert layer.bridge.world == 2
        assert layer.comp is None
        assert layer.comp_norm is None
        assert layer.kv_concat is None
        assert layer.index_cache_fan is None
        assert layer.sa.indexer_s_kv == 0
        assert layer.kv_persist_fan is not None

    if decode:
        assert [read.outputs["y"].shape for read in kv_reads] == [
            (8, 128, 512),
            (8, 128, 512),
            (8, 160, 512),
            (8, 129, 512),
        ]
        assert layers[2].index_cache_read is not None
    else:
        assert not kv_reads
        assert [layer.bridge.world for layer in layers] == [2, 2, 3, 3]
        assert layers[2].comp is not None
        assert layers[2].kv_concat is not None
        assert layers[2].index_cache_fan is not None

    graph.validate()


def test_optimizer_validation_ignores_zero_ratio(monkeypatch):
    monkeypatch.setattr(optimization, "COMPRESS_RATIOS", (0, 4, 128))

    optimization._validate_args(
        [object()], 8, 256, True,
        cp=2, dp=4, ep=8, pp_partition=[1], n_gpus=8,
    )


@pytest.mark.parametrize("decode", [False, True])
def test_zero_ratio_layer_cluster_optimization(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 1)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=8, seq_prefill=128, decode=decode)
    hardware = B300Cluster(n_nodes=1)
    kwargs = dict(cp=1, dp=8, ep=8, pp_partition=[1], n_gpus=8)

    if decode:
        graph, placement = optimization.optimize_model_cluster_decode(
            graph, layers, hardware, emb, read_input, kv_reads, output_head,
            seq_prefill=128, **kwargs)
    else:
        graph, placement = optimization.optimize_model_cluster_prefill(
            graph, layers, hardware, emb, read_input, output_head, **kwargs)

    graph.validate()
    placement.validate(graph)


def test_pareto_enumeration_ignores_zero_ratio(monkeypatch):
    monkeypatch.setattr(finder, "N_LAYERS", 43, raising=False)
    monkeypatch.setattr(finder, "N_EXPERTS", 256, raising=False)
    monkeypatch.setattr(finder, "WINDOW", 128, raising=False)
    monkeypatch.setattr(
        finder, "COMPRESS_RATIOS", (0, 4, 128), raising=False)

    configs = finder.enumerate_parallel_configs(8, 8192)

    assert configs
