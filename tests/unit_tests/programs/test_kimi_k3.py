"""Tests for the Kimi-K3 text-model implementation."""

import pytest

from rooflang.language.kernels.comm import AllGather, Scatter
from rooflang.language.kernels.forward import (
    AttnRes, KimiK3DeltaAttn, KimiK3DeltaAttnCpMerge,
    KimiK3DeltaAttnCpSummary, KimiK3MlaAttn, PartialRMSNorm,
)
from rooflang.programs.experiments import find_pareto_frontier as finder
from rooflang.programs.models import kimi_k3
from rooflang.programs.models.kimi_k3 import model, optimization
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.runtime.simulator import Simulator


def test_config_matches_kimi_k3_text_model():
    assert kimi_k3.D == 7168
    assert kimi_k3.N_LAYERS == 93
    assert kimi_k3.H == 96
    assert kimi_k3.Q_LORA == 1536
    assert kimi_k3.KV_LORA == 512
    assert kimi_k3.QK_NOPE_HD == 128
    assert kimi_k3.QK_ROPE_HD == 64
    assert kimi_k3.QK_HD == 192
    assert kimi_k3.V_HD == 128
    assert kimi_k3.KV_CACHE_DIM == 576
    assert kimi_k3.KDA_HD == 128
    assert kimi_k3.KDA_CHUNK == 64
    assert kimi_k3.KDA_CONV == 4
    assert kimi_k3.ATTN_RES_BLOCK == 12
    assert kimi_k3.DENSE_LAYERS == 1
    assert kimi_k3.DENSE_INTER == 33792
    assert kimi_k3.N_EXPERTS == 896
    assert kimi_k3.TOPK == 16
    assert kimi_k3.ROUTED_D == 3584
    assert kimi_k3.MOE_INTER == 3072
    assert kimi_k3.N_SHARED_EXPERTS == 2
    assert kimi_k3.SHARED_INTER == 6144
    assert not hasattr(kimi_k3, "MLA_CACHE_DTYPE")
    assert not hasattr(kimi_k3, "KDA_STATE_DTYPE")
    assert len(kimi_k3.FULL_ATTN_LAYERS) == 24
    assert len(kimi_k3.KDA_LAYERS) == 69
    assert kimi_k3.FULL_ATTN_LAYERS[:3] == (3, 7, 11)
    assert kimi_k3.FULL_ATTN_LAYERS[-2:] == (91, 92)
    assert not hasattr(kimi_k3, "MAX_CONTEXT")


@pytest.mark.parametrize("decode", [False, True])
def test_text_graph_layer_types_caches_and_precisions(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 4)
    graph, layers, emb, _, kv_reads, output_head = model.declare_model(
        batch_size=8, seq_prefill=128, decode=decode)

    assert [layer.is_kda for layer in layers] == [True, True, True, False]
    assert [layer.is_dense for layer in layers] == [True, False, False, False]
    assert all(isinstance(layer.sa, KimiK3DeltaAttn)
               for layer in layers[:3])
    assert isinstance(layers[3].sa, KimiK3MlaAttn)
    assert type(layers[3].sa).__name__ == "KimiK3MlaAttn"
    assert isinstance(layers[3].kv_norm, PartialRMSNorm)
    assert layers[3].kv_norm.input_dim == 576
    assert layers[3].kv_norm.norm_dim == 512
    assert layers[3].sa.inputs["kv"].dtype == "fp8"
    assert layers[3].sa.inputs["q"].dtype == "bf16"
    assert layers[3].sa.dtype_ == "fp8"
    assert layers[3].sa.kv_transform_dtype == "fp8"
    assert layers[3].sa.weights["kv_b"].dtype == "fp8"
    assert layers[3].sa.weights["kv_b"].shape == (
        512, 96 * (128 + 128))
    assert layers[3].sa.weights["kv_b_scale"].dtype == "ue8m0"
    assert layers[0].sa.mode == ("recurrent" if decode else "chunk")
    assert layers[0].sa.dtype_ == "bf16"
    assert layers[0].state_store.outputs["state"].dtype == "bf16"
    assert layers[0].state_store.outputs["conv_state"].dtype == "bf16"
    assert isinstance(layers[0].mlp_res, AttnRes)
    assert layers[0].mlp_res.R == 1
    assert isinstance(output_head[0], AttnRes)
    assert all(getattr(layers[0], field).weights["w"].dtype == "fp8"
               for field in ("kda_wq", "kda_wk", "kda_wv",
                             "kda_f_a", "kda_f_b", "output_gate", "wo"))
    assert layers[0].kda_beta.weights["w"].dtype == "bf16"
    assert layers[0].dense_up.weights["w"].dtype == "fp8"
    assert layers[0].dense_down.weights["w"].dtype == "fp8"
    assert not layers[0].experts
    assert layers[1].routed_down.weights["w"].dtype == "bf16"
    assert layers[1].experts[0].weights["w"].dtype == "fp4"
    assert layers[1].sw_up.weights["w"].dtype == "fp8"
    assert layers[1].sw_down.weights["w"].dtype == "fp8"
    assert layers[1].gate.weights["w"].dtype == "bf16"
    assert layers[1].gate.outputs["y"].dtype == "fp32"
    assert all(getattr(layers[3], field).weights["w"].dtype == "fp8"
               for field in ("wq_a", "wq_b", "wkv",
                             "output_gate", "wo"))
    assert len(layers[1].experts) == 2 * kimi_k3.N_EXPERTS
    assert emb.weights["emb"].weight_id != output_head[-2].weights["w"].weight_id

    local_kda = layers[0].sa
    expected = 8 * 96 * (
        6 * (1 if decode else 128) * 128 * 128
        + (0 if decode else 3 * 128 * 64 * 128
           + 128 * 64**2)
    )
    if decode:
        expected = 8 * 96 * (7 * 128 * 128 + 2 * 128)
    assert local_kda.attention_flops == expected

    if decode:
        assert len(kv_reads) == 4
        assert layers[0].sa.inputs["state"].dtype == "bf16"
        assert layers[0].sa.inputs["conv_state"].dtype == "bf16"
        assert layers[0].conv_state_read is not None
        assert layers[3].sa.S_kv == 128
    else:
        assert not kv_reads
        assert layers[3].kv_cache_quant.outputs["y"].dtype == "fp8"

    graph.validate()


def test_prefill_kda_cp_summary_allgather_and_merge(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 4)
    graph, layers, emb, read_input, _, output_head = model.declare_model(
        batch_size=8, seq_prefill=128, decode=False)
    graph, placement = optimization.optimize_model_cluster_prefill(
        graph, layers, B300Cluster(n_nodes=1), emb, read_input, output_head,
        cp=2, dp=4, ep=8, pp_partition=[4], n_gpus=8)

    for layer in layers[:3]:
        summaries = layer._kda_cp_summary_cp_dp_copies
        merges = layer._kda_cp_merge_cp_dp_copies
        allgathers = layer._kda_cp_allgathers
        assert len(summaries) == 8
        assert len(merges) == 8
        assert len(allgathers) == 4
        assert all(isinstance(kernel, KimiK3DeltaAttnCpSummary)
                   for kernel in summaries)
        assert all(isinstance(kernel, KimiK3DeltaAttnCpMerge)
                   for kernel in merges)
        assert all(isinstance(kernel, AllGather) for kernel in allgathers)
        assert [kernel.rank for kernel in merges] == [0, 1] * 4
        assert all("initial_state" in kernel.inputs
                   for kernel in layer._sa_cp_dp_copies)
        assert summaries[0].flops_by_dtype["fp32"] > 0
        assert summaries[0].outputs["summary"].dtype == "fp32"
        assert summaries[1].flops == 0
        for dp_rank in range(4):
            source = summaries[dp_rank * 2]
            sink = layer._sa_cp_dp_copies[dp_rank * 2 + 1]
            assert "conv_halo" not in \
                layer._sa_cp_dp_copies[dp_rank * 2].inputs
            assert sink.inputs["conv_halo"].dtype == "bf16"
            source_memory = placement.get_tensor_memory(
                source.outputs["conv_halo"])
            assert placement.get_tensor_memory(
                sink.inputs["conv_halo"]) is source_memory
            assert source_memory is not placement.get_tensor_memory(
                sink.outputs["y"])
        assert merges[0].input_bytes == 0
        assert merges[1].input_bytes > 0

    assert not hasattr(layers[3], "_kda_cp_allgathers")
    graph.validate()
    placement.validate(graph)


def test_decode_uses_context_cp_for_mla_and_batch_cp_for_kda(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 4)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=8, seq_prefill=128, decode=True)
    graph, placement = optimization.optimize_model_cluster_decode(
        graph, layers, B300Cluster(n_nodes=1), emb, read_input, kv_reads,
        output_head, seq_prefill=128, cp=2, dp=4, ep=8,
        pp_partition=[4], n_gpus=8)

    for layer in layers[:3]:
        copies = layer._decode_copies["cp_dp"]["sa"]
        assert len(copies) == 8
        assert all(kernel.B == 1 and kernel.S == 1 for kernel in copies)
        assert all(kernel.mode == "recurrent" for kernel in copies)
        assert not hasattr(layer, "_kda_cp_allgathers")

    mla_copies = layers[3]._decode_copies["cp_dp"]["sa"]
    assert len(mla_copies) == 8
    assert all(kernel.B == 2 for kernel in mla_copies)
    assert all(kernel.S_kv == 64 for kernel in mla_copies)
    assert all(kernel.inputs["kv"].dtype == "fp8" for kernel in mla_copies)
    barrier = layers[0]._kv_persist_barrier_cp_dp_copies[0]
    assert barrier.inputs["kv0"].shape == (1, 96, 128, 128)
    assert barrier.inputs["kv0_conv"].shape == (1, 96, 3, 128, 4)
    assert barrier.inputs["kv3"].shape == (2, 64, 576)
    graph.validate()
    placement.validate(graph)


def test_pareto_model_selection_supports_kimi_k3():
    finder._select_model("kimi_k3")

    assert finder.N_LAYERS == 93
    assert finder.N_EXPERTS == 896
    assert finder.WINDOW is None
    assert finder.COMPRESS_RATIOS == ()


@pytest.mark.parametrize("decode", [False, True])
def test_one_layer_cluster_graph_simulates(monkeypatch, decode):
    monkeypatch.setattr(model, "N_LAYERS", 1)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=8, seq_prefill=128, decode=decode)
    hardware = B300Cluster(n_nodes=1)
    kwargs = dict(cp=2, dp=4, ep=8, pp_partition=[1], n_gpus=8)
    if decode:
        graph, placement = optimization.optimize_model_cluster_decode(
            graph, layers, hardware, emb, read_input, kv_reads,
            output_head, seq_prefill=128, **kwargs)
    else:
        graph, placement = optimization.optimize_model_cluster_prefill(
            graph, layers, hardware, emb, read_input, output_head, **kwargs)

    result = Simulator(graph, placement, hardware).run()
    assert result.total_time_us > 0
    if not decode:
        remote_sinks = layers[0]._sa_cp_dp_copies[1::2]
        assert all(
            next(entry for entry in result.trace if entry.kernel is sink)
            .network_elapsed_time_us > 0
            for sink in remote_sinks
        )


def test_decode_kda_bridge_crosses_pipeline_boundary(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 2)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=8, seq_prefill=128, decode=True)
    hardware = B300Cluster(n_nodes=2)
    graph, placement = optimization.optimize_model_cluster_decode(
        graph, layers, hardware, emb, read_input, kv_reads, output_head,
        seq_prefill=128, cp=2, dp=4, ep=8,
        pp_partition=[1, 1], n_gpus=16)

    assert len(layers[1]._decode_copies["cp_dp"]["bridge"]) == 8
    assert len(layers[1]._pp_block_residual_sends) == 8
    assert Simulator(graph, placement, hardware).run().total_time_us > 0


@pytest.mark.parametrize(
    ("pp_partition", "boundary_id", "copy_group"),
    [([3, 10], 3, "dp"), ([6, 7], 6, "cp_dp")],
)
def test_decode_attn_residual_crosses_pipeline_boundary(
        monkeypatch, pp_partition, boundary_id, copy_group):
    monkeypatch.setattr(model, "N_LAYERS", 13)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=1, seq_prefill=128, decode=True)
    hardware = B300Cluster(n_nodes=1)

    graph, placement = optimization.optimize_model_cluster_decode(
        graph, layers, hardware, emb, read_input, kv_reads, output_head,
        seq_prefill=128, cp=1, dp=1, ep=1,
        pp_partition=pp_partition, n_gpus=2)

    boundary = layers[boundary_id]
    assert len(boundary._pp_block_residual_sends) == 1
    destination_memories = {
        placement.get_tensor_memory(copy.inputs["x"])
        for copy in boundary._decode_copies[copy_group]["block_in_fan"]
    }
    assert {memory.name for memory in destination_memories} == {
        "n0-hbm3e-1",
    }
    assert Simulator(graph, placement, hardware).run().total_time_us > 0


def test_decode_existing_cp_comm_materializes_pp_residual(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 25)
    graph, layers, emb, read_input, kv_reads, output_head = \
        model.declare_model(batch_size=2, seq_prefill=128, decode=True)
    hardware = B300Cluster(n_nodes=1)

    graph, placement = optimization.optimize_model_cluster_decode(
        graph, layers, hardware, emb, read_input, kv_reads, output_head,
        seq_prefill=128, cp=2, dp=1, ep=2,
        pp_partition=[24, 1], n_gpus=4)

    boundary = layers[24]
    assert boundary.is_kda
    assert boundary._pp_block_residual_sends == []
    block_in_fans = boundary._decode_copies["cp_dp"]["block_in_fan"]
    assert len(block_in_fans) == 2
    assert all(isinstance(graph._in_edges(fan)[0].src, Scatter)
               for fan in block_in_fans)
    destination_memories = {
        placement.get_tensor_memory(fan.inputs["x"]).name
        for fan in block_in_fans
    }
    assert destination_memories == {"n0-hbm3e-2", "n0-hbm3e-3"}
    graph.validate()
    placement.validate(graph)


def test_prefill_attn_residual_crosses_pipeline_boundary(monkeypatch):
    monkeypatch.setattr(model, "N_LAYERS", 13)
    graph, layers, emb, read_input, _, output_head = model.declare_model(
        batch_size=8, seq_prefill=128, decode=False)
    hardware = B300Cluster(n_nodes=2)

    graph, placement = optimization.optimize_model_cluster_prefill(
        graph, layers, hardware, emb, read_input, output_head,
        cp=2, dp=4, ep=8, pp_partition=[6, 7], n_gpus=16)

    boundary = layers[6]
    assert len(boundary._pp_block_residual_sends) == 8
    destination_memories = {
        placement.get_tensor_memory(copy.inputs["x"])
        for copy in boundary._block_in_fan_cp_dp_copies
    }
    assert all(memory.name.startswith("n1-")
               for memory in destination_memories)
    graph.validate()
    placement.validate(graph)
