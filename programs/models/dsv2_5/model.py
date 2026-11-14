# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""DeepSeek V2.5 inference — Declaration Phase.

Builds the logical compute graph (add_kernel + add_data_edge).
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import (
    ElementwiseOp, Embedding, Gemm, MlaAttn, Nop, PartialRMSNorm,
    ReadInput, RMSNorm, Sampling, Slice, StridedGemm, TokenCombine,
    TokenDispatch,
)
from rooflang.language.kernels.identity import Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor

from rooflang.programs.models.dsv2_5.config import (
    BATCH, D, DENSE_INTER, DENSE_LAYERS, H, KV_CACHE_DIM, KV_LORA,
    MOE_INTER, N_EXPERTS, N_LAYERS, QK_HD, QK_NOPE_HD, Q_LORA,
    ROUTER_SCORING_FUNC, S_PREFILL, SHARED_INTER, TOPK, V, V_HD,
)


# ── Kernel factories ────────────────────────────────────────────────────

def _make_gemm(B, S, N, K, out_dtype="bf16"):
    """BF16-weight GEMM with BF16 activations and configurable output."""
    k = Gemm(B * S, N, K, "bf16", "bf16", out_dtype)
    k.inputs = {"x": Tensor("bf16", (B, S, K))}
    k.weights = {"w": Tensor("bf16", (K, N))}
    k.outputs = {"y": Tensor(out_dtype, (B, S, N))}
    return k


def _make_norm(B, S, dim):
    k = RMSNorm(B * S, dim, "bf16")
    k.inputs = {"x": Tensor("bf16", (B, S, dim))}
    k.weights = {"g": Tensor("bf16", (dim,))}
    k.outputs = {"y": Tensor("bf16", (B, S, dim))}
    return k


def _make_partial_norm(B, S, input_dim, norm_dim):
    """RMS-normalize the latent prefix while forwarding the RoPE suffix."""
    k = PartialRMSNorm(B * S, input_dim, norm_dim, "bf16")
    k.inputs = {"x": Tensor("bf16", (B, S, input_dim))}
    k.weights = {"g": Tensor("bf16", (norm_dim,))}
    k.outputs = {"y": Tensor("bf16", (B, S, input_dim))}
    return k


def _make_gated_up(B, S, N, K):
    """SwiGLU fused gate+up: 2·M·(2N)·K flops, writes M·N output."""
    M = B * S
    k = StridedGemm(M, 2 * N, K, "bf16", "bf16", "bf16",
                    out_elems=M * N)
    k.inputs = {"x": Tensor("bf16", (B, S, K))}
    k.weights = {"w": Tensor("bf16", (K, 2 * N))}
    k.outputs = {"y": Tensor("bf16", (B, S, N))}
    return k


# ── Per-layer metadata for optimization phase ───────────────────────────

@dataclass
class LayerMeta:
    bridge: Kernel = None
    attn_norm: Kernel = None
    attn_fan: Kernel = None
    wq_a: Kernel = None
    q_norm: Kernel = None
    wq_b: Kernel = None
    wkv: Kernel = None
    kv_norm: Kernel = None
    sa: Kernel = None
    wo: Kernel = None
    attn_add: Kernel = None
    ffn_bridge: Kernel = None
    ffn_norm: Kernel = None
    ffn_fan: Kernel = None
    dense_up: Kernel = None
    dense_down: Kernel = None
    gate: Kernel = None
    dispatch: Kernel = None
    combine: Kernel = None
    sw_up: Kernel = None
    sw_down: Kernel = None
    moe_add: Kernel = None
    ffn_add: Kernel = None
    experts: List[Kernel] = field(default_factory=list)
    # KV cache and persistence metadata
    kv_cache_fan: Kernel = None
    kv_persist_fan: Kernel = None
    kv_persist_barrier: Kernel = None
    kv_sink: Kernel = None

    @property
    def is_dense(self):
        return self.dense_up is not None


def _tag_weights(kernel, layer_id, name):
    """Tag kernel's weight Tensors with a weight_id for simulator dedup."""
    for port, t in kernel.weights.items():
        t.weight_id = f"L{layer_id}_{name}_{port}"


_WEIGHTED_LAYER_FIELDS = (
    "attn_norm", "wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "sa",
    "wo", "ffn_norm", "dense_up", "dense_down", "gate", "sw_up",
    "sw_down",
)


def _tag_layer_weights(layer, layer_id):
    """Tag all shared weights belonging to one transformer layer."""
    for name in _WEIGHTED_LAYER_FIELDS:
        kernel = getattr(layer, name)
        if kernel is not None:
            _tag_weights(kernel, layer_id, name)
    for eid in range(len(layer.experts) // 2):
        _tag_weights(layer.experts[eid * 2], layer_id,
                     f"expert{eid}_up")
        _tag_weights(layer.experts[eid * 2 + 1], layer_id,
                     f"expert{eid}_down")


def _build_token_input(g, B, S):
    """Build the token reader and embedding shared by both stages."""
    M = B * S
    read_input = ReadInput(M, "int32")
    read_input.inputs = {"tokens": Tensor("int32", (B, S, 1))}
    read_input.outputs = {"tokens": Tensor("int32", (B, S, 1))}
    g.add_kernel(read_input)

    emb = Embedding(M, V, D)
    emb.inputs = {"idx": Tensor("int32", (B, S, 1))}
    emb.weights = {"emb": Tensor("bf16", (V, D))}
    emb.outputs = {"y": Tensor("bf16", (B, S, D))}
    g.add_kernel(emb)
    g.add_data_edge(read_input, emb, {"tokens": "idx"})
    _tag_weights(emb, -1, "emb")
    return read_input, emb


def _build_output_head(g, B, hidden_src):
    """Build the shared final norm, logits projection, and sampler."""
    final_norm = _make_norm(B, 1, D)
    g.add_kernel(final_norm)
    g.add_data_edge(hidden_src, final_norm, {"y": "x"})
    _tag_weights(final_norm, -1, "final_norm")

    logits = _make_gemm(B, 1, V, D)
    g.add_kernel(logits)
    g.add_data_edge(final_norm, logits, {"y": "x"})
    _tag_weights(logits, -1, "logits")

    sampling = Sampling(B, V)
    sampling.inputs = {"logits": Tensor("bf16", (B, 1, V))}
    sampling.outputs = {"y": Tensor("int32", (B, 1, 1))}
    g.add_kernel(sampling)
    g.add_data_edge(logits, sampling, {"y": "logits"})

    return final_norm, logits, sampling


def _build_layers(g, B, S, context_len, prev_out):
    """Build N_LAYERS transformer layers into graph g.

    Args:
        g: ComputeGraph to add kernels to.
        B: batch size.
        S: sequence length.
        context_len: None for prefill mode. For decode mode, the persistent
            prefix length whose latent KV cache is consumed.
        prev_out: kernel whose "y" output feeds into the first layer.
    Returns (layers, last_output_kernel).

    N_LAYERS=60 covers every decoder layer; DeepSeek-V2.5 has no MTP layer.
    """
    M = B * S
    layers = []
    is_prefill = context_len is None

    for layer_id in range(N_LAYERS):
        cache_len = S if is_prefill else context_len
        L = LayerMeta()

        # ── Input fan-out (residual + attention) ──────────────────
        bridge = Spawn(world=2)
        bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
        bridge.outputs = {
            "y": Tensor("bf16", (B, S, D)),
            "y2": Tensor("bf16", (B, S, D)),
        }
        g.add_kernel(bridge)
        g.add_data_edge(prev_out, bridge, {"y": "x"})
        L.bridge = bridge

        # ── Attention ─────────────────────────────────────────────
        attn_norm = _make_norm(B, S, D)
        g.add_kernel(attn_norm)
        g.add_data_edge(bridge, attn_norm, {"y": "x"})
        L.attn_norm = attn_norm

        # Fan-out after norm: Q path + newly computed KV path.
        attn_fan = Spawn(world=2)
        attn_fan.inputs = {"x": Tensor("bf16", (B, S, D))}
        attn_fan.outputs = {"y": Tensor("bf16", (B, S, D)),
                            "y2": Tensor("bf16", (B, S, D))}
        g.add_kernel(attn_fan)
        g.add_data_edge(attn_norm, attn_fan, {"y": "x"})
        L.attn_fan = attn_fan

        # Q path
        wq_a = _make_gemm(B, S, Q_LORA, D)
        g.add_kernel(wq_a)
        g.add_data_edge(attn_fan, wq_a, {"y": "x"})
        L.wq_a = wq_a

        q_norm = _make_norm(B, S, Q_LORA)
        g.add_kernel(q_norm)
        g.add_data_edge(wq_a, q_norm, {"y": "x"})
        L.q_norm = q_norm

        wq_b = _make_gemm(B, S, H * QK_HD, Q_LORA)
        g.add_kernel(wq_b)
        g.add_data_edge(q_norm, wq_b, {"y": "x"})
        L.wq_b = wq_b

        # KV path.  The 512 latent rank and the 64 RoPE dims share the cache.
        wkv = _make_gemm(B, S, KV_CACHE_DIM, D)
        g.add_kernel(wkv)
        g.add_data_edge(attn_fan, wkv, {"y2": "x"})
        L.wkv = wkv

        kv_norm = _make_partial_norm(B, S, KV_CACHE_DIM, KV_LORA)
        g.add_kernel(kv_norm)
        g.add_data_edge(wkv, kv_norm, {"y": "x"})
        L.kv_norm = kv_norm

        sa = MlaAttn(
            B, H, S, cache_len, QK_HD, V_HD, KV_CACHE_DIM, KV_LORA,
            QK_NOPE_HD, dtype="bf16", kv_transform_dtype="bf16",
            q_dtype="bf16", kv_dtype="bf16", out_dtype="bf16",
            causal=is_prefill,
        )
        sa.inputs = {
            "q": Tensor("bf16", (B, S, H * QK_HD)),
            "kv": Tensor("bf16", (B, cache_len, KV_CACHE_DIM)),
        }
        sa.weights = {
            "kv_b": Tensor("bf16",
                           (KV_LORA, H * (QK_NOPE_HD + V_HD))),
        }
        sa.outputs = {"y": Tensor("bf16", (B, S, H * V_HD))}
        g.add_kernel(sa)
        g.add_data_edge(wq_b, sa, {"y": "q"})
        L.sa = sa

        if is_prefill:
            kv_cache_fan = Spawn(world=2)
            kv_cache_fan.inputs = {
                "x": Tensor("bf16", (B, S, KV_CACHE_DIM))}
            kv_cache_fan.outputs = {
                "y": Tensor("bf16", (B, S, KV_CACHE_DIM)),
                "y2": Tensor("bf16", (B, S, KV_CACHE_DIM)),
            }
            g.add_kernel(kv_cache_fan)
            g.add_data_edge(kv_norm, kv_cache_fan, {"y": "x"})
            g.add_data_edge(kv_cache_fan, sa, {"y": "kv"})
            L.kv_cache_fan = kv_cache_fan
            L.kv_persist_fan = kv_cache_fan
        else:
            kv_sink = Nop()
            kv_sink.inputs = {
                "kv": Tensor("bf16", (B, S, KV_CACHE_DIM))}
            g.add_kernel(kv_sink)
            g.add_data_edge(kv_norm, kv_sink, {"y": "kv"})
            L.kv_sink = kv_sink

        wo = _make_gemm(B, S, D, H * V_HD)
        g.add_kernel(wo)
        g.add_data_edge(sa, wo, {"y": "x"})
        L.wo = wo

        attn_add = ElementwiseOp(M, D, "bf16")
        attn_add.inputs = {
            "a": Tensor("bf16", (B, S, D)),
            "b": Tensor("bf16", (B, S, D)),
        }
        attn_add.outputs = {"y": Tensor("bf16", (B, S, D))}
        g.add_kernel(attn_add)
        g.add_data_edge(bridge, attn_add, {"y2": "a"})
        g.add_data_edge(wo, attn_add, {"y": "b"})
        L.attn_add = attn_add

        # ── FFN residual fan-out ──────────────────────────────────
        ffn_bridge = Spawn(world=2)
        ffn_bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
        ffn_bridge.outputs = {"y": Tensor("bf16", (B, S, D)),
                              "y2": Tensor("bf16", (B, S, D))}
        g.add_kernel(ffn_bridge)
        g.add_data_edge(attn_add, ffn_bridge, {"y": "x"})
        L.ffn_bridge = ffn_bridge

        # ── FFN / MoE ─────────────────────────────────────────────
        ffn_norm = _make_norm(B, S, D)
        g.add_kernel(ffn_norm)
        g.add_data_edge(ffn_bridge, ffn_norm, {"y": "x"})
        L.ffn_norm = ffn_norm

        if layer_id < DENSE_LAYERS:
            dense_up = _make_gated_up(B, S, DENSE_INTER, D)
            g.add_kernel(dense_up)
            g.add_data_edge(ffn_norm, dense_up, {"y": "x"})
            L.dense_up = dense_up

            dense_down = _make_gemm(B, S, D, DENSE_INTER)
            g.add_kernel(dense_down)
            g.add_data_edge(dense_up, dense_down, {"y": "x"})
            L.dense_down = dense_down

            ffn_add = ElementwiseOp(M, D, "bf16")
            ffn_add.inputs = {
                "a": Tensor("bf16", (B, S, D)),
                "b": Tensor("bf16", (B, S, D)),
            }
            ffn_add.outputs = {"y": Tensor("bf16", (B, S, D))}
            g.add_kernel(ffn_add)
            g.add_data_edge(ffn_bridge, ffn_add, {"y2": "a"})
            g.add_data_edge(dense_down, ffn_add, {"y": "b"})
            L.ffn_add = ffn_add

            _tag_layer_weights(L, layer_id)
            prev_out = ffn_add
            layers.append(L)
            continue

        # FFN fan-out: gate + dispatch + shared expert (3 consumers)
        ffn_fan = Spawn(world=3)
        ffn_fan.inputs = {"x": Tensor("bf16", (B, S, D))}
        ffn_fan.outputs = {"y": Tensor("bf16", (B, S, D)),
                           "y2": Tensor("bf16", (B, S, D)),
                           "y3": Tensor("bf16", (B, S, D))}
        g.add_kernel(ffn_fan)
        g.add_data_edge(ffn_norm, ffn_fan, {"y": "x"})
        L.ffn_fan = ffn_fan

        # Router scores stay FP32, matching MoEGate's softmax accumulation.
        gate = _make_gemm(B, S, N_EXPERTS, D, out_dtype="fp32")
        g.add_kernel(gate)
        g.add_data_edge(ffn_fan, gate, {"y": "x"})
        L.gate = gate

        # Dispatch: softmax routing + token scatter to experts
        M_e = Fraction(M * TOPK, N_EXPERTS)
        dispatch = TokenDispatch(
            M, D, N_EXPERTS, TOPK,
            scoring_func=ROUTER_SCORING_FUNC)
        dispatch.inputs = {"x": Tensor("bf16", (B, S, D)),
                           "routing": Tensor("fp32", (B, S, N_EXPERTS))}
        dispatch.outputs = {f"o{i}": Tensor("bf16", (M_e, D))
                            for i in range(N_EXPERTS)}
        g.add_kernel(dispatch)
        g.add_data_edge(gate, dispatch, {"y": "routing"})
        g.add_data_edge(ffn_fan, dispatch, {"y2": "x"})
        L.dispatch = dispatch

        # Combine: weighted sum of expert outputs
        combine = TokenCombine(M, D, N_EXPERTS, TOPK)
        combine.inputs = {f"i{i}": Tensor("bf16", (M_e, D))
                          for i in range(N_EXPERTS)}
        combine.outputs = {"y": Tensor("bf16", (B, S, D))}
        g.add_kernel(combine)
        L.combine = combine

        # Expert kernels (fused gate+up and down projection per expert)
        L.experts = []
        for eid in range(N_EXPERTS):
            up = StridedGemm(M_e, 2 * MOE_INTER, D, "bf16", "bf16",
                             out_elems=M_e * MOE_INTER)
            up.inputs = {"x": Tensor("bf16", (M_e, D))}
            up.weights = {"w": Tensor("bf16", (D, 2 * MOE_INTER))}
            up.outputs = {"y": Tensor("bf16", (M_e, MOE_INTER))}
            g.add_kernel(up)

            down = Gemm(M_e, D, MOE_INTER, "bf16", "bf16")
            down.inputs = {"x": Tensor("bf16", (M_e, MOE_INTER))}
            down.weights = {"w": Tensor("bf16", (MOE_INTER, D))}
            down.outputs = {"y": Tensor("bf16", (M_e, D))}
            g.add_kernel(down)
            g.add_data_edge(up, down, {"y": "x"})
            g.add_data_edge(dispatch, up, {f"o{eid}": "x"})
            g.add_data_edge(down, combine, {"y": f"i{eid}"})

            L.experts.extend([up, down])

        # Shared experts (2 x 1536, parallel with routed experts)
        sw_up = _make_gated_up(B, S, SHARED_INTER, D)
        g.add_kernel(sw_up)
        g.add_data_edge(ffn_fan, sw_up, {"y3": "x"})
        L.sw_up = sw_up

        sw_down = _make_gemm(B, S, D, SHARED_INTER)
        g.add_kernel(sw_down)
        g.add_data_edge(sw_up, sw_down, {"y": "x"})
        L.sw_down = sw_down

        # ── MoE output: routed + shared experts ───────────────────
        moe_add = ElementwiseOp(M, D, "bf16")
        moe_add.inputs = {"a": Tensor("bf16", (B, S, D)),
                          "b": Tensor("bf16", (B, S, D))}
        moe_add.outputs = {"y": Tensor("bf16", (B, S, D))}
        g.add_kernel(moe_add)
        g.add_data_edge(combine, moe_add, {"y": "a"})
        g.add_data_edge(sw_down, moe_add, {"y": "b"})
        L.moe_add = moe_add

        # ── FFN residual: attn_residual + moe_output ─────────────
        ffn_add = ElementwiseOp(M, D, "bf16")
        ffn_add.inputs = {"a": Tensor("bf16", (B, S, D)),
                          "b": Tensor("bf16", (B, S, D))}
        ffn_add.outputs = {"y": Tensor("bf16", (B, S, D))}
        g.add_kernel(ffn_add)
        g.add_data_edge(ffn_bridge, ffn_add, {"y2": "a"})
        g.add_data_edge(moe_add, ffn_add, {"y": "b"})
        L.ffn_add = ffn_add

        # ── Weight tagging (for simulator dedup) ─────────────────────
        _tag_layer_weights(L, layer_id)

        prev_out = ffn_add
        layers.append(L)

    return layers, prev_out


def _build_decode_kv_cache_read(g, B, context_len, layers):
    """Build persistent external latent-KV reads for decode attention."""
    kv_cache_reads = []
    for layer in layers:
        kv_read = ReadInput(B * context_len * KV_CACHE_DIM, "bf16")
        kv_read.inputs = {
            "kv": Tensor("bf16", (B, context_len, KV_CACHE_DIM))}
        kv_read.outputs = {
            "y": Tensor("bf16", (B, context_len, KV_CACHE_DIM))}
        g.add_kernel(kv_read)

        cache_fan = Spawn(world=2)
        cache_fan.inputs = {
            "x": Tensor("bf16", (B, context_len, KV_CACHE_DIM))}
        cache_fan.outputs = {
            "y": Tensor("bf16", (B, context_len, KV_CACHE_DIM)),
            "y2": Tensor("bf16", (B, context_len, KV_CACHE_DIM)),
        }
        g.add_kernel(cache_fan)
        g.add_data_edge(kv_read, cache_fan, {"y": "x"})
        g.add_data_edge(cache_fan, layer.sa, {"y": "kv"})

        layer.kv_cache_fan = cache_fan
        layer.kv_persist_fan = cache_fan
        kv_cache_reads.append(kv_read)
    return kv_cache_reads


def _build_kv_persistence_barrier(
    g, layers, output_src, output_name,
):
    """Keep every latent KV cache alive until the stage output is ready."""
    barrier = Nop()
    barrier.inputs = {}
    for layer_id, layer in enumerate(layers):
        main_cache = layer.kv_persist_fan.outputs["y2"]
        barrier.inputs[f"kv{layer_id}"] = Tensor(
            main_cache.dtype, main_cache.shape)
    barrier.inputs[output_name] = Tensor(
        output_src.outputs["y"].dtype, output_src.outputs["y"].shape)
    barrier.outputs = {"done": Tensor("int32", (1,))}
    g.add_kernel(barrier)

    for layer_id, layer in enumerate(layers):
        g.add_data_edge(
            layer.kv_persist_fan, barrier, {"y2": f"kv{layer_id}"})
        layer.kv_persist_barrier = barrier
    g.add_data_edge(output_src, barrier, {"y": output_name})
    return barrier


def declare_model(
    batch_size=BATCH,
    seq_prefill=S_PREFILL,
    decode=False,
):
    """Build either a prefill graph or a single-step decode graph.

    Args:
        batch_size: number of independent sequences (>= 64 for MoE routing).
        seq_prefill: prefill sequence length, or the persistent prefix length
            whose latent KV cache is consumed by decode.
        decode: whether to build the single-step decode stage.

    Returns:
        (g, layers, emb, read_input, kv_cache_reads, output_head)
        - layers: list[LayerMeta] for the selected stage
        - emb/read_input: token input kernels shared by both stages
        - kv_cache_reads: per-layer latent KV ReadInput kernels (decode only)
        - output_head: (last_token, final_norm, logits, sampling) for prefill;
            (final_norm, logits, sampling) for decode
    """
    is_prefill = not decode

    g = ComputeGraph()
    B = batch_size
    S = seq_prefill if is_prefill else 1
    context_len = None if is_prefill else seq_prefill

    read_input, emb = _build_token_input(g, B, S)
    layers, last_output = _build_layers(g, B, S, context_len, emb)

    kv_cache_reads = []
    output_head = ()
    if is_prefill:
        last_token = Slice()
        last_token.inputs = {
            "x": Tensor("bf16", (B, S, D))}
        last_token.outputs = {
            "y": Tensor("bf16", (B, 1, D))}
        g.add_kernel(last_token)
        g.add_data_edge(last_output, last_token, {"y": "x"})
        output_head = (
            last_token, *_build_output_head(g, B, last_token))

        _build_kv_persistence_barrier(
            g, layers, output_head[-1], "prefill_output")
    else:
        kv_cache_reads = _build_decode_kv_cache_read(
            g, B, context_len, layers)
        output_head = _build_output_head(g, B, last_output)
        _build_kv_persistence_barrier(
            g, layers, output_head[-1], "decode_output")

    g.validate()
    return g, layers, emb, read_input, kv_cache_reads, output_head
