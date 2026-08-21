"""GLM-5.2-FP8 inference — Declaration Phase.

Builds the logical compute graph (add_kernel + add_data_edge).
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import (
    ElementwiseOp, Embedding, Gemm, Glm52SparseAttn, LayerNorm, Nop,
    ReadInput, RMSNorm, Sampling, Slice, StridedGemm, TokenCombine,
    TokenDispatch,
)
from rooflang.language.kernels.identity import Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor
from rooflang.language.utils import gemm_scale_bytes

from rooflang.programs.models.glm_5_2_fp8.config import (
    BATCH, D, DENSE_INTER, DENSE_LAYERS, FULL_INDEXER_LAYERS, H, INDEX_H,
    INDEX_HD, INDEX_TOPK, KV_CACHE_DIM, KV_LORA, MOE_INTER, N_EXPERTS,
    N_LAYERS, QK_HD, QK_NOPE_HD, Q_LORA,
    ROUTER_SCORING_FUNC, S_PREFILL, TOPK, V, V_HD,
)


# ── Kernel factories ────────────────────────────────────────────────────

def _make_gemm(B, S, N, K, w_dtype, a_dtype="bf16", out_dtype="bf16"):
    M = B * S
    k = Gemm(M, N, K, w_dtype, a_dtype, out_dtype)
    k.inputs = {"x": Tensor(a_dtype, (B, S, K))}
    k.weights = {"w": Tensor(w_dtype, (K, N))}
    scale_bytes = gemm_scale_bytes(N, K, w_dtype)
    if scale_bytes > 0:
        k.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
    k.outputs = {"y": Tensor(out_dtype, (B, S, N))}
    return k


def _make_norm(B, S, dim):
    M = B * S
    k = RMSNorm(M, dim, "bf16")
    k.inputs = {"x": Tensor("bf16", (B, S, dim))}
    k.weights = {"g": Tensor("bf16", (dim,))}
    k.outputs = {"y": Tensor("bf16", (B, S, dim))}
    return k


def _make_layer_norm(B, S, dim):
    k = LayerNorm(B * S, dim, "bf16")
    k.inputs = {"x": Tensor("bf16", (B, S, dim))}
    k.weights = {
        "g": Tensor("bf16", (dim,)),
        "b": Tensor("bf16", (dim,)),
    }
    k.outputs = {"y": Tensor("bf16", (B, S, dim))}
    return k


def _make_gated_up(B, S, N, K, w_dtype, a_dtype="bf16", out_dtype="bf16"):
    """SwiGLU fused gate+up: 2·M·(2N)·K flops, writes M·N output."""
    M = B * S
    k = StridedGemm(M, 2 * N, K, w_dtype, a_dtype, out_dtype, out_elems=M * N)
    k.inputs = {"x": Tensor(a_dtype, (B, S, K))}
    k.weights = {"w": Tensor(w_dtype, (K, 2 * N))}
    scale_bytes = gemm_scale_bytes(2 * N, K, w_dtype)
    if scale_bytes > 0:
        k.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
    k.outputs = {"y": Tensor(out_dtype, (B, S, N))}
    return k


# ── Per-layer metadata for optimization phase ───────────────────────────

@dataclass
class LayerMeta:
    bridge: Kernel = None
    attn_norm: Kernel = None
    attn_fan: Kernel = None
    wq_a: Kernel = None
    q_norm: Kernel = None
    q_fan: Kernel = None
    wq_b: Kernel = None
    wkv: Kernel = None
    kv_norm: Kernel = None
    kv_cache_quant: Kernel = None
    sa: Kernel = None
    wo: Kernel = None
    attn_add: Kernel = None
    index_wq: Kernel = None
    index_wk: Kernel = None
    index_norm: Kernel = None
    index_weights: Kernel = None
    index_cache_quant: Kernel = None
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
    index_cache_fan: Kernel = None
    index_cache_read: Kernel = None
    index_sink: Kernel = None

    @property
    def has_full_indexer(self):
        return self.sa.indexer_mode == "full"

    @property
    def is_dense(self):
        return self.dense_up is not None


def _tag_weights(kernel, layer_id, name):
    """Tag kernel's weight Tensors with a weight_id for simulator dedup."""
    for port, t in kernel.weights.items():
        t.weight_id = f"L{layer_id}_{name}_{port}"


_WEIGHTED_LAYER_FIELDS = (
    "attn_norm", "wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "sa",
    "wo", "index_wq", "index_wk", "index_norm", "index_weights",
    "ffn_norm", "dense_up", "dense_down", "gate", "sw_up", "sw_down",
)


def _tag_layer_weights(layer, layer_id):
    """Tag all shared weights belonging to one transformer layer."""
    for name in _WEIGHTED_LAYER_FIELDS:
        kernel = getattr(layer, name)
        if kernel is not None:
            _tag_weights(kernel, layer_id, name)
    for eid in range(len(layer.experts) // 2):
        _tag_weights(layer.experts[eid * 2], layer_id, f"expert{eid}_up")
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

    logits = _make_gemm(B, 1, V, D, "bf16")
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
            prefix length whose full main/index caches are consumed.
        prev_out: kernel whose "y" output feeds into the first layer.
    Returns (layers, last_output_kernel).

    The checkpoint's MTP layer 78 is intentionally excluded by N_LAYERS=78.
    """
    M = B * S
    layers = []
    is_prefill = context_len is None

    for layer_id in range(N_LAYERS):
        has_indexer = layer_id in FULL_INDEXER_LAYERS
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

        attn_world = 4 if has_indexer else 2
        attn_fan = Spawn(world=attn_world)
        attn_fan.inputs = {"x": Tensor("bf16", (B, S, D))}
        attn_fan.outputs = {
            "y": Tensor("bf16", (B, S, D)),
            "y2": Tensor("bf16", (B, S, D)),
        }
        if has_indexer:
            attn_fan.outputs.update({
                "y3": Tensor("bf16", (B, S, D)),
                "y4": Tensor("bf16", (B, S, D)),
            })
        g.add_kernel(attn_fan)
        g.add_data_edge(attn_norm, attn_fan, {"y": "x"})
        L.attn_fan = attn_fan

        # Main query path
        wq_a = _make_gemm(B, S, Q_LORA, D, "fp8")
        g.add_kernel(wq_a)
        g.add_data_edge(attn_fan, wq_a, {"y": "x"})
        L.wq_a = wq_a

        q_norm = _make_norm(B, S, Q_LORA)
        g.add_kernel(q_norm)
        g.add_data_edge(wq_a, q_norm, {"y": "x"})
        L.q_norm = q_norm

        q_source = q_norm
        q_source_port = "y"
        if has_indexer:
            q_fan = Spawn(world=2)
            q_fan.inputs = {"x": Tensor("bf16", (B, S, Q_LORA))}
            q_fan.outputs = {
                "y": Tensor("bf16", (B, S, Q_LORA)),
                "y2": Tensor("bf16", (B, S, Q_LORA)),
            }
            g.add_kernel(q_fan)
            g.add_data_edge(q_norm, q_fan, {"y": "x"})
            L.q_fan = q_fan
            q_source = q_fan

        wq_b = _make_gemm(B, S, H * QK_HD, Q_LORA, "fp8")
        g.add_kernel(wq_b)
        g.add_data_edge(q_source, wq_b, {q_source_port: "x"})
        L.wq_b = wq_b

        # Main KV path. RoPE is intentionally omitted, matching dsv4_pro's
        # modeling granularity; the aggregate 576-dimensional cache remains.
        wkv = _make_gemm(B, S, KV_CACHE_DIM, D, "fp8")
        g.add_kernel(wkv)
        g.add_data_edge(attn_fan, wkv, {"y2": "x"})
        L.wkv = wkv

        kv_norm = _make_norm(B, S, KV_CACHE_DIM)
        g.add_kernel(kv_norm)
        g.add_data_edge(wkv, kv_norm, {"y": "x"})
        L.kv_norm = kv_norm

        sa = Glm52SparseAttn(
            B, H, S, min(INDEX_TOPK, cache_len), cache_len,
            QK_HD, V_HD, KV_CACHE_DIM,
            dtype="bf16", kv_lora_rank=KV_LORA,
            qk_nope_head_dim=QK_NOPE_HD, kv_transform_dtype="fp8",
            indexer_mode="full" if has_indexer else "shared",
            indexer_s_kv=cache_len if has_indexer else 0,
            indexer_h=INDEX_H if has_indexer else 0,
            indexer_hd=INDEX_HD if has_indexer else 0,
            indexer_dtype="fp8", indexer_compute_dtype="fp8",
            indexer_reduce_dtype="fp32",
            q_dtype="bf16", kv_dtype="fp8", out_dtype="bf16",
            index_q_dtype="bf16", index_weight_dtype="fp32",
            causal=is_prefill,
        )
        sa.inputs = {
            "q": Tensor("bf16", (B, S, H * QK_HD)),
            "kv": Tensor("fp8", (B, cache_len, KV_CACHE_DIM)),
        }
        sa.weights = {
            "kv_b": Tensor(
                "fp8", (KV_LORA, H * (QK_NOPE_HD + V_HD))),
        }
        scale_bytes = gemm_scale_bytes(
            H * (QK_NOPE_HD + V_HD), KV_LORA, "fp8")
        if scale_bytes > 0:
            sa.weights["kv_b_scale"] = Tensor("ue8m0", (int(scale_bytes),))
        sa.outputs = {"y": Tensor("bf16", (B, S, H * V_HD))}
        g.add_kernel(sa)
        g.add_data_edge(wq_b, sa, {"y": "q"})
        L.sa = sa

        if is_prefill:
            kv_cache_quant = Slice()
            kv_cache_quant.inputs = {
                "x": Tensor("bf16", (B, S, KV_CACHE_DIM))}
            kv_cache_quant.outputs = {
                "y": Tensor("fp8", (B, S, KV_CACHE_DIM))}
            g.add_kernel(kv_cache_quant)
            g.add_data_edge(kv_norm, kv_cache_quant, {"y": "x"})
            L.kv_cache_quant = kv_cache_quant

            kv_cache_fan = Spawn(world=2)
            kv_cache_fan.inputs = {
                "x": Tensor("fp8", (B, S, KV_CACHE_DIM))}
            kv_cache_fan.outputs = {
                "y": Tensor("fp8", (B, S, KV_CACHE_DIM)),
                "y2": Tensor("fp8", (B, S, KV_CACHE_DIM)),
            }
            g.add_kernel(kv_cache_fan)
            g.add_data_edge(kv_cache_quant, kv_cache_fan, {"y": "x"})
            g.add_data_edge(kv_cache_fan, sa, {"y": "kv"})
            L.kv_cache_fan = kv_cache_fan
            L.kv_persist_fan = kv_cache_fan
        else:
            kv_sink = Nop()
            kv_sink.inputs = {"kv": Tensor("bf16", (B, S, KV_CACHE_DIM))}
            g.add_kernel(kv_sink)
            g.add_data_edge(kv_norm, kv_sink, {"y": "kv"})
            L.kv_sink = kv_sink

        if has_indexer:
            # Index query is projected from q_a residual; key and head weights are
            # projected from the normalized hidden state.
            index_wq = _make_gemm(B, S, INDEX_H * INDEX_HD, Q_LORA, "fp8")
            g.add_kernel(index_wq)
            g.add_data_edge(q_fan, index_wq, {"y2": "x"})
            L.index_wq = index_wq

            index_wk = _make_gemm(B, S, INDEX_HD, D, "fp8")
            g.add_kernel(index_wk)
            g.add_data_edge(attn_fan, index_wk, {"y3": "x"})
            L.index_wk = index_wk

            index_norm = _make_layer_norm(B, S, INDEX_HD)
            g.add_kernel(index_norm)
            g.add_data_edge(index_wk, index_norm, {"y": "x"})
            L.index_norm = index_norm

            index_weights = _make_gemm(
                B, S, INDEX_H, D, "bf16", "bf16", "fp32")
            g.add_kernel(index_weights)
            g.add_data_edge(attn_fan, index_weights, {"y4": "x"})
            L.index_weights = index_weights

            sa.inputs.update({
                "index_q": Tensor(
                    "bf16", (B, S, INDEX_H * INDEX_HD)),
                "index_kv": Tensor("fp8", (B, cache_len, INDEX_HD)),
                "index_weights": Tensor("fp32", (B, S, INDEX_H)),
            })
            g.add_data_edge(index_wq, sa, {"y": "index_q"})
            g.add_data_edge(index_weights, sa, {"y": "index_weights"})

            if is_prefill:
                index_cache_quant = Slice()
                index_cache_quant.inputs = {
                    "x": Tensor("bf16", (B, S, INDEX_HD))}
                index_cache_quant.outputs = {
                    "y": Tensor("fp8", (B, S, INDEX_HD))}
                g.add_kernel(index_cache_quant)
                g.add_data_edge(index_norm, index_cache_quant, {"y": "x"})
                L.index_cache_quant = index_cache_quant

                index_cache_fan = Spawn(world=2)
                index_cache_fan.inputs = {
                    "x": Tensor("fp8", (B, S, INDEX_HD))}
                index_cache_fan.outputs = {
                    "y": Tensor("fp8", (B, S, INDEX_HD)),
                    "y2": Tensor("fp8", (B, S, INDEX_HD)),
                }
                g.add_kernel(index_cache_fan)
                g.add_data_edge(
                    index_cache_quant, index_cache_fan, {"y": "x"})
                g.add_data_edge(index_cache_fan, sa, {"y": "index_kv"})
                L.index_cache_fan = index_cache_fan
            else:
                index_sink = Nop()
                index_sink.inputs = {
                    "index_kv": Tensor("bf16", (B, S, INDEX_HD))}
                g.add_kernel(index_sink)
                g.add_data_edge(index_norm, index_sink, {"y": "index_kv"})
                L.index_sink = index_sink

        wo = _make_gemm(B, S, D, H * V_HD, "fp8")
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
            dense_up = _make_gated_up(B, S, DENSE_INTER, D, "fp8")
            g.add_kernel(dense_up)
            g.add_data_edge(ffn_norm, dense_up, {"y": "x"})
            L.dense_up = dense_up

            dense_down = _make_gemm(B, S, D, DENSE_INTER, "fp8")
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

        gate = _make_gemm(B, S, N_EXPERTS, D, "fp32", "bf16", "fp32")
        g.add_kernel(gate)
        g.add_data_edge(ffn_fan, gate, {"y": "x"})
        L.gate = gate

        # Dispatch: sigmoid routing + token scatter to experts
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

        # Expert kernels (up_proj + down_proj per expert)
        L.experts = []
        for eid in range(N_EXPERTS):
            up = StridedGemm(M_e, 2 * MOE_INTER, D, "fp8", "bf16",
                             out_elems=M_e * MOE_INTER)
            up.inputs = {"x": Tensor("bf16", (M_e, D))}
            up.weights = {"w": Tensor("fp8", (D, 2 * MOE_INTER))}
            scale_bytes = gemm_scale_bytes(2 * MOE_INTER, D, "fp8")
            if scale_bytes > 0:
                up.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            up.outputs = {"y": Tensor("bf16", (M_e, MOE_INTER))}
            g.add_kernel(up)

            down = Gemm(M_e, D, MOE_INTER, "fp8", "bf16")
            down.inputs = {"x": Tensor("bf16", (M_e, MOE_INTER))}
            down.weights = {"w": Tensor("fp8", (MOE_INTER, D))}
            scale_bytes = gemm_scale_bytes(D, MOE_INTER, "fp8")
            if scale_bytes > 0:
                down.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            down.outputs = {"y": Tensor("bf16", (M_e, D))}
            g.add_kernel(down)
            g.add_data_edge(up, down, {"y": "x"})
            g.add_data_edge(dispatch, up, {f"o{eid}": "x"})
            g.add_data_edge(down, combine, {"y": f"i{eid}"})

            L.experts.extend([up, down])

        # Shared expert (parallel with routed — reads from ffn_fan)
        sw_up = _make_gated_up(B, S, MOE_INTER, D, "fp8", "bf16")
        g.add_kernel(sw_up)
        g.add_data_edge(ffn_fan, sw_up, {"y3": "x"})
        L.sw_up = sw_up

        sw_down = _make_gemm(B, S, D, MOE_INTER, "fp8", "bf16")
        g.add_kernel(sw_down)
        g.add_data_edge(sw_up, sw_down, {"y": "x"})
        L.sw_down = sw_down

        # ── MoE output: routed + shared expert ────────────────────
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
    """Build persistent external KV-cache reads for decode attention."""
    kv_cache_reads = []
    for layer in layers:
        kv_read = ReadInput(B * context_len * KV_CACHE_DIM, "fp8")
        kv_read.inputs = {
            "kv": Tensor("fp8", (B, context_len, KV_CACHE_DIM))}
        kv_read.outputs = {
            "y": Tensor("fp8", (B, context_len, KV_CACHE_DIM))}
        g.add_kernel(kv_read)

        cache_fan = Spawn(world=2)
        cache_fan.inputs = {
            "x": Tensor("fp8", (B, context_len, KV_CACHE_DIM))}
        cache_fan.outputs = {
            "y": Tensor("fp8", (B, context_len, KV_CACHE_DIM)),
            "y2": Tensor("fp8", (B, context_len, KV_CACHE_DIM)),
        }
        g.add_kernel(cache_fan)
        g.add_data_edge(kv_read, cache_fan, {"y": "x"})
        g.add_data_edge(cache_fan, layer.sa, {"y": "kv"})

        layer.kv_cache_fan = cache_fan
        layer.kv_persist_fan = cache_fan
        kv_cache_reads.append(kv_read)

        if layer.has_full_indexer:
            index_read = ReadInput(B * context_len * INDEX_HD, "fp8")
            index_read.inputs = {
                "index_kv": Tensor("fp8", (B, context_len, INDEX_HD))}
            index_read.outputs = {
                "y": Tensor("fp8", (B, context_len, INDEX_HD))}
            g.add_kernel(index_read)

            index_fan = Spawn(world=2)
            index_fan.inputs = {
                "x": Tensor("fp8", (B, context_len, INDEX_HD))}
            index_fan.outputs = {
                "y": Tensor("fp8", (B, context_len, INDEX_HD)),
                "y2": Tensor("fp8", (B, context_len, INDEX_HD)),
            }
            g.add_kernel(index_fan)
            g.add_data_edge(index_read, index_fan, {"y": "x"})
            g.add_data_edge(index_fan, layer.sa, {"y": "index_kv"})
            layer.index_cache_read = index_read
            layer.index_cache_fan = index_fan
    return kv_cache_reads


def _build_kv_persistence_barrier(
    g, layers, output_src, output_name,
):
    """Keep every compact KV cache alive until the stage output is ready."""
    barrier = Nop()
    barrier.inputs = {}
    for layer_id, layer in enumerate(layers):
        main_cache = layer.kv_persist_fan.outputs["y2"]
        barrier.inputs[f"kv{layer_id}"] = Tensor(
            main_cache.dtype, main_cache.shape)
        if layer.index_cache_fan is not None:
            index_cache = layer.index_cache_fan.outputs["y2"]
            barrier.inputs[f"kv{layer_id}_index"] = Tensor(
                index_cache.dtype, index_cache.shape)
    barrier.inputs[output_name] = Tensor(
        output_src.outputs["y"].dtype, output_src.outputs["y"].shape)
    barrier.outputs = {"done": Tensor("int32", (1,))}
    g.add_kernel(barrier)

    for layer_id, layer in enumerate(layers):
        g.add_data_edge(
            layer.kv_persist_fan, barrier, {"y2": f"kv{layer_id}"})
        if layer.index_cache_fan is not None:
            g.add_data_edge(
                layer.index_cache_fan, barrier,
                {"y2": f"kv{layer_id}_index"})
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
        batch_size: number of independent sequences.
        seq_prefill: prefill sequence length, or the persistent prefix length
            whose full KV caches are consumed by decode.
        decode: whether to build the single-step decode stage.

    Returns:
        (g, layers, emb, read_input, kv_cache_reads, output_head)
        - layers: list[LayerMeta] for the selected stage
        - emb/read_input: token input kernels shared by both stages
        - kv_cache_reads: per-layer main KV ReadInput kernels (decode only)
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
