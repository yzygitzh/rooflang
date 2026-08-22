"""Kimi-K3 text-model inference — Declaration Phase.

Builds the logical compute graph (add_kernel + add_data_edge).
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import (
    AttnRes, ElementwiseOp, Embedding, Gemm, KimiK3DeltaAttn,
    KimiK3DeltaAttnStateStore, KimiK3MlaAttn, Nop, PartialRMSNorm,
    ReadInput, RMSNorm, Sampling, Slice,
    StridedGemm, TokenCombine, TokenDispatch,
)
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor
from rooflang.language.utils import gemm_scale_bytes

from rooflang.programs.models.kimi_k3.config import (
    ATTN_RES_BLOCK, BATCH, D, DENSE_INTER, DENSE_LAYERS,
    FULL_ATTN_LAYERS, H, KDA_CHUNK, KDA_CONV, KDA_HD,
    KV_CACHE_DIM, KV_LORA, MOE_INTER, N_EXPERTS,
    N_LAYERS, QK_HD, QK_NOPE_HD, Q_LORA, ROUTED_D,
    ROUTER_SCORING_FUNC, S_PREFILL, SHARED_INTER, TOPK, V, V_HD,
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


def _make_gated_up(B, S, N, K, w_dtype, a_dtype="bf16",
                   out_dtype="bf16"):
    """SiTU fused gate+up: projection FLOPs with one N-wide output."""
    M = B * S
    k = StridedGemm(M, 2 * N, K, w_dtype, a_dtype, out_dtype,
                    out_elems=M * N)
    k.inputs = {"x": Tensor(a_dtype, (B, S, K))}
    k.weights = {"w": Tensor(w_dtype, (K, 2 * N))}
    scale_bytes = gemm_scale_bytes(2 * N, K, w_dtype)
    if scale_bytes > 0:
        k.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
    k.outputs = {"y": Tensor(out_dtype, (B, S, N))}
    return k


def _make_attn_res(B, S, residual_count):
    k = AttnRes(B, S, D, residual_count, "bf16", "fp32")
    k.inputs = {
        "prefix": Tensor("bf16", (B, S, D)),
        "residual": Tensor("bf16", (B, S, residual_count, D)),
    }
    k.weights = {
        "norm": Tensor("bf16", (D,)),
        "proj": Tensor("bf16", (D,)),
    }
    k.outputs = {"y": Tensor("bf16", (B, S, D))}
    return k


# ── Per-layer metadata for optimization phase ───────────────────────────

@dataclass
class LayerMeta:
    bridge: Kernel = None
    block_in_fan: Kernel = None
    block_out_fan: Kernel = None
    pre_attn_res: Kernel = None
    block_append: Kernel = None
    attn_norm: Kernel = None
    attn_fan: Kernel = None
    wq_a: Kernel = None
    q_norm: Kernel = None
    wq_b: Kernel = None
    wkv: Kernel = None
    kv_norm: Kernel = None
    kda_wq: Kernel = None
    kda_wk: Kernel = None
    kda_wv: Kernel = None
    kda_f_a: Kernel = None
    kda_f_b: Kernel = None
    kda_beta: Kernel = None
    output_gate: Kernel = None
    kv_cache_quant: Kernel = None
    sa: Kernel = None
    sa_fan: Kernel = None
    attn_out_norm: Kernel = None
    attn_gate: Kernel = None
    state_store: Kernel = None
    wo: Kernel = None
    attn_add: Kernel = None
    ffn_bridge: Kernel = None
    mlp_res: Kernel = None
    ffn_norm: Kernel = None
    ffn_fan: Kernel = None
    dense_up: Kernel = None
    dense_down: Kernel = None
    gate: Kernel = None
    routed_down: Kernel = None
    dispatch: Kernel = None
    combine: Kernel = None
    routed_norm: Kernel = None
    routed_up: Kernel = None
    sw_up: Kernel = None
    sw_down: Kernel = None
    moe_add: Kernel = None
    ffn_add: Kernel = None
    experts: List[Kernel] = field(default_factory=list)
    # Cache and persistence metadata
    kv_cache_fan: Kernel = None
    conv_state_fan: Kernel = None
    kv_persist_fan: Kernel = None
    kv_persist_barrier: Kernel = None
    kv_sink: Kernel = None
    conv_state_read: Kernel = None

    @property
    def is_kda(self):
        return isinstance(self.sa, KimiK3DeltaAttn)

    @property
    def is_dense(self):
        return self.dense_up is not None


def _tag_weights(kernel, layer_id, name):
    """Tag kernel's weight Tensors with a weight_id for simulator dedup."""
    for port, t in kernel.weights.items():
        t.weight_id = f"L{layer_id}_{name}_{port}"


_WEIGHTED_LAYER_FIELDS = (
    "pre_attn_res", "attn_norm", "wq_a", "q_norm", "wq_b", "wkv",
    "kv_norm", "kda_wq", "kda_wk", "kda_wv", "kda_f_a", "kda_f_b",
    "kda_beta", "output_gate", "sa", "attn_out_norm", "wo",
    "mlp_res", "ffn_norm", "dense_up", "dense_down", "gate",
    "routed_down", "routed_norm", "routed_up", "sw_up", "sw_down",
)


def _tag_layer_weights(layer, layer_id):
    """Tag all shared weights belonging to one transformer layer."""
    for name in _WEIGHTED_LAYER_FIELDS:
        kernel = getattr(layer, name)
        if kernel is not None:
            _tag_weights(kernel, layer_id, name)
    for eid in range(len(layer.experts) // 2):
        _tag_weights(
            layer.experts[eid * 2], layer_id,
            f"expert{eid}_up")
        _tag_weights(
            layer.experts[eid * 2 + 1], layer_id,
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


def _build_mla(g, layer, B, S, cache_len, attn_fan, is_prefill):
    """Build one Kimi dense-MLA token-mixing path."""
    wq_a = _make_gemm(B, S, Q_LORA, D, "fp8")
    g.add_kernel(wq_a)
    g.add_data_edge(attn_fan, wq_a, {"y": "x"})
    layer.wq_a = wq_a

    q_norm = _make_norm(B, S, Q_LORA)
    g.add_kernel(q_norm)
    g.add_data_edge(wq_a, q_norm, {"y": "x"})
    layer.q_norm = q_norm

    wq_b = _make_gemm(B, S, H * QK_HD, Q_LORA, "fp8")
    g.add_kernel(wq_b)
    g.add_data_edge(q_norm, wq_b, {"y": "x"})
    layer.wq_b = wq_b

    wkv = _make_gemm(B, S, KV_CACHE_DIM, D, "fp8")
    g.add_kernel(wkv)
    g.add_data_edge(attn_fan, wkv, {"y2": "x"})
    layer.wkv = wkv

    kv_norm = PartialRMSNorm(B * S, KV_CACHE_DIM, KV_LORA, "bf16")
    kv_norm.inputs = {"x": Tensor("bf16", (B, S, KV_CACHE_DIM))}
    kv_norm.weights = {"g": Tensor("bf16", (KV_LORA,))}
    kv_norm.outputs = {"y": Tensor("bf16", (B, S, KV_CACHE_DIM))}
    g.add_kernel(kv_norm)
    g.add_data_edge(wkv, kv_norm, {"y": "x"})
    layer.kv_norm = kv_norm

    sa = KimiK3MlaAttn(
        B, H, S, cache_len, QK_HD, V_HD, KV_CACHE_DIM, KV_LORA,
        QK_NOPE_HD, dtype="fp8", kv_transform_dtype="fp8",
        q_dtype="bf16", kv_dtype="fp8", out_dtype="bf16",
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
        sa.weights["kv_b_scale"] = Tensor(
            "ue8m0", (int(scale_bytes),))
    sa.outputs = {"y": Tensor("bf16", (B, S, H * V_HD))}
    g.add_kernel(sa)
    g.add_data_edge(wq_b, sa, {"y": "q"})
    layer.sa = sa

    if is_prefill:
        kv_cache_quant = Slice()
        kv_cache_quant.inputs = {
            "x": Tensor("bf16", (B, S, KV_CACHE_DIM))}
        kv_cache_quant.outputs = {
            "y": Tensor("fp8", (B, S, KV_CACHE_DIM))}
        g.add_kernel(kv_cache_quant)
        g.add_data_edge(kv_norm, kv_cache_quant, {"y": "x"})
        layer.kv_cache_quant = kv_cache_quant

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
        layer.kv_cache_fan = kv_cache_fan
        layer.kv_persist_fan = kv_cache_fan
    else:
        kv_sink = Nop()
        kv_sink.inputs = {"kv": Tensor("bf16", (B, S, KV_CACHE_DIM))}
        g.add_kernel(kv_sink)
        g.add_data_edge(kv_norm, kv_sink, {"y": "kv"})
        layer.kv_sink = kv_sink

    return sa


def _build_kda(g, layer, B, S, attn_fan, is_prefill):
    """Build one KDA token-mixing path."""
    projection = H * KDA_HD
    paths = []
    for field, port in (("kda_wq", "y"), ("kda_wk", "y2"),
                        ("kda_wv", "y3")):
        kernel = _make_gemm(B, S, projection, D, "fp8")
        g.add_kernel(kernel)
        g.add_data_edge(attn_fan, kernel, {port: "x"})
        setattr(layer, field, kernel)
        paths.append(kernel)
    wq, wk, wv = paths

    f_a = _make_gemm(B, S, KDA_HD, D, "fp8")
    g.add_kernel(f_a)
    g.add_data_edge(attn_fan, f_a, {"y4": "x"})
    layer.kda_f_a = f_a

    f_b = _make_gemm(B, S, projection, KDA_HD, "fp8")
    g.add_kernel(f_b)
    g.add_data_edge(f_a, f_b, {"y": "x"})
    layer.kda_f_b = f_b

    beta = _make_gemm(B, S, H, D, "bf16", "bf16", "fp32")
    g.add_kernel(beta)
    g.add_data_edge(attn_fan, beta, {"y5": "x"})
    layer.kda_beta = beta

    sa = KimiK3DeltaAttn(
        B, H, S, KDA_HD, KDA_HD,
        "chunk" if is_prefill else "recurrent",
        KDA_CHUNK, KDA_CONV, "bf16", "bf16",
    )
    sa.inputs = {
        "q": Tensor("bf16", (B, S, H * KDA_HD)),
        "k": Tensor("bf16", (B, S, H * KDA_HD)),
        "v": Tensor("bf16", (B, S, H * KDA_HD)),
        "g": Tensor("bf16", (B, S, H * KDA_HD)),
        "beta": Tensor("fp32", (B, S, H)),
    }
    if not is_prefill:
        sa.inputs.update({
            "state": Tensor(
                "bf16", (B, H, KDA_HD, KDA_HD)),
            "conv_state": Tensor(
                "bf16", (B, H, 3, KDA_HD, KDA_CONV)),
        })
    sa.weights = {
        "conv_w": Tensor("bf16", (3, H, KDA_HD, KDA_CONV)),
        "conv_b": Tensor("bf16", (3, H, KDA_HD)),
        "A_log": Tensor("fp32", (H,)),
        "dt_bias": Tensor("fp32", (H, KDA_HD)),
    }
    sa.outputs = {"y": Tensor("bf16", (B, S, H * KDA_HD))}
    g.add_kernel(sa)
    for source, port in ((wq, "q"), (wk, "k"), (wv, "v"),
                         (f_b, "g"), (beta, "beta")):
        g.add_data_edge(source, sa, {"y": port})
    layer.sa = sa

    sa_fan = Spawn(world=2)
    sa_fan.inputs = {"x": Tensor("bf16", (B, S, projection))}
    sa_fan.outputs = {
        "y": Tensor("bf16", (B, S, projection)),
        "y2": Tensor("bf16", (B, S, projection)),
    }
    g.add_kernel(sa_fan)
    g.add_data_edge(sa, sa_fan, {"y": "x"})
    layer.sa_fan = sa_fan

    state_store = KimiK3DeltaAttnStateStore(
        B, H, S, KDA_HD, KDA_HD, KDA_CONV, "bf16")
    state_store.inputs = {"x": Tensor("bf16", (B, S, projection))}
    state_store.outputs = {
        "state": Tensor(
            "bf16", (B, H, KDA_HD, KDA_HD)),
        "conv_state": Tensor(
            "bf16", (B, H, 3, KDA_HD, KDA_CONV)),
    }
    g.add_kernel(state_store)
    g.add_data_edge(sa_fan, state_store, {"y2": "x"})
    layer.state_store = state_store
    return sa


def _build_layers(g, B, S, context_len, prev_out):
    """Build N_LAYERS Kimi-K3 text transformer layers into graph g."""
    M = B * S
    layers = []
    is_prefill = context_len is None
    block_residual = None
    residual_count = 0

    for layer_id in range(N_LAYERS):
        is_kda = layer_id not in FULL_ATTN_LAYERS
        cache_len = S if is_prefill else context_len
        layer = LayerMeta()

        # ── Input fan-out and pre-attention AttnRes ──────────────
        bridge = Spawn(world=2)
        bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
        bridge.outputs = {
            "y": Tensor("bf16", (B, S, D)),
            "y2": Tensor("bf16", (B, S, D)),
        }
        g.add_kernel(bridge)
        g.add_data_edge(prev_out, bridge, {"y": "x"})
        layer.bridge = bridge

        attn_input = bridge
        attn_input_port = "y"
        if block_residual is not None:
            block_in_fan = Spawn(
                world=2 if layer_id % ATTN_RES_BLOCK == 0 else 3)
            block_in_fan.inputs = {
                "x": Tensor("bf16", (B, S, residual_count, D))}
            block_in_fan.outputs = {
                "y": Tensor("bf16", (B, S, residual_count, D)),
                "y2": Tensor("bf16", (B, S, residual_count, D)),
            }
            if layer_id % ATTN_RES_BLOCK != 0:
                block_in_fan.outputs["y3"] = Tensor(
                    "bf16", (B, S, residual_count, D))
            g.add_kernel(block_in_fan)
            g.add_data_edge(
                block_residual[0], block_in_fan,
                {block_residual[1]: "x"})
            layer.block_in_fan = block_in_fan

            pre_attn_res = _make_attn_res(B, S, residual_count)
            g.add_kernel(pre_attn_res)
            g.add_data_edge(bridge, pre_attn_res, {"y": "prefix"})
            g.add_data_edge(
                block_in_fan, pre_attn_res, {"y": "residual"})
            layer.pre_attn_res = pre_attn_res
            attn_input = pre_attn_res

        if layer_id % ATTN_RES_BLOCK == 0:
            block_append = Concat()
            block_append.inputs = {
                "prefix": Tensor("bf16", (B, S, D))}
            if block_residual is not None:
                block_append.inputs["residual"] = Tensor(
                    "bf16", (B, S, residual_count, D))
            block_append.outputs = {
                "y": Tensor("bf16", (B, S, residual_count + 1, D))}
            g.add_kernel(block_append)
            g.add_data_edge(bridge, block_append, {"y2": "prefix"})
            if block_residual is not None:
                g.add_data_edge(
                    block_in_fan, block_append, {"y2": "residual"})
            residual_count += 1
            layer.block_append = block_append

            block_out_fan = Spawn(world=2)
            block_out_fan.inputs = {
                "x": Tensor("bf16", (B, S, residual_count, D))}
            block_out_fan.outputs = {
                "y": Tensor("bf16", (B, S, residual_count, D)),
                "y2": Tensor("bf16", (B, S, residual_count, D)),
            }
            g.add_kernel(block_out_fan)
            g.add_data_edge(block_append, block_out_fan, {"y": "x"})
            layer.block_out_fan = block_out_fan
            mlp_block_residual = (block_out_fan, "y")
            block_residual = (block_out_fan, "y2")
        else:
            mlp_block_residual = (block_in_fan, "y2")
            block_residual = (block_in_fan, "y3")

        # ── Attention / KDA ──────────────────────────────────────
        attn_norm = _make_norm(B, S, D)
        g.add_kernel(attn_norm)
        g.add_data_edge(attn_input, attn_norm, {attn_input_port: "x"})
        layer.attn_norm = attn_norm

        attn_world = 6 if is_kda else 3
        attn_fan = Spawn(world=attn_world)
        attn_fan.inputs = {"x": Tensor("bf16", (B, S, D))}
        attn_fan.outputs = {
            "y" if rank == 0 else f"y{rank + 1}":
            Tensor("bf16", (B, S, D))
            for rank in range(attn_world)
        }
        g.add_kernel(attn_fan)
        g.add_data_edge(attn_norm, attn_fan, {"y": "x"})
        layer.attn_fan = attn_fan

        if is_kda:
            sa = _build_kda(g, layer, B, S, attn_fan, is_prefill)
            output_gate_port = "y6"

            attn_out_norm = _make_norm(B, S, H * KDA_HD)
            g.add_kernel(attn_out_norm)
            g.add_data_edge(layer.sa_fan, attn_out_norm, {"y": "x"})
            layer.attn_out_norm = attn_out_norm
            attn_output = attn_out_norm
        else:
            sa = _build_mla(
                g, layer, B, S, cache_len, attn_fan, is_prefill)
            output_gate_port = "y3"
            attn_output = sa

        output_gate = _make_gemm(B, S, H * V_HD, D, "fp8")
        g.add_kernel(output_gate)
        g.add_data_edge(attn_fan, output_gate, {output_gate_port: "x"})
        layer.output_gate = output_gate

        attn_gate = ElementwiseOp(
            M, H * V_HD, "bf16", op="sigmoid_mul")
        attn_gate.inputs = {
            "a": Tensor("bf16", (B, S, H * V_HD)),
            "b": Tensor("bf16", (B, S, H * V_HD)),
        }
        attn_gate.outputs = {"y": Tensor("bf16", (B, S, H * V_HD))}
        g.add_kernel(attn_gate)
        g.add_data_edge(attn_output, attn_gate, {"y": "a"})
        g.add_data_edge(output_gate, attn_gate, {"y": "b"})
        layer.attn_gate = attn_gate

        wo = _make_gemm(B, S, D, H * V_HD, "fp8")
        g.add_kernel(wo)
        g.add_data_edge(attn_gate, wo, {"y": "x"})
        layer.wo = wo

        if layer_id % ATTN_RES_BLOCK == 0:
            prefix_after_attn = wo
        else:
            attn_add = ElementwiseOp(M, D, "bf16")
            attn_add.inputs = {
                "a": Tensor("bf16", (B, S, D)),
                "b": Tensor("bf16", (B, S, D)),
            }
            attn_add.outputs = {"y": Tensor("bf16", (B, S, D))}
            g.add_kernel(attn_add)
            g.add_data_edge(bridge, attn_add, {"y2": "a"})
            g.add_data_edge(wo, attn_add, {"y": "b"})
            layer.attn_add = attn_add
            prefix_after_attn = attn_add

        # ── MLP AttnRes and normalization ────────────────────────
        ffn_bridge = Spawn(world=2)
        ffn_bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
        ffn_bridge.outputs = {
            "y": Tensor("bf16", (B, S, D)),
            "y2": Tensor("bf16", (B, S, D)),
        }
        g.add_kernel(ffn_bridge)
        g.add_data_edge(prefix_after_attn, ffn_bridge, {"y": "x"})
        layer.ffn_bridge = ffn_bridge

        mlp_res = _make_attn_res(B, S, residual_count)
        g.add_kernel(mlp_res)
        g.add_data_edge(ffn_bridge, mlp_res, {"y": "prefix"})
        g.add_data_edge(
            mlp_block_residual[0], mlp_res,
            {mlp_block_residual[1]: "residual"})
        layer.mlp_res = mlp_res

        ffn_norm = _make_norm(B, S, D)
        g.add_kernel(ffn_norm)
        g.add_data_edge(mlp_res, ffn_norm, {"y": "x"})
        layer.ffn_norm = ffn_norm

        if layer_id < DENSE_LAYERS:
            dense_up = _make_gated_up(B, S, DENSE_INTER, D, "fp8")
            g.add_kernel(dense_up)
            g.add_data_edge(ffn_norm, dense_up, {"y": "x"})
            layer.dense_up = dense_up

            dense_down = _make_gemm(B, S, D, DENSE_INTER, "fp8")
            g.add_kernel(dense_down)
            g.add_data_edge(dense_up, dense_down, {"y": "x"})
            layer.dense_down = dense_down
            mlp_output = dense_down
        else:
            # Gate, routed-latent path, and shared experts.
            ffn_fan = Spawn(world=3)
            ffn_fan.inputs = {"x": Tensor("bf16", (B, S, D))}
            ffn_fan.outputs = {
                "y": Tensor("bf16", (B, S, D)),
                "y2": Tensor("bf16", (B, S, D)),
                "y3": Tensor("bf16", (B, S, D)),
            }
            g.add_kernel(ffn_fan)
            g.add_data_edge(ffn_norm, ffn_fan, {"y": "x"})
            layer.ffn_fan = ffn_fan

            # Optimized serving uses a BF16 gate GEMM with FP32 accumulation
            # and logits; scoring, correction, and Top-K remain in FP32.
            gate = Gemm(M, N_EXPERTS, D, "bf16", "bf16", "fp32")
            gate.inputs = {"x": Tensor("bf16", (B, S, D))}
            gate.weights = {"w": Tensor("bf16", (D, N_EXPERTS))}
            gate.outputs = {"y": Tensor("fp32", (B, S, N_EXPERTS))}
            g.add_kernel(gate)
            g.add_data_edge(ffn_fan, gate, {"y": "x"})
            layer.gate = gate

            # Latent-MoE adapters are BF16 in the official checkpoint; only
            # the routed experts themselves use packed MXFP4 weights.
            routed_down = _make_gemm(B, S, ROUTED_D, D, "bf16")
            g.add_kernel(routed_down)
            g.add_data_edge(ffn_fan, routed_down, {"y2": "x"})
            layer.routed_down = routed_down

            M_e = Fraction(M * TOPK, N_EXPERTS)
            dispatch = TokenDispatch(
                M, ROUTED_D, N_EXPERTS, TOPK,
                scoring_func=ROUTER_SCORING_FUNC)
            dispatch.inputs = {
                "x": Tensor("bf16", (B, S, ROUTED_D)),
                "routing": Tensor("fp32", (B, S, N_EXPERTS)),
            }
            dispatch.outputs = {
                f"o{eid}": Tensor("bf16", (M_e, ROUTED_D))
                for eid in range(N_EXPERTS)
            }
            g.add_kernel(dispatch)
            g.add_data_edge(gate, dispatch, {"y": "routing"})
            g.add_data_edge(routed_down, dispatch, {"y": "x"})
            layer.dispatch = dispatch

            combine = TokenCombine(M, ROUTED_D, N_EXPERTS, TOPK)
            combine.inputs = {
                f"i{eid}": Tensor("bf16", (M_e, ROUTED_D))
                for eid in range(N_EXPERTS)
            }
            combine.outputs = {"y": Tensor("bf16", (B, S, ROUTED_D))}
            g.add_kernel(combine)
            layer.combine = combine

            for eid in range(N_EXPERTS):
                up = StridedGemm(
                    M_e, 2 * MOE_INTER, ROUTED_D, "fp4", "bf16",
                    out_elems=M_e * MOE_INTER)
                up.inputs = {"x": Tensor("bf16", (M_e, ROUTED_D))}
                up.weights = {
                    "w": Tensor("fp4", (ROUTED_D, 2 * MOE_INTER))}
                scale_bytes = gemm_scale_bytes(
                    2 * MOE_INTER, ROUTED_D, "fp4")
                if scale_bytes > 0:
                    up.weights["s"] = Tensor(
                        "ue8m0", (int(scale_bytes),))
                up.outputs = {"y": Tensor("bf16", (M_e, MOE_INTER))}
                g.add_kernel(up)

                down = Gemm(M_e, ROUTED_D, MOE_INTER, "fp4", "bf16")
                down.inputs = {"x": Tensor("bf16", (M_e, MOE_INTER))}
                down.weights = {
                    "w": Tensor("fp4", (MOE_INTER, ROUTED_D))}
                scale_bytes = gemm_scale_bytes(ROUTED_D, MOE_INTER, "fp4")
                if scale_bytes > 0:
                    down.weights["s"] = Tensor(
                        "ue8m0", (int(scale_bytes),))
                down.outputs = {"y": Tensor("bf16", (M_e, ROUTED_D))}
                g.add_kernel(down)
                g.add_data_edge(up, down, {"y": "x"})
                g.add_data_edge(
                    dispatch, up, {f"o{eid}": "x"})
                g.add_data_edge(
                    down, combine, {"y": f"i{eid}"})
                layer.experts.extend([up, down])

            routed_norm = _make_norm(B, S, ROUTED_D)
            g.add_kernel(routed_norm)
            g.add_data_edge(combine, routed_norm, {"y": "x"})
            layer.routed_norm = routed_norm

            routed_up = _make_gemm(B, S, D, ROUTED_D, "bf16")
            g.add_kernel(routed_up)
            g.add_data_edge(routed_norm, routed_up, {"y": "x"})
            layer.routed_up = routed_up

            sw_up = _make_gated_up(B, S, SHARED_INTER, D, "fp8")
            g.add_kernel(sw_up)
            g.add_data_edge(ffn_fan, sw_up, {"y3": "x"})
            layer.sw_up = sw_up

            sw_down = _make_gemm(B, S, D, SHARED_INTER, "fp8")
            g.add_kernel(sw_down)
            g.add_data_edge(sw_up, sw_down, {"y": "x"})
            layer.sw_down = sw_down

            moe_add = ElementwiseOp(M, D, "bf16")
            moe_add.inputs = {
                "a": Tensor("bf16", (B, S, D)),
                "b": Tensor("bf16", (B, S, D)),
            }
            moe_add.outputs = {"y": Tensor("bf16", (B, S, D))}
            g.add_kernel(moe_add)
            g.add_data_edge(routed_up, moe_add, {"y": "a"})
            g.add_data_edge(sw_down, moe_add, {"y": "b"})
            layer.moe_add = moe_add
            mlp_output = moe_add

        ffn_add = ElementwiseOp(M, D, "bf16")
        ffn_add.inputs = {
            "a": Tensor("bf16", (B, S, D)),
            "b": Tensor("bf16", (B, S, D)),
        }
        ffn_add.outputs = {"y": Tensor("bf16", (B, S, D))}
        g.add_kernel(ffn_add)
        g.add_data_edge(ffn_bridge, ffn_add, {"y2": "a"})
        g.add_data_edge(mlp_output, ffn_add, {"y": "b"})
        layer.ffn_add = ffn_add

        _tag_layer_weights(layer, layer_id)
        prev_out = ffn_add
        layers.append(layer)

    return layers, prev_out, block_residual, residual_count


def _build_decode_cache_read(g, B, context_len, layers):
    """Build persistent external MLA/KDA cache reads for decode."""
    cache_reads = []
    for layer in layers:
        if not layer.is_kda:
            kv_read = ReadInput(B * context_len * KV_CACHE_DIM,
                                "fp8")
            kv_read.inputs = {
                "kv": Tensor(
                    "fp8", (B, context_len, KV_CACHE_DIM))}
            kv_read.outputs = {
                "y": Tensor(
                    "fp8", (B, context_len, KV_CACHE_DIM))}
            g.add_kernel(kv_read)

            cache_fan = Spawn(world=2)
            cache_fan.inputs = {
                "x": Tensor(
                    "fp8", (B, context_len, KV_CACHE_DIM))}
            cache_fan.outputs = {
                "y": Tensor(
                    "fp8", (B, context_len, KV_CACHE_DIM)),
                "y2": Tensor(
                    "fp8", (B, context_len, KV_CACHE_DIM)),
            }
            g.add_kernel(cache_fan)
            g.add_data_edge(kv_read, cache_fan, {"y": "x"})
            g.add_data_edge(cache_fan, layer.sa, {"y": "kv"})
            layer.kv_cache_fan = cache_fan
            layer.kv_persist_fan = cache_fan
            cache_reads.append(kv_read)
            continue

        state_read = ReadInput(B * H * KDA_HD * KDA_HD, "bf16")
        state_read.inputs = {
            "state": Tensor(
                "bf16", (B, H, KDA_HD, KDA_HD))}
        state_read.outputs = {
            "y": Tensor("bf16", (B, H, KDA_HD, KDA_HD))}
        g.add_kernel(state_read)

        state_fan = Spawn(world=1)
        state_fan.inputs = {
            "x": Tensor("bf16", (B, H, KDA_HD, KDA_HD))}
        state_fan.outputs = {
            "y": Tensor("bf16", (B, H, KDA_HD, KDA_HD))}
        g.add_kernel(state_fan)
        g.add_data_edge(state_read, state_fan, {"y": "x"})
        g.add_data_edge(state_fan, layer.sa, {"y": "state"})

        conv_elements = B * H * 3 * KDA_HD * KDA_CONV
        conv_read = ReadInput(conv_elements, "bf16")
        conv_read.inputs = {
            "conv_state": Tensor(
                "bf16", (B, H, 3, KDA_HD, KDA_CONV))}
        conv_read.outputs = {
            "y": Tensor(
                "bf16", (B, H, 3, KDA_HD, KDA_CONV))}
        g.add_kernel(conv_read)

        conv_fan = Spawn(world=1)
        conv_fan.inputs = {
            "x": Tensor(
                "bf16", (B, H, 3, KDA_HD, KDA_CONV))}
        conv_fan.outputs = {
            "y": Tensor(
                "bf16", (B, H, 3, KDA_HD, KDA_CONV))}
        g.add_kernel(conv_fan)
        g.add_data_edge(conv_read, conv_fan, {"y": "x"})
        g.add_data_edge(conv_fan, layer.sa, {"y": "conv_state"})

        layer.kv_cache_fan = state_fan
        layer.conv_state_fan = conv_fan
        layer.conv_state_read = conv_read
        cache_reads.append(state_read)
    return cache_reads


def _build_cache_persistence_barrier(g, layers, output_src, output_name):
    """Keep every MLA cache or KDA state alive through stage completion."""
    barrier = Nop()
    barrier.inputs = {}
    for layer_id, layer in enumerate(layers):
        if layer.is_kda:
            state = layer.state_store.outputs["state"]
            conv_state = layer.state_store.outputs["conv_state"]
            barrier.inputs[f"kv{layer_id}"] = Tensor(
                state.dtype, state.shape)
            barrier.inputs[f"kv{layer_id}_conv"] = Tensor(
                conv_state.dtype, conv_state.shape)
        else:
            cache = layer.kv_persist_fan.outputs["y2"]
            barrier.inputs[f"kv{layer_id}"] = Tensor(
                cache.dtype, cache.shape)
    barrier.inputs[output_name] = Tensor(
        output_src.outputs["y"].dtype, output_src.outputs["y"].shape)
    barrier.outputs = {"done": Tensor("int32", (1,))}
    g.add_kernel(barrier)

    for layer_id, layer in enumerate(layers):
        if layer.is_kda:
            g.add_data_edge(
                layer.state_store, barrier,
                {"state": f"kv{layer_id}",
                 "conv_state": f"kv{layer_id}_conv"})
        else:
            g.add_data_edge(
                layer.kv_persist_fan, barrier, {"y2": f"kv{layer_id}"})
        layer.kv_persist_barrier = barrier
    g.add_data_edge(output_src, barrier, {"y": output_name})
    return barrier


def declare_model(batch_size=BATCH, seq_prefill=S_PREFILL, decode=False):
    """Build either a text-only prefill graph or one-step decode graph."""
    is_prefill = not decode
    g = ComputeGraph()
    B = batch_size
    S = seq_prefill if is_prefill else 1
    context_len = None if is_prefill else seq_prefill

    read_input, emb = _build_token_input(g, B, S)
    layers, last_output, block_residual, residual_count = _build_layers(
        g, B, S, context_len, emb)

    output_attn_res = _make_attn_res(B, S, residual_count)
    g.add_kernel(output_attn_res)
    g.add_data_edge(last_output, output_attn_res, {"y": "prefix"})
    g.add_data_edge(
        block_residual[0], output_attn_res,
        {block_residual[1]: "residual"})
    _tag_weights(output_attn_res, -1, "output_attn_res")

    cache_reads = []
    output_head = ()
    if is_prefill:
        last_token = Slice()
        last_token.inputs = {"x": Tensor("bf16", (B, S, D))}
        last_token.outputs = {"y": Tensor("bf16", (B, 1, D))}
        g.add_kernel(last_token)
        g.add_data_edge(output_attn_res, last_token, {"y": "x"})
        output_head = (
            output_attn_res, last_token,
            *_build_output_head(g, B, last_token),
        )
        _build_cache_persistence_barrier(
            g, layers, output_head[-1], "prefill_output")
    else:
        cache_reads = _build_decode_cache_read(g, B, context_len, layers)
        output_head = (
            output_attn_res, *_build_output_head(g, B, output_attn_res),
        )
        _build_cache_persistence_barrier(
            g, layers, output_head[-1], "decode_output")

    g.validate()
    return g, layers, emb, read_input, cache_reads, output_head
