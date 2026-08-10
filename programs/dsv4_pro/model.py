"""DeepSeek V4 Pro inference — Declaration Phase.

Builds the logical compute graph (add_kernel + add_data_edge).
"""

from dataclasses import dataclass, field
from typing import List

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import (
    ElementwiseOp, Embedding, Gemm, Nop, ReadInput, RMSNorm, Sampling,
    Slice, SparseAttn, StridedGemm, TokenCombine, TokenDispatch,
)
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor
from rooflang.language.utils import gemm_scale_bytes

from rooflang.programs.dsv4_pro.config import (
    BATCH, COMPRESS_RATIOS, D, H, HD, INDEX_TOPK, KV_DIM,
    MOE_INTER, N_EXPERTS, N_LAYERS, O_GROUPS, O_LORA,
    Q_LORA, S_PREFILL, TOPK, V, WINDOW,
)


# ── Kernel factories ────────────────────────────────────────────────────

def make_gemm(B, S, N, K, w_dtype, a_dtype="bf16", out_dtype="bf16"):
    M = B * S
    k = Gemm(M, N, K, w_dtype, a_dtype, out_dtype)
    k.inputs = {"x": Tensor(a_dtype, (B, S, K))}
    k.weights = {"w": Tensor(w_dtype, (K, N))}
    scale_bytes = gemm_scale_bytes(N, K, w_dtype)
    if scale_bytes > 0:
        k.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
    k.outputs = {"y": Tensor(out_dtype, (B, S, N))}
    return k


def make_norm(B, S, dim):
    M = B * S
    k = RMSNorm(M, dim, "bf16")
    k.inputs = {"x": Tensor("bf16", (B, S, dim))}
    k.weights = {"g": Tensor("bf16", (dim,))}
    k.outputs = {"y": Tensor("bf16", (B, S, dim))}
    return k


def make_gated_up(B, S, N, K, w_dtype, a_dtype="bf16", out_dtype="bf16"):
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
    wq_b: Kernel = None
    wkv: Kernel = None
    kv_norm: Kernel = None
    comp: Kernel = None
    comp_buf: Kernel = None
    comp_concat: Kernel = None
    comp_norm: Kernel = None
    kv_norm_fan: Kernel = None
    comp_norm_fan: Kernel = None
    kv_concat: Kernel = None
    sa: Kernel = None
    wo_a: Kernel = None
    wo_b: Kernel = None
    attn_add: Kernel = None
    ffn_bridge: Kernel = None
    ffn_norm: Kernel = None
    ffn_fan: Kernel = None
    gate: Kernel = None
    dispatch: Kernel = None
    combine: Kernel = None
    sw_up: Kernel = None
    sw_down: Kernel = None
    moe_add: Kernel = None
    ffn_add: Kernel = None
    experts: List[Kernel] = field(default_factory=list)
    # Decode KV chain (set by declare_model after _build_layers)
    kv_win_slice: Kernel = None
    kv_cache_fan: Kernel = None
    kv_comp_slice: Kernel = None
    kv_acc: Kernel = None
    kv_acc_fan: Kernel = None
    persist_kv_cache: bool = False
    kv_persist_fan: Kernel = None
    kv_persist_barrier: Kernel = None


@dataclass
class DecodeStepMeta:
    """Per-decode-step metadata (token input + embedding + layer list)."""
    read_input: Kernel = None
    emb: Kernel = None
    final_norm: Kernel = None
    logits: Kernel = None
    sampling: Kernel = None
    layers: List[LayerMeta] = field(default_factory=list)


def _tag_weights(kernel, layer_id, name):
    """Tag kernel's weight Tensors with a weight_id for simulator dedup."""
    for port, t in kernel.weights.items():
        t.weight_id = f"L{layer_id}_{name}_{port}"


_WEIGHTED_LAYER_FIELDS = (
    "attn_norm", "wq_a", "q_norm", "wq_b", "wkv", "kv_norm",
    "comp", "comp_norm", "wo_a", "wo_b", "ffn_norm", "gate",
    "sw_up", "sw_down",
)


def _tag_layer_weights(layer, layer_id):
    """Tag all shared weights belonging to one transformer layer."""
    for name in _WEIGHTED_LAYER_FIELDS:
        kernel = getattr(layer, name)
        if kernel is not None:
            _tag_weights(kernel, layer_id, name)

    for eid in range(N_EXPERTS):
        _tag_weights(layer.experts[eid * 2], layer_id, f"expert{eid}_up")
        _tag_weights(layer.experts[eid * 2 + 1], layer_id,
                     f"expert{eid}_down")


def _build_output_head(g, B, hidden_src):
    """Build the shared final norm, logits projection, and sampler."""
    final_norm = make_norm(B, 1, D)
    g.add_kernel(final_norm)
    g.add_data_edge(hidden_src, final_norm, {"y": "x"})
    _tag_weights(final_norm, -1, "final_norm")

    logits = make_gemm(B, 1, V, D, "bf16")
    g.add_kernel(logits)
    g.add_data_edge(final_norm, logits, {"y": "x"})
    _tag_weights(logits, -1, "logits")

    sampling = Sampling(B, V)
    sampling.inputs = {"logits": Tensor("bf16", (B, 1, V))}
    sampling.outputs = {"y": Tensor("int32", (B, 1, 1))}
    g.add_kernel(sampling)
    g.add_data_edge(logits, sampling, {"y": "logits"})

    return final_norm, logits, sampling


def _build_layers(
    g, B, S, context_len, prev_out, has_decode=False,
    capture_kv_cache=False,
):
    """Build N_LAYERS transformer layers into graph g.

    Args:
        g: ComputeGraph to add kernels to.
        B: batch size.
        S: sequence length.
        context_len: None for prefill mode (builds compressor + kv_concat).
            For decode mode: integer pfx_len (total tokens already processed).
            Per-layer S_kv = WINDOW + total_compressed.
        prev_out: kernel whose "y" output feeds into the first layer.
        has_decode: if True (prefill in full model), add Spawn fans after
            kv_norm/comp_norm so decode can read the initial KV cache.

    Returns (layers, last_output_kernel).
    """
    M = B * S
    layers = []

    for layer_id in range(N_LAYERS):
        ratio = COMPRESS_RATIOS[layer_id]
        L = LayerMeta()

        # ── Input fan-out (residual + attention + optional compressor) ──
        if context_len is None:
            # Prefill: y→attn_norm, y2→comp, y3→attn_residual
            bridge = Spawn(world=3)
            bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
            bridge.outputs = {"y": Tensor("bf16", (B, S, D)),
                              "y2": Tensor("bf16", (B, S, D)),
                              "y3": Tensor("bf16", (B, S, D))}
            g.add_kernel(bridge)
            g.add_data_edge(prev_out, bridge, {"y": "x"})
        else:
            # Decode: bridge structure depends on whether compressor fires
            decode_has_comp = (context_len + 1) % ratio == 0
            if decode_has_comp:
                # ratio=4: y→attn_norm, y2→comp, y3→attn_residual
                bridge = Spawn(world=3)
                bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
                bridge.outputs = {"y": Tensor("bf16", (B, S, D)),
                                  "y2": Tensor("bf16", (B, S, D)),
                                  "y3": Tensor("bf16", (B, S, D))}
            else:
                # ratio=128: y→attn_norm, y2→attn_residual
                bridge = Spawn(world=2)
                bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
                bridge.outputs = {"y": Tensor("bf16", (B, S, D)),
                                  "y2": Tensor("bf16", (B, S, D))}
            g.add_kernel(bridge)
            g.add_data_edge(prev_out, bridge, {"y": "x"})
        L.bridge = bridge

        # ── Attention ─────────────────────────────────────────────
        attn_norm = make_norm(B, S, D)
        g.add_kernel(attn_norm)
        g.add_data_edge(bridge, attn_norm, {"y": "x"})
        L.attn_norm = attn_norm

        # Fan-out after norm: Q path + KV path
        attn_fan = Spawn(world=2)
        attn_fan.inputs = {"x": Tensor("bf16", (B, S, D))}
        attn_fan.outputs = {"y": Tensor("bf16", (B, S, D)),
                            "y2": Tensor("bf16", (B, S, D))}
        g.add_kernel(attn_fan)
        g.add_data_edge(attn_norm, attn_fan, {"y": "x"})
        L.attn_fan = attn_fan

        # Q path
        wq_a = make_gemm(B, S, Q_LORA, D, "fp8")
        g.add_kernel(wq_a)
        g.add_data_edge(attn_fan, wq_a, {"y": "x"})
        L.wq_a = wq_a

        q_norm = make_norm(B, S, Q_LORA)
        g.add_kernel(q_norm)
        g.add_data_edge(wq_a, q_norm, {"y": "x"})
        L.q_norm = q_norm

        wq_b = make_gemm(B, S, H * HD, Q_LORA, "fp8")
        g.add_kernel(wq_b)
        g.add_data_edge(q_norm, wq_b, {"y": "x"})
        L.wq_b = wq_b

        # KV path (branch from attn fan-out)
        wkv = make_gemm(B, S, KV_DIM, D, "fp8")
        g.add_kernel(wkv)
        g.add_data_edge(attn_fan, wkv, {"y2": "x"})
        L.wkv = wkv

        kv_norm = make_norm(B, S, KV_DIM)
        g.add_kernel(kv_norm)
        g.add_data_edge(wkv, kv_norm, {"y": "x"})
        L.kv_norm = kv_norm

        if context_len is None:
            # Prefill: build compressor + kv_concat
            if ratio in (128, 4):
                coff = 2 if ratio == 4 else 1
                S_comp = S // ratio
                M_comp = B * S_comp
                comp_out_elems = M_comp * KV_DIM
                comp = StridedGemm(M, 2 * coff * KV_DIM, D, "fp32", "bf16",
                                   in_elems=M * D, out_elems=comp_out_elems)
                comp.inputs = {"x": Tensor("bf16", (B, S, D))}
                comp.weights = {"w": Tensor("fp32", (D, 2 * coff * KV_DIM))}
                comp.outputs = {"y": Tensor("bf16", (B, S_comp, KV_DIM))}
                g.add_kernel(comp)
                g.add_data_edge(bridge, comp, {"y2": "x"})
                L.comp = comp

                comp_norm = make_norm(B, S_comp, KV_DIM)
                g.add_kernel(comp_norm)
                g.add_data_edge(comp, comp_norm, {"y": "x"})
                L.comp_norm = comp_norm

            # Sparse attention (prefill: S_kv = WINDOW + S//ratio)
            k_sel = WINDOW
            if ratio == 128:
                k_sel = WINDOW + S // 128
            elif ratio == 4:
                k_sel = WINDOW + INDEX_TOPK

            S_comp = S // ratio
            S_kv = WINDOW + S_comp
            sa = SparseAttn(B, H, 1, S, k_sel, S_kv, HD, "bf16", kv_factor=1)
            sa.inputs = {"q": Tensor("bf16", (B, S, H * HD)),
                         "kv": Tensor("bf16", (B, S_kv, KV_DIM))}
            sa.outputs = {"y": Tensor("bf16", (B, S, H * HD))}
            g.add_kernel(sa)
            g.add_data_edge(wq_b, sa, {"y": "q"})

            # Window slice: extract last WINDOW tokens from full KV
            kv_win_slice = Slice()
            kv_win_slice.inputs = {
                "x": Tensor("bf16", (B, S, KV_DIM))}
            kv_win_slice.outputs = {
                "y": Tensor("bf16", (B, WINDOW, KV_DIM))}
            g.add_kernel(kv_win_slice)
            g.add_data_edge(kv_norm, kv_win_slice, {"y": "x"})

            # KV cache = concat(window_kv, compressed_kv)
            kv_concat = Concat()
            kv_concat.inputs = {"a": Tensor("bf16", (B, WINDOW, KV_DIM)),
                                "b": Tensor("bf16", (B, S_comp, KV_DIM))}
            kv_concat.outputs = {"y": Tensor("bf16", (B, S_kv, KV_DIM))}
            g.add_kernel(kv_concat)

            if has_decode:
                # Fan window: y→kv_concat (prefill SA), y2→decode
                kv_norm_fan = Spawn(world=2)
                kv_norm_fan.inputs = {
                    "x": Tensor("bf16", (B, WINDOW, KV_DIM))}
                kv_norm_fan.outputs = {
                    "y": Tensor("bf16", (B, WINDOW, KV_DIM)),
                    "y2": Tensor("bf16", (B, WINDOW, KV_DIM))}
                g.add_kernel(kv_norm_fan)
                g.add_data_edge(kv_win_slice, kv_norm_fan, {"y": "x"})
                g.add_data_edge(kv_norm_fan, kv_concat, {"y": "a"})
                L.kv_norm_fan = kv_norm_fan

                comp_norm_fan = Spawn(world=2)
                comp_norm_fan.inputs = {
                    "x": Tensor("bf16", (B, S_comp, KV_DIM))}
                comp_norm_fan.outputs = {
                    "y": Tensor("bf16", (B, S_comp, KV_DIM)),
                    "y2": Tensor("bf16", (B, S_comp, KV_DIM))}
                g.add_kernel(comp_norm_fan)
                g.add_data_edge(comp_norm, comp_norm_fan, {"y": "x"})
                g.add_data_edge(comp_norm_fan, kv_concat, {"y": "b"})
                L.comp_norm_fan = comp_norm_fan
            else:
                g.add_data_edge(kv_win_slice, kv_concat, {"y": "a"})
                g.add_data_edge(comp_norm, kv_concat, {"y": "b"})

            if capture_kv_cache:
                kv_persist_fan = Spawn(world=2)
                kv_persist_fan.inputs = {
                    "x": Tensor("bf16", (B, S_kv, KV_DIM))}
                kv_persist_fan.outputs = {
                    "y": Tensor("bf16", (B, S_kv, KV_DIM)),
                    "y2": Tensor("bf16", (B, S_kv, KV_DIM)),
                }
                g.add_kernel(kv_persist_fan)
                g.add_data_edge(kv_concat, kv_persist_fan, {"y": "x"})
                g.add_data_edge(kv_persist_fan, sa, {"y": "kv"})
                L.kv_persist_fan = kv_persist_fan
            else:
                g.add_data_edge(kv_concat, sa, {"y": "kv"})

            L.kv_win_slice = kv_win_slice
            L.kv_concat = kv_concat
        else:
            # Decode: window + compressed KV cache
            decode_has_comp = (context_len + 1) % ratio == 0
            total_compressed = (context_len + 1) // ratio
            S_kv = min(WINDOW, context_len + 1) + total_compressed

            k_sel = WINDOW
            if ratio == 128:
                k_sel = WINDOW + total_compressed
            elif ratio == 4:
                k_sel = WINDOW + INDEX_TOPK

            sa = SparseAttn(B, H, 1, S, k_sel, S_kv, HD, "bf16", kv_factor=1)
            sa.inputs = {"q": Tensor("bf16", (B, S, H * HD)),
                         "kv": Tensor("bf16", (B, S_kv, KV_DIM))}
            sa.outputs = {"y": Tensor("bf16", (B, S, H * HD))}
            g.add_kernel(sa)
            g.add_data_edge(wq_b, sa, {"y": "q"})

            # Compressor: fires when this step completes a compression group
            if decode_has_comp:
                coff = 2 if ratio == 4 else 1
                M_buf = B * ratio
                # Buffer: previous ratio-1 hidden states (already in HBM)
                comp_buf = ReadInput(B * (ratio - 1) * D, "bf16")
                comp_buf.inputs = {
                    "buf": Tensor("bf16", (B, ratio - 1, D))}
                comp_buf.outputs = {
                    "y": Tensor("bf16", (B, ratio - 1, D))}
                g.add_kernel(comp_buf)
                L.comp_buf = comp_buf

                # Concat buffer + current hidden state → (B, ratio, D)
                comp_concat = Concat()
                comp_concat.inputs = {
                    "a": Tensor("bf16", (B, ratio - 1, D)),
                    "b": Tensor("bf16", (B, 1, D))}
                comp_concat.outputs = {
                    "y": Tensor("bf16", (B, ratio, D))}
                g.add_kernel(comp_concat)
                g.add_data_edge(comp_buf, comp_concat, {"y": "a"})
                g.add_data_edge(bridge, comp_concat, {"y2": "b"})
                L.comp_concat = comp_concat

                # Compressor projection
                comp = StridedGemm(M_buf, 2 * coff * KV_DIM, D,
                                   "fp32", "bf16",
                                   in_elems=M_buf * D,
                                   out_elems=B * 1 * KV_DIM)
                comp.inputs = {"x": Tensor("bf16", (B, ratio, D))}
                comp.weights = {
                    "w": Tensor("fp32", (D, 2 * coff * KV_DIM))}
                comp.outputs = {"y": Tensor("bf16", (B, 1, KV_DIM))}
                g.add_kernel(comp)
                g.add_data_edge(comp_concat, comp, {"y": "x"})
                L.comp = comp

                comp_norm = make_norm(B, 1, KV_DIM)
                g.add_kernel(comp_norm)
                g.add_data_edge(comp, comp_norm, {"y": "x"})
                L.comp_norm = comp_norm

        L.sa = sa

        # Output projection (grouped linear: O_GROUPS independent Gemms)
        wo_a = StridedGemm(M, O_GROUPS * O_LORA, H * HD // O_GROUPS,
                           "bf16", "bf16", in_elems=M * H * HD)
        wo_a.inputs = {"x": Tensor("bf16", (B, S, H * HD))}
        wo_a.weights = {"w": Tensor("bf16",
                        (H * HD // O_GROUPS, O_GROUPS * O_LORA))}
        wo_a.outputs = {"y": Tensor("bf16", (B, S, O_GROUPS * O_LORA))}
        g.add_kernel(wo_a)
        g.add_data_edge(sa, wo_a, {"y": "x"})
        L.wo_a = wo_a

        wo_b = make_gemm(B, S, D, O_GROUPS * O_LORA, "fp8")
        g.add_kernel(wo_b)
        g.add_data_edge(wo_a, wo_b, {"y": "x"})
        L.wo_b = wo_b

        # ── Attention residual: input + attention_output ──────────
        attn_add = ElementwiseOp(M, D, "bf16")
        attn_add.inputs = {"a": Tensor("bf16", (B, S, D)),
                           "b": Tensor("bf16", (B, S, D))}
        attn_add.outputs = {"y": Tensor("bf16", (B, S, D))}
        g.add_kernel(attn_add)
        if context_len is None or decode_has_comp:
            g.add_data_edge(bridge, attn_add, {"y3": "a"})
        else:
            g.add_data_edge(bridge, attn_add, {"y2": "a"})
        g.add_data_edge(wo_b, attn_add, {"y": "b"})
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
        ffn_norm = make_norm(B, S, D)
        g.add_kernel(ffn_norm)
        g.add_data_edge(ffn_bridge, ffn_norm, {"y": "x"})
        L.ffn_norm = ffn_norm

        # FFN fan-out: gate + dispatch + shared expert (3 consumers)
        ffn_fan = Spawn(world=3)
        ffn_fan.inputs = {"x": Tensor("bf16", (B, S, D))}
        ffn_fan.outputs = {"y": Tensor("bf16", (B, S, D)),
                           "y2": Tensor("bf16", (B, S, D)),
                           "y3": Tensor("bf16", (B, S, D))}
        g.add_kernel(ffn_fan)
        g.add_data_edge(ffn_norm, ffn_fan, {"y": "x"})
        L.ffn_fan = ffn_fan

        gate = make_gemm(B, S, N_EXPERTS, D, "bf16", "bf16", "fp32")
        g.add_kernel(gate)
        g.add_data_edge(ffn_fan, gate, {"y": "x"})
        L.gate = gate

        # Dispatch: softmax routing + token scatter to experts
        M_e = M * TOPK // N_EXPERTS
        dispatch = TokenDispatch(M, D, N_EXPERTS, TOPK)
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
            up = StridedGemm(M_e, 2 * MOE_INTER, D, "fp4", "bf16",
                             out_elems=M_e * MOE_INTER)
            up.inputs = {"x": Tensor("bf16", (M_e, D))}
            up.weights = {"w": Tensor("fp4", (D, 2 * MOE_INTER))}
            scale_bytes = gemm_scale_bytes(2 * MOE_INTER, D, "fp4")
            if scale_bytes > 0:
                up.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            up.outputs = {"y": Tensor("bf16", (M_e, MOE_INTER))}
            g.add_kernel(up)

            down = Gemm(M_e, D, MOE_INTER, "fp4", "bf16")
            down.inputs = {"x": Tensor("bf16", (M_e, MOE_INTER))}
            down.weights = {"w": Tensor("fp4", (MOE_INTER, D))}
            scale_bytes = gemm_scale_bytes(D, MOE_INTER, "fp4")
            if scale_bytes > 0:
                down.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            down.outputs = {"y": Tensor("bf16", (M_e, D))}
            g.add_kernel(down)
            g.add_data_edge(up, down, {"y": "x"})
            g.add_data_edge(dispatch, up, {f"o{eid}": "x"})
            g.add_data_edge(down, combine, {"y": f"i{eid}"})

            L.experts.extend([up, down])

        # Shared expert (parallel with routed — reads from ffn_fan)
        sw_up = make_gated_up(B, S, MOE_INTER, D, "fp8", "bf16")
        g.add_kernel(sw_up)
        g.add_data_edge(ffn_fan, sw_up, {"y3": "x"})
        L.sw_up = sw_up

        sw_down = make_gemm(B, S, D, MOE_INTER, "fp8", "bf16")
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


def _build_kv_cache_views(g, B, cache_src, cache_src_port, cache_len,
                          win_keep, comp_len):
    """Fan out a KV cache into its window and compressed portions."""
    kv_cache_fan = Spawn(world=2)
    kv_cache_fan.inputs = {
        "x": Tensor("bf16", (B, cache_len, KV_DIM))}
    kv_cache_fan.outputs = {
        "y": Tensor("bf16", (B, cache_len, KV_DIM)),
        "y2": Tensor("bf16", (B, cache_len, KV_DIM))}
    g.add_kernel(kv_cache_fan)
    g.add_data_edge(cache_src, kv_cache_fan, {cache_src_port: "x"})

    kv_win_slice = Slice()
    kv_win_slice.inputs = {
        "x": Tensor("bf16", (B, cache_len, KV_DIM))}
    kv_win_slice.outputs = {
        "y": Tensor("bf16", (B, win_keep, KV_DIM))}
    g.add_kernel(kv_win_slice)
    g.add_data_edge(kv_cache_fan, kv_win_slice, {"y": "x"})

    kv_comp_slice = Slice()
    kv_comp_slice.inputs = {
        "x": Tensor("bf16", (B, cache_len, KV_DIM))}
    kv_comp_slice.outputs = {
        "y": Tensor("bf16", (B, comp_len, KV_DIM))}
    g.add_kernel(kv_comp_slice)
    g.add_data_edge(kv_cache_fan, kv_comp_slice, {"y2": "x"})

    return kv_cache_fan, kv_win_slice, kv_comp_slice


def _build_kv_accumulator(g, B, S_kv, win_keep, comp_cache_len,
                          has_comp, layer, kv_win_src, comp_cache_src):
    """Append current KV and optional compression output to cached KV."""
    kv_acc = Concat()
    kv_acc.inputs = {
        "window": Tensor("bf16", (B, win_keep, KV_DIM)),
        "kv": Tensor("bf16", (B, 1, KV_DIM)),
        "comp_cache": Tensor("bf16", (B, comp_cache_len, KV_DIM)),
    }
    if has_comp:
        kv_acc.inputs["comp"] = Tensor("bf16", (B, 1, KV_DIM))
    kv_acc.outputs = {"y": Tensor("bf16", (B, S_kv, KV_DIM))}
    g.add_kernel(kv_acc)

    g.add_data_edge(kv_win_src, kv_acc, {"y": "window"})
    g.add_data_edge(layer.kv_norm, kv_acc, {"y": "kv"})
    comp_cache_kernel, comp_cache_port = comp_cache_src
    g.add_data_edge(comp_cache_kernel, kv_acc,
                    {comp_cache_port: "comp_cache"})
    if has_comp:
        g.add_data_edge(layer.comp_norm, kv_acc, {"y": "comp"})

    return kv_acc


def _build_decode_steps(g, B, pfx_len, n_steps, prefill_layers=None,
                        kv_cache_reads=None, first_token_src=None):
    """Unroll n_steps decode steps into graph g.

    Args:
        g: ComputeGraph to add kernels to.
        B: batch size.
        pfx_len: number of tokens already processed before decode starts.
        n_steps: number of decode steps to unroll.
        prefill_layers: List[LayerMeta] from prefill (for prefill+decode KV).
        kv_cache_reads: List[Kernel] per-layer ReadInput (for decode-only KV).
        first_token_src: (kernel, edge_map) for step 0's token input.
            e.g. (pfx_sampling, {"y": "idx"}) or (dec_read, {"tokens": "idx"}).

    Returns: List[DecodeStepMeta], one per step.
    """
    has_prefill = prefill_layers is not None and len(prefill_layers) > 0
    steps = []

    for step_idx in range(n_steps):
        context_len = pfx_len + step_idx

        # ── Token input + embedding ───────────────────────────────────
        dec_emb = Embedding(B, V, D)
        dec_emb.inputs = {"idx": Tensor("int32", (B, 1, 1))}
        dec_emb.weights = {"emb": Tensor("bf16", (V, D))}
        dec_emb.outputs = {"y": Tensor("bf16", (B, 1, D))}
        g.add_kernel(dec_emb)

        if step_idx == 0:
            token_src, token_edge = first_token_src
            g.add_data_edge(token_src, dec_emb, token_edge)
            dec_read = token_src if not has_prefill else None
        else:
            prev_step = steps[step_idx - 1]
            g.add_data_edge(prev_step.sampling, dec_emb, {"y": "idx"})
            dec_read = None

        _tag_weights(dec_emb, -1, "emb")

        # ── Layers ────────────────────────────────────────────────────
        dec_layers, step_last = _build_layers(
            g, B, 1, context_len, dec_emb)

        # ── Output head ───────────────────────────────────────────────
        final_norm, logits, sampling = _build_output_head(g, B, step_last)

        decode_step = DecodeStepMeta(
            read_input=dec_read, emb=dec_emb,
            final_norm=final_norm, logits=logits, sampling=sampling,
            layers=dec_layers)

        uses_prefill_cache = step_idx == 0 and has_prefill

        # ── KV cache data edges ───────────────────────────────────────
        for layer_id in range(N_LAYERS):
            ratio = COMPRESS_RATIOS[layer_id]
            dec_L = decode_step.layers[layer_id]

            has_comp = (context_len + 1) % ratio == 0
            total_compressed = (context_len + 1) // ratio
            S_kv = min(WINDOW, context_len + 1) + total_compressed
            win_keep = min(WINDOW - 1, context_len)
            if uses_prefill_cache:
                # Step 0 prefill+decode: KV from prefill window fan-out
                pfx_L = prefill_layers[layer_id]
                comp_cache_len = pfx_len // ratio

                kv_win_slice = Slice()
                kv_win_slice.inputs = {
                    "x": Tensor("bf16", (B, WINDOW, KV_DIM))}
                kv_win_slice.outputs = {
                    "y": Tensor("bf16", (B, win_keep, KV_DIM))}
                g.add_kernel(kv_win_slice)
                g.add_data_edge(pfx_L.kv_norm_fan, kv_win_slice,
                                {"y2": "x"})
                dec_L.kv_win_slice = kv_win_slice

                comp_cache_src = (pfx_L.comp_norm_fan, "y2")

            elif step_idx == 0:
                # Step 0 decode-only: KV from ReadInput
                cache_len = min(WINDOW, pfx_len) + pfx_len // ratio
                comp_cache_len = pfx_len // ratio
                kv_cache_fan, kv_win_slice, kv_comp_slice = (
                    _build_kv_cache_views(
                        g, B, kv_cache_reads[layer_id], "y", cache_len,
                        win_keep, comp_cache_len))

            else:
                # Steps 1+: KV from previous step's kv_acc_fan
                prev_L = steps[step_idx - 1].layers[layer_id]
                prev_kv_acc_fan = prev_L.kv_acc_fan
                cache_len = prev_kv_acc_fan.outputs["y2"].shape[1]
                comp_cache_len = context_len // ratio
                kv_cache_fan, kv_win_slice, kv_comp_slice = (
                    _build_kv_cache_views(
                        g, B, prev_kv_acc_fan, "y2", cache_len,
                        win_keep, comp_cache_len))

            if not uses_prefill_cache:
                dec_L.kv_cache_fan = kv_cache_fan
                dec_L.kv_win_slice = kv_win_slice
                dec_L.kv_comp_slice = kv_comp_slice
                comp_cache_src = (kv_comp_slice, "y")

            kv_acc = _build_kv_accumulator(
                g, B, S_kv, win_keep, comp_cache_len, has_comp,
                dec_L, kv_win_slice, comp_cache_src)

            # Fan-out kv_acc for multi-step: sa + next step's KV chain
            if step_idx < n_steps - 1:
                kv_acc_fan = Spawn(world=2)
                kv_acc_fan.inputs = {
                    "x": Tensor("bf16", kv_acc.outputs["y"].shape)}
                kv_acc_fan.outputs = {
                    "y": Tensor("bf16", kv_acc.outputs["y"].shape),
                    "y2": Tensor("bf16", kv_acc.outputs["y"].shape)}
                g.add_kernel(kv_acc_fan)
                g.add_data_edge(kv_acc, kv_acc_fan, {"y": "x"})
                g.add_data_edge(kv_acc_fan, dec_L.sa, {"y": "kv"})
                dec_L.kv_acc_fan = kv_acc_fan
            else:
                g.add_data_edge(kv_acc, dec_L.sa, {"y": "kv"})
            dec_L.kv_acc = kv_acc

        steps.append(decode_step)

    return steps


def declare_model(
    batch_size=BATCH,
    seq_prefill=S_PREFILL,
    decode=False,
    kv_prefill_len=None,
    n_decode_steps=1,
    persist_kv_cache=False,
):
    """Build compute graph for prefill, decode, or both.

    Args:
        batch_size: number of independent sequences (>= 64 for MoE routing).
        seq_prefill: prefill sequence length. None to skip prefill.
        decode: whether to build decode step(s).
        kv_prefill_len: for decode-only mode, the original prefill length
            that generated the KV cache. Ignored when seq_prefill is set.
        n_decode_steps: number of decode steps to unroll (default 1).
        persist_kv_cache: whether the placement optimizer should retain every
            compact prefill KV cache until prefill completes.

    Returns:
        (g, prefill_layers, decode_steps, emb, read_input, kv_cache_reads,
         pfx_out_head)
        - prefill_layers: list[LayerMeta] (empty if seq_prefill is None)
        - decode_steps: list[DecodeStepMeta] (empty if decode=False)
        - emb: prefill Embedding kernel (None if no prefill)
        - read_input: prefill ReadInput kernel (None if no prefill)
        - kv_cache_reads: list[Kernel] per-layer KV cache ReadInput
            (empty unless decode-only mode)
        - pfx_out_head: list of prefill output head kernels (empty if N/A)
    """
    g = ComputeGraph()
    B = batch_size
    has_prefill = seq_prefill is not None
    has_decode = decode
    if persist_kv_cache and not has_prefill:
        raise ValueError("persist_kv_cache requires a prefill graph")

    prefill_layers = []
    decode_steps = []
    emb = None
    read_input = None
    kv_cache_reads = []

    # ── Prefill ──────────────────────────────────────────────────
    if has_prefill:
        S_p = seq_prefill
        M_p = B * S_p

        read_input = ReadInput(M_p, "int32")
        read_input.inputs = {"tokens": Tensor("int32", (B, S_p, 1))}
        read_input.outputs = {"tokens": Tensor("int32", (B, S_p, 1))}
        g.add_kernel(read_input)

        emb = Embedding(M_p, V, D)
        emb.inputs = {"idx": Tensor("int32", (B, S_p, 1))}
        emb.weights = {"emb": Tensor("bf16", (V, D))}
        emb.outputs = {"y": Tensor("bf16", (B, S_p, D))}
        g.add_kernel(emb)
        g.add_data_edge(read_input, emb, {"tokens": "idx"})
        _tag_weights(emb, -1, "emb")

        prefill_layers, prefill_last = _build_layers(
            g, B, S_p, None, emb, has_decode=has_decode,
            capture_kv_cache=persist_kv_cache)
        for layer in prefill_layers:
            layer.persist_kv_cache = persist_kv_cache
        if persist_kv_cache:
            barrier = Nop()
            barrier.inputs = {
                f"kv{layer_id}": Tensor(
                    layer.kv_persist_fan.outputs["y2"].dtype,
                    layer.kv_persist_fan.outputs["y2"].shape,
                )
                for layer_id, layer in enumerate(prefill_layers)
            }
            barrier.inputs["prefill_output"] = Tensor(
                prefill_last.outputs["y"].dtype,
                prefill_last.outputs["y"].shape,
            )
            barrier.outputs = {"done": Tensor("int32", (1,))}
            g.add_kernel(barrier)
            for layer_id, layer in enumerate(prefill_layers):
                g.add_data_edge(
                    layer.kv_persist_fan, barrier,
                    {"y2": f"kv{layer_id}"},
                )
            g.add_data_edge(
                prefill_last, barrier, {"y": "prefill_output"})
            for layer in prefill_layers:
                layer.kv_persist_barrier = barrier

    # ── Decode steps ─────────────────────────────────────────────
    if has_decode:
        pfx_len = seq_prefill if has_prefill else kv_prefill_len
        if pfx_len is None:
            raise ValueError(
                "kv_prefill_len required when seq_prefill is None")

        # For decode-only: per-layer KV cache ReadInput (from CPU/NVMe)
        if not has_prefill:
            for layer_id in range(N_LAYERS):
                ratio = COMPRESS_RATIOS[layer_id]
                cache_len = min(WINDOW, pfx_len) + pfx_len // ratio
                kv_read = ReadInput(B * cache_len * KV_DIM, "bf16")
                kv_read.inputs = {
                    "kv": Tensor("bf16", (B, cache_len, KV_DIM))}
                kv_read.outputs = {
                    "y": Tensor("bf16", (B, cache_len, KV_DIM))}
                g.add_kernel(kv_read)
                kv_cache_reads.append(kv_read)

        # Prefill output head: last position → sampling → first decode token
        prefill_output_head = []
        if has_prefill:
            S_p = seq_prefill
            pfx_slice = Slice()
            pfx_slice.inputs = {"x": Tensor("bf16", (B, S_p, D))}
            pfx_slice.outputs = {"y": Tensor("bf16", (B, 1, D))}
            g.add_kernel(pfx_slice)
            g.add_data_edge(prefill_last, pfx_slice, {"y": "x"})

            pfx_norm, pfx_logits, pfx_sampling = _build_output_head(
                g, B, pfx_slice)

            prefill_output_head = [pfx_slice, pfx_norm, pfx_logits,
                                   pfx_sampling]

        # Determine first token source for decode step 0
        if has_prefill:
            first_token_src = (pfx_sampling, {"y": "idx"})
        else:
            dec_read = ReadInput(B, "int32")
            dec_read.inputs = {"tokens": Tensor("int32", (B, 1, 1))}
            dec_read.outputs = {"tokens": Tensor("int32", (B, 1, 1))}
            g.add_kernel(dec_read)
            first_token_src = (dec_read, {"tokens": "idx"})

        decode_steps = _build_decode_steps(
            g, B, pfx_len, n_decode_steps,
            prefill_layers=prefill_layers if has_prefill else None,
            kv_cache_reads=kv_cache_reads if not has_prefill else None,
            first_token_src=first_token_src)

    g.validate()
    pfx_out_head = prefill_output_head if has_decode and has_prefill else []
    return (g, prefill_layers, decode_steps, emb, read_input,
            kv_cache_reads, pfx_out_head)
