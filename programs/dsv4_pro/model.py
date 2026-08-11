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
    # KV cache and persistence metadata
    kv_win_slice: Kernel = None
    kv_cache_fan: Kernel = None
    kv_persist_fan: Kernel = None
    kv_persist_barrier: Kernel = None

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


def _build_layers(g, B, S, context_len, prev_out):
    """Build N_LAYERS transformer layers into graph g.

    Args:
        g: ComputeGraph to add kernels to.
        B: batch size.
        S: sequence length.
        context_len: None for prefill mode (builds compressor + kv_concat).
            For decode mode: integer pfx_len (total tokens already processed).
            Per-layer S_kv = WINDOW + total_compressed.
        prev_out: kernel whose "y" output feeds into the first layer.
    Returns (layers, last_output_kernel).
    """
    M = B * S
    layers = []
    is_prefill = context_len is None

    for layer_id in range(N_LAYERS):
        ratio = COMPRESS_RATIOS[layer_id]
        L = LayerMeta()

        # ── Input fan-out (residual + attention + optional compressor) ──
        bridge = Spawn(world=3 if is_prefill else 2)
        bridge.inputs = {"x": Tensor("bf16", (B, S, D))}
        bridge.outputs = {
            "y": Tensor("bf16", (B, S, D)),
            "y2": Tensor("bf16", (B, S, D)),
        }
        if is_prefill:
            bridge.outputs["y3"] = Tensor("bf16", (B, S, D))
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

        if is_prefill:
            compressed_len = S // ratio
            window_len = WINDOW

            # Prefill builds the compact KV cache consumed by attention.
            if ratio in (128, 4):
                coff = 2 if ratio == 4 else 1
                M_comp = B * compressed_len
                comp_out_elems = M_comp * KV_DIM
                comp = StridedGemm(M, 2 * coff * KV_DIM, D, "fp32", "bf16",
                                   in_elems=M * D, out_elems=comp_out_elems)
                comp.inputs = {"x": Tensor("bf16", (B, S, D))}
                comp.weights = {"w": Tensor("fp32", (D, 2 * coff * KV_DIM))}
                comp.outputs = {
                    "y": Tensor("bf16", (B, compressed_len, KV_DIM))}
                g.add_kernel(comp)
                g.add_data_edge(bridge, comp, {"y2": "x"})
                L.comp = comp

                comp_norm = make_norm(B, compressed_len, KV_DIM)
                g.add_kernel(comp_norm)
                g.add_data_edge(comp, comp_norm, {"y": "x"})
                L.comp_norm = comp_norm
        else:
            compressed_len = context_len // ratio
            window_len = min(WINDOW, context_len)

        S_kv = window_len + compressed_len
        k_sel = WINDOW
        if ratio == 128:
            k_sel += compressed_len
        elif ratio == 4:
            k_sel += INDEX_TOPK

        sa = SparseAttn(B, H, 1, S, k_sel, S_kv, HD, "bf16", kv_factor=1)
        sa.inputs = {"q": Tensor("bf16", (B, S, H * HD)),
                     "kv": Tensor("bf16", (B, S_kv, KV_DIM))}
        sa.outputs = {"y": Tensor("bf16", (B, S, H * HD))}
        g.add_kernel(sa)
        g.add_data_edge(wq_b, sa, {"y": "q"})

        if is_prefill:
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
                                "b": Tensor(
                                    "bf16", (B, compressed_len, KV_DIM))}
            kv_concat.outputs = {"y": Tensor("bf16", (B, S_kv, KV_DIM))}
            g.add_kernel(kv_concat)

            g.add_data_edge(kv_win_slice, kv_concat, {"y": "a"})
            g.add_data_edge(comp_norm, kv_concat, {"y": "b"})

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

            L.kv_win_slice = kv_win_slice
            L.kv_concat = kv_concat

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
        residual_port = "y3" if is_prefill else "y2"
        g.add_data_edge(bridge, attn_add, {residual_port: "a"})
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


def _build_decode_kv_cache_read(g, B, context_len, layers):
    """Build persistent external KV-cache reads for decode attention."""
    kv_cache_reads = []
    for layer_id, layer in enumerate(layers):
        ratio = COMPRESS_RATIOS[layer_id]
        cache_len = min(WINDOW, context_len) + context_len // ratio

        kv_read = ReadInput(B * cache_len * KV_DIM, "bf16")
        kv_read.inputs = {
            "kv": Tensor("bf16", (B, cache_len, KV_DIM))}
        kv_read.outputs = {
            "y": Tensor("bf16", (B, cache_len, KV_DIM))}
        g.add_kernel(kv_read)

        cache_fan = Spawn(world=2)
        cache_fan.inputs = {
            "x": Tensor("bf16", (B, cache_len, KV_DIM))}
        cache_fan.outputs = {
            "y": Tensor("bf16", (B, cache_len, KV_DIM)),
            "y2": Tensor("bf16", (B, cache_len, KV_DIM)),
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
    """Keep every compact KV cache alive until the stage output is ready."""
    barrier = Nop()
    barrier.inputs = {
        f"kv{layer_id}": Tensor(
            layer.kv_persist_fan.outputs["y2"].dtype,
            layer.kv_persist_fan.outputs["y2"].shape,
        )
        for layer_id, layer in enumerate(layers)
    }
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
            whose KV cache is consumed by decode.
        decode: whether to build the single-step decode stage.

    Returns:
        (g, layers, emb, read_input, kv_cache_reads, output_head)
        - layers: list[LayerMeta] for the selected stage
        - emb/read_input: token input kernels shared by both stages
        - kv_cache_reads: per-layer KV cache ReadInput kernels (decode only)
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
