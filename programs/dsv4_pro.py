"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 DP=8.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model() — split_kernel for DP, control edges, placement
  C. simulate() — DES execution + trace export
"""

from dataclasses import dataclass, field
from typing import List, Set

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.forward import (
    ElementwiseOp, Embedding, Gemm, ReadInput, RMSNorm, Sampling, SparseAttn,
    StridedGemm, TokenCombine, TokenDispatch,
)
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.optimization.comm import optimize_comms
from rooflang.language.optimization.split import (
    batch_split,
)
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.language.utils import dtype_bytes, gemm_scale_bytes
from rooflang.programs.presets.b300 import B300ClusterA, B300SuperChipA
from rooflang.runtime.simulator import Simulator
from rooflang.runtime.trace_export import export_trace
from rooflang.runtime.graph_export import export_graph

# ── Config ──────────────────────────────────────────────────────────────
D = 7168
N_LAYERS = 61
H = 128
HD = 512
Q_LORA = 1536
KV_DIM = 512
O_GROUPS = 16
O_LORA = 1024
N_EXPERTS = 384
TOPK = 6
MOE_INTER = 3072
WINDOW = 128
INDEX_TOPK = 1024
DP = 8
EP = 8
N_LOCAL_EXPERTS = N_EXPERTS // EP
V = 129280
BATCH = 512
S_PREFILL = 8192
COMPRESS_RATIOS = [128, 128] + [v for _ in range(29) for v in (4, 128)] + [4]


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
    experts: List[List[Kernel]] = field(default_factory=list)
    # Decode KV chain (set by declare_model after _build_layers)
    kv_acc: Kernel = None
    kv_spawn: Kernel = None


@dataclass
class DecodeStepMeta:
    """Per-decode-step metadata (token input + embedding + layer list)."""
    read_input: Kernel = None
    emb: Kernel = None
    final_norm: Kernel = None
    logits: Kernel = None
    sampling: Kernel = None
    layers: List[LayerMeta] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# A. Declaration Phase
# ═══════════════════════════════════════════════════════════════════════

def _build_layers(g, B, S, context_len, prev_out, has_decode=False):
    """Build N_LAYERS transformer layers into graph g.

    Args:
        g: ComputeGraph to add kernels to.
        B: batch size.
        S: sequence length.
        context_len: None for prefill mode (builds compressor + kv_concat).
            For decode mode: (prefill_S, n_cached_decode_tokens) tuple.
            Per-layer S_kv = WINDOW + total_compressed.
        prev_out: kernel whose "y" output feeds into the first layer.
        has_decode: if True (prefill in full model), add Spawn fans after
            kv_norm/comp_norm so decode can read the initial KV cache.

    Returns (layers, last_output_kernel).
    """
    M = B * S
    is_decode = context_len is not None
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
            decode_has_comp = (S >= ratio)
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

            # Sparse attention (prefill: S_kv = S + S//ratio)
            k_sel = WINDOW
            if ratio == 128:
                k_sel = WINDOW + S // 128
            elif ratio == 4:
                k_sel = WINDOW + INDEX_TOPK

            S_kv = S + S // ratio
            S_comp = S // ratio
            sa = SparseAttn(B, H, 1, S, k_sel, S_kv, HD, "bf16", kv_factor=1)
            sa.inputs = {"q": Tensor("bf16", (B, S, H * HD)),
                         "kv": Tensor("bf16", (B, S_kv, KV_DIM))}
            sa.outputs = {"y": Tensor("bf16", (B, S, H * HD))}
            g.add_kernel(sa)
            g.add_data_edge(wq_b, sa, {"y": "q"})

            # KV cache = concat(window_kv, compressed_kv)
            kv_concat = Concat()
            kv_concat.inputs = {"a": Tensor("bf16", (B, S, KV_DIM)),
                                "b": Tensor("bf16", (B, S_comp, KV_DIM))}
            kv_concat.outputs = {"y": Tensor("bf16", (B, S_kv, KV_DIM))}
            g.add_kernel(kv_concat)

            if has_decode:
                # Fan-out kv_norm and comp_norm: y→kv_concat, y2→decode
                kv_norm_fan = Spawn(world=2)
                kv_norm_fan.inputs = {"x": Tensor("bf16", (B, S, KV_DIM))}
                kv_norm_fan.outputs = {
                    "y": Tensor("bf16", (B, S, KV_DIM)),
                    "y2": Tensor("bf16", (B, S, KV_DIM))}
                g.add_kernel(kv_norm_fan)
                g.add_data_edge(kv_norm, kv_norm_fan, {"y": "x"})
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
                g.add_data_edge(kv_norm, kv_concat, {"y": "a"})
                g.add_data_edge(comp_norm, kv_concat, {"y": "b"})

            g.add_data_edge(kv_concat, sa, {"y": "kv"})

            L.kv_concat = kv_concat
        else:
            # Decode: window + compressed KV cache
            pfx_S, n_cached = context_len
            decode_has_comp = (S >= ratio)

            if decode_has_comp:
                total_compressed = (pfx_S + n_cached + S) // ratio
            else:
                total_compressed = pfx_S // ratio

            S_kv = min(WINDOW, pfx_S + n_cached + S) + total_compressed

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

            # Compressor for ratio=4 layers (fires every step)
            if decode_has_comp:
                coff = 2 if ratio == 4 else 1
                S_comp = S // ratio
                comp_out_elems = B * S_comp * KV_DIM
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

        # Expert kernels per GPU (up_proj + down_proj per expert)
        L.experts = []
        for gpu_id in range(DP):
            gpu_experts = []
            for eid in range(N_LOCAL_EXPERTS):
                global_eid = gpu_id * N_LOCAL_EXPERTS + eid
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
                g.add_data_edge(dispatch, up, {f"o{global_eid}": "x"})
                g.add_data_edge(down, combine, {"y": f"i{global_eid}"})

                gpu_experts.extend([up, down])

            L.experts.append(gpu_experts)

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

        prev_out = ffn_add
        layers.append(L)

    return layers, prev_out


def declare_model(batch_size=BATCH, seq_prefill=S_PREFILL,
                  n_decode_steps=0, kv_prefill_len=None):
    """Build compute graph for prefill, decode, or both.

    Three modes:
      - Prefill only: declare_model(batch_size=64, seq_prefill=8192)
      - Decode only:  declare_model(batch_size=64, seq_prefill=None,
                          n_decode_steps=4, kv_prefill_len=8192)
      - Prefill+decode: declare_model(batch_size=64, seq_prefill=8192,
                          n_decode_steps=4)

    Args:
        batch_size: number of independent sequences (>= 64 for MoE routing).
        seq_prefill: prefill sequence length. None to skip prefill.
        n_decode_steps: number of decode steps. 0 for prefill-only.
        kv_prefill_len: for decode-only mode, the original prefill length
            that generated the KV cache (needed to compute per-layer cache
            sizes). Ignored when seq_prefill is set.

    Returns:
        (g, prefill_layers, decode_steps, emb, read_input, kv_cache_reads,
         pfx_out_head)
        - prefill_layers: list[LayerMeta] (empty if seq_prefill is None)
        - decode_steps: list[DecodeStepMeta], one per step
        - emb: prefill Embedding kernel (None if no prefill)
        - read_input: prefill ReadInput kernel (None if no prefill)
        - kv_cache_reads: list[Kernel] per-layer KV cache ReadInput
            (empty unless decode-only mode)
        - pfx_out_head: list of prefill output head kernels (empty if N/A)
    """
    g = ComputeGraph()
    B = batch_size
    has_prefill = seq_prefill is not None
    has_decode = n_decode_steps > 0

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

        prefill_layers, prefill_last = _build_layers(
            g, B, S_p, None, emb, has_decode=has_decode)

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
                cache_len = WINDOW + pfx_len // ratio
                kv_read = ReadInput(B * cache_len * KV_DIM, "bf16")
                kv_read.inputs = {
                    "kv": Tensor("bf16", (B, cache_len, KV_DIM))}
                kv_read.outputs = {
                    "y": Tensor("bf16", (B, cache_len, KV_DIM))}
                g.add_kernel(kv_read)
                kv_cache_reads.append(kv_read)

        # Prefill output head: last position → sampling → first decode token
        prefill_output_head = []  # [slice, norm, logits, sampling]
        if has_prefill:
            S_p = seq_prefill
            pfx_slice = Concat()
            pfx_slice.inputs = {"x": Tensor("bf16", (B, S_p, D))}
            pfx_slice.outputs = {"y": Tensor("bf16", (B, 1, D))}
            g.add_kernel(pfx_slice)
            g.add_data_edge(prefill_last, pfx_slice, {"y": "x"})

            pfx_norm = make_norm(B, 1, D)
            g.add_kernel(pfx_norm)
            g.add_data_edge(pfx_slice, pfx_norm, {"y": "x"})

            pfx_logits = make_gemm(B, 1, V, D, "bf16")
            g.add_kernel(pfx_logits)
            g.add_data_edge(pfx_norm, pfx_logits, {"y": "x"})

            pfx_sampling = Sampling(B, V)
            pfx_sampling.inputs = {"logits": Tensor("bf16", (B, 1, V))}
            pfx_sampling.outputs = {"y": Tensor("int32", (B, 1, 1))}
            g.add_kernel(pfx_sampling)
            g.add_data_edge(pfx_logits, pfx_sampling, {"y": "logits"})

            prefill_output_head = [pfx_slice, pfx_norm, pfx_logits,
                                   pfx_sampling]
            prev_step_sampling = pfx_sampling
        else:
            prev_step_sampling = None

        for step in range(n_decode_steps):
            ctx = (pfx_len, step)

            # Input: single token (B, 1, 1) → embedding (B, 1, D)
            dec_emb = Embedding(B, V, D)
            dec_emb.inputs = {"idx": Tensor("int32", (B, 1, 1))}
            dec_emb.weights = {"emb": Tensor("bf16", (V, D))}
            dec_emb.outputs = {"y": Tensor("bf16", (B, 1, D))}
            g.add_kernel(dec_emb)

            if prev_step_sampling is not None:
                g.add_data_edge(prev_step_sampling, dec_emb, {"y": "idx"})
                dec_read = None
            else:
                # Step 0 decode-only: single token from host
                dec_read = ReadInput(B, "int32")
                dec_read.inputs = {"tokens": Tensor("int32", (B, 1, 1))}
                dec_read.outputs = {"tokens": Tensor("int32", (B, 1, 1))}
                g.add_kernel(dec_read)
                g.add_data_edge(dec_read, dec_emb, {"tokens": "idx"})

            # Layers process single token: S=1
            dec_step_layers, step_last = _build_layers(
                g, B, 1, ctx, dec_emb)

            # Output head: norm → logits → sampling (all on single token)
            final_norm = make_norm(B, 1, D)
            g.add_kernel(final_norm)
            g.add_data_edge(step_last, final_norm, {"y": "x"})

            logits = make_gemm(B, 1, V, D, "bf16")
            g.add_kernel(logits)
            g.add_data_edge(final_norm, logits, {"y": "x"})

            sampling = Sampling(B, V)
            sampling.inputs = {"logits": Tensor("bf16", (B, 1, V))}
            sampling.outputs = {"y": Tensor("int32", (B, 1, 1))}
            g.add_kernel(sampling)
            g.add_data_edge(logits, sampling, {"y": "logits"})

            prev_step_sampling = sampling

            step_meta = DecodeStepMeta(
                read_input=dec_read, emb=dec_emb,
                final_norm=final_norm, logits=logits, sampling=sampling,
                layers=dec_step_layers)
            decode_steps.append(step_meta)

        # ── KV cache data edges (window + compressed) ────────────────
        # With S_d=1, no compressor fires in decode (1 < ratio for all layers).
        # KV cache size: WINDOW + pfx_len//ratio (static, grows by 1 per step
        # in the window ring buffer but total size is capped at WINDOW).
        for layer_id in range(N_LAYERS):
            ratio = COMPRESS_RATIOS[layer_id]
            total_compressed = pfx_len // ratio

            if not has_prefill:
                prev_cache_src = kv_cache_reads[layer_id]
                prev_cache_out_port = "y"
                prev_cache_len = WINDOW + total_compressed

            for step in range(n_decode_steps):
                dec_L = decode_steps[step].layers[layer_id]

                current_S_kv = (min(WINDOW, pfx_len + step + 1)
                                + total_compressed)

                kv_acc = Concat()
                if step == 0 and has_prefill:
                    pfx_L = prefill_layers[layer_id]
                    S_p = seq_prefill
                    kv_acc.inputs = {
                        "pfx_kv": Tensor("bf16", (B, S_p, KV_DIM)),
                        "pfx_comp": Tensor("bf16",
                                           (B, S_p // ratio, KV_DIM)),
                        "kv": Tensor("bf16", (B, 1, KV_DIM)),
                    }
                    kv_acc.outputs = {
                        "y": Tensor("bf16", (B, current_S_kv, KV_DIM))}
                    g.add_kernel(kv_acc)
                    g.add_data_edge(pfx_L.kv_norm_fan, kv_acc,
                                    {"y2": "pfx_kv"})
                    g.add_data_edge(pfx_L.comp_norm_fan, kv_acc,
                                    {"y2": "pfx_comp"})
                    g.add_data_edge(dec_L.kv_norm, kv_acc, {"y": "kv"})
                else:
                    kv_acc.inputs = {
                        "a": Tensor("bf16",
                                    (B, prev_cache_len, KV_DIM)),
                        "b": Tensor("bf16", (B, 1, KV_DIM)),
                    }
                    kv_acc.outputs = {
                        "y": Tensor("bf16", (B, current_S_kv, KV_DIM))}
                    g.add_kernel(kv_acc)
                    g.add_data_edge(prev_cache_src, kv_acc,
                                    {prev_cache_out_port: "a"})
                    g.add_data_edge(dec_L.kv_norm, kv_acc, {"y": "b"})

                dec_L.kv_acc = kv_acc

                if step < n_decode_steps - 1:
                    kv_step_fan = Spawn(world=2)
                    kv_step_fan.inputs = {
                        "x": Tensor("bf16",
                                    (B, current_S_kv, KV_DIM))}
                    kv_step_fan.outputs = {
                        "y": Tensor("bf16",
                                    (B, current_S_kv, KV_DIM)),
                        "y2": Tensor("bf16",
                                     (B, current_S_kv, KV_DIM))}
                    g.add_kernel(kv_step_fan)
                    g.add_data_edge(kv_acc, kv_step_fan, {"y": "x"})
                    g.add_data_edge(kv_step_fan, dec_L.sa, {"y": "kv"})
                    dec_L.kv_spawn = kv_step_fan
                    prev_cache_src = kv_step_fan
                    prev_cache_out_port = "y2"
                else:
                    g.add_data_edge(kv_acc, dec_L.sa, {"y": "kv"})

                prev_cache_len = current_S_kv

    g.validate()
    pfx_out_head = prefill_output_head if has_decode and has_prefill else []
    return (g, prefill_layers, decode_steps, emb, read_input,
            kv_cache_reads, pfx_out_head)


# ═══════════════════════════════════════════════════════════════════════
# B. Optimization Phase
# ═══════════════════════════════════════════════════════════════════════

def optimize_model(g, layers, hw, emb=None, read_input=None,
                   decode_steps=None, kv_cache_reads=None,
                   prefill_output_head=None):
    """Apply split_kernel for DP, add control edges, and place."""
    gpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and "nvidia-b300" in c.name],
        key=lambda c: c.name)

    # ── Phase 1: DP splits (batch dim) ───────────────────────────────
    emb_copies = None
    if emb is not None:
        _, emb_copies, _ = g.split_kernel(batch_split, emb, DP)

    def _split_layer(L):
        """Split all kernels in a LayerMeta by batch dimension."""
        _, L._bridge_copies, _ = g.split_kernel(batch_split, L.bridge, DP)
        _, L._attn_norm_copies, _ = g.split_kernel(batch_split, L.attn_norm, DP)
        _, L._attn_fan_copies, _ = g.split_kernel(batch_split, L.attn_fan, DP)
        if L.comp is not None:
            _, L._comp_copies, _ = g.split_kernel(batch_split, L.comp, DP)
        if L.comp_norm is not None:
            _, L._comp_norm_copies, _ = g.split_kernel(
                batch_split, L.comp_norm, DP)
        _, L._wq_a_copies, _ = g.split_kernel(batch_split, L.wq_a, DP)
        _, L._q_norm_copies, _ = g.split_kernel(batch_split, L.q_norm, DP)
        _, L._wq_b_copies, _ = g.split_kernel(batch_split, L.wq_b, DP)
        _, L._wkv_copies, _ = g.split_kernel(batch_split, L.wkv, DP)
        _, L._kv_norm_copies, _ = g.split_kernel(batch_split, L.kv_norm, DP)
        if L.kv_norm_fan is not None:
            _, L._kv_norm_fan_copies, _ = g.split_kernel(
                batch_split, L.kv_norm_fan, DP)
        if L.comp_norm_fan is not None:
            _, L._comp_norm_fan_copies, _ = g.split_kernel(
                batch_split, L.comp_norm_fan, DP)
        if L.kv_concat is not None:
            _, L._kv_concat_copies, _ = g.split_kernel(
                batch_split, L.kv_concat, DP)
        _, L._sa_copies, _ = g.split_kernel(batch_split, L.sa, DP)
        _, L._wo_a_copies, _ = g.split_kernel(batch_split, L.wo_a, DP)
        _, L._wo_b_copies, _ = g.split_kernel(batch_split, L.wo_b, DP)
        _, L._attn_add_copies, _ = g.split_kernel(batch_split, L.attn_add, DP)
        _, L._ffn_bridge_copies, _ = g.split_kernel(
            batch_split, L.ffn_bridge, DP)
        _, L._ffn_norm_copies, _ = g.split_kernel(batch_split, L.ffn_norm, DP)
        _, L._ffn_fan_copies, _ = g.split_kernel(batch_split, L.ffn_fan, DP)
        _, L._gate_copies, _ = g.split_kernel(batch_split, L.gate, DP)
        _, L._dispatch_copies, _ = g.split_kernel(batch_split, L.dispatch, DP)
        _, L._combine_copies, _ = g.split_kernel(
            batch_split, L.combine, DP)
        _, L._sw_up_copies, _ = g.split_kernel(batch_split, L.sw_up, DP)
        _, L._sw_down_copies, _ = g.split_kernel(batch_split, L.sw_down, DP)
        _, L._moe_add_copies, _ = g.split_kernel(batch_split, L.moe_add, DP)
        _, L._ffn_add_copies, _ = g.split_kernel(batch_split, L.ffn_add, DP)
        if L.kv_acc is not None:
            _, L._kv_acc_copies, _ = g.split_kernel(
                batch_split, L.kv_acc, DP)
        if L.kv_spawn is not None:
            _, L._kv_spawn_copies, _ = g.split_kernel(
                batch_split, L.kv_spawn, DP)

    for L in layers:
        _split_layer(L)

    # Split decode steps
    if decode_steps:
        for step_meta in decode_steps:
            if step_meta.emb is not None:
                _, step_meta._emb_copies, _ = g.split_kernel(
                    batch_split, step_meta.emb, DP)
            _, step_meta._final_norm_copies, _ = g.split_kernel(
                batch_split, step_meta.final_norm, DP)
            _, step_meta._logits_copies, _ = g.split_kernel(
                batch_split, step_meta.logits, DP)
            _, step_meta._sampling_copies, _ = g.split_kernel(
                batch_split, step_meta.sampling, DP)
            for L in step_meta.layers:
                _split_layer(L)

    # Split prefill output head
    if prefill_output_head:
        _pfx_out_copies = []
        for k in prefill_output_head:
            _, copies, _ = g.split_kernel(batch_split, k, DP)
            _pfx_out_copies.append(copies)

    # Split KV cache reads (decode-only)
    if kv_cache_reads:
        _kv_read_copies = []
        for kv_read in kv_cache_reads:
            _, copies, _ = g.split_kernel(batch_split, kv_read, DP)
            _kv_read_copies.append(copies)

    # ── Placement ─────────────────────────────────────────────────
    p = Placement(hardware=hw, graph=g)

    if emb_copies is not None:
        for i, c in enumerate(emb_copies):
            p.set_kernel_device(c, gpus[i])

    if read_input is not None:
        p.set_kernel_device(read_input, gpus[0])
        cpu = [c for c in hw.nodes if isinstance(c, Compute)
               and "intel-xeon" in c.name][0]
        cpu_mem = hw.find_local_memory(cpu)
        p.set_tensor_memory(read_input.inputs["tokens"], cpu_mem)

    def _place_layer(L):
        """Place all DP copies of a layer onto their respective GPUs."""
        always_copies = [L._bridge_copies,
                         L._attn_norm_copies,
                         L._attn_fan_copies, L._wq_a_copies,
                         L._q_norm_copies, L._wq_b_copies,
                         L._wkv_copies, L._kv_norm_copies,
                         L._sa_copies,
                         L._wo_a_copies, L._wo_b_copies,
                         L._attn_add_copies, L._ffn_bridge_copies,
                         L._ffn_norm_copies, L._ffn_fan_copies,
                         L._gate_copies, L._dispatch_copies,
                         L._combine_copies,
                         L._sw_up_copies, L._sw_down_copies,
                         L._moe_add_copies, L._ffn_add_copies]
        if L.kv_concat is not None:
            always_copies.append(L._kv_concat_copies)
        if L.kv_norm_fan is not None:
            always_copies.append(L._kv_norm_fan_copies)
        if L.comp_norm_fan is not None:
            always_copies.append(L._comp_norm_fan_copies)
        if L.kv_acc is not None:
            always_copies.append(L._kv_acc_copies)
        if L.kv_spawn is not None:
            always_copies.append(L._kv_spawn_copies)
        for copies in always_copies:
            for i, c in enumerate(copies):
                p.set_kernel_device(c, gpus[i])
        if L.comp is not None:
            for i, c in enumerate(L._comp_copies):
                p.set_kernel_device(c, gpus[i])
        if L.comp_norm is not None:
            for i, c in enumerate(L._comp_norm_copies):
                p.set_kernel_device(c, gpus[i])

        # Expert kernels → respective GPUs
        for gpu_id, gpu_experts in enumerate(L.experts):
            for k in gpu_experts:
                p.set_kernel_device(k, gpus[gpu_id])

        # Expert input locality: placed on destination GPU's HBM
        for gpu_id, gpu_experts in enumerate(L.experts):
            local_mem = hw.find_local_memory(gpus[gpu_id])
            for eid in range(0, len(gpu_experts), 2):
                up_kernel = gpu_experts[eid]
                p.set_tensor_memory(up_kernel.inputs["x"], local_mem)

        # Dispatch RDMA: each copy writes expert outputs to target GPU's HBM
        for copy in L._dispatch_copies:
            for gpu_id in range(DP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    p.set_tensor_memory(
                        copy.outputs[f"o{global_eid}"], local_mem)

        # Combine RDMA: each copy reads expert outputs from source GPU's HBM
        for copy in L._combine_copies:
            for gpu_id in range(DP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    p.set_tensor_memory(
                        copy.inputs[f"i{global_eid}"], local_mem)

    for L in layers:
        _place_layer(L)

    # Place decode steps
    if decode_steps:
        cpu = [c for c in hw.nodes if isinstance(c, Compute)
               and "intel-xeon" in c.name][0]
        cpu_mem = hw.find_local_memory(cpu)
        for step_meta in decode_steps:
            if step_meta.read_input is not None:
                p.set_kernel_device(step_meta.read_input, gpus[0])
                p.set_tensor_memory(
                    step_meta.read_input.inputs["tokens"], cpu_mem)
            if step_meta.emb is not None:
                for i, c in enumerate(step_meta._emb_copies):
                    p.set_kernel_device(c, gpus[i])
            for i, c in enumerate(step_meta._final_norm_copies):
                p.set_kernel_device(c, gpus[i])
            for i, c in enumerate(step_meta._logits_copies):
                p.set_kernel_device(c, gpus[i])
            for i, c in enumerate(step_meta._sampling_copies):
                p.set_kernel_device(c, gpus[i])
            for L in step_meta.layers:
                _place_layer(L)

    # Place prefill output head
    if prefill_output_head:
        for layer_copies in _pfx_out_copies:
            for i, c in enumerate(layer_copies):
                p.set_kernel_device(c, gpus[i])

    # Place KV cache reads (decode-only)
    if kv_cache_reads:
        cpu = [c for c in hw.nodes if isinstance(c, Compute)
               and "intel-xeon" in c.name][0]
        cpu_mem = hw.find_local_memory(cpu)
        for layer_copies in _kv_read_copies:
            for i, c in enumerate(layer_copies):
                p.set_kernel_device(c, gpus[i])
                p.set_tensor_memory(c.inputs["kv"], cpu_mem)

    optimize_comms(g, p)

    g.validate()
    p.validate(g)
    return g, p


# ═══════════════════════════════════════════════════════════════════════
# C. Simulation Phase
# ═══════════════════════════════════════════════════════════════════════

def simulate(g, p, hw, trace_path="dsv4_pro_prefill.json"):
    """Run DES simulator and export trace."""
    result = Simulator(g, p, hw).run()
    export_trace(result, trace_path)
    return result


# ── Main ────────────────────────────────────────────────────────────────

def _collect_kernels(obj) -> Set[Kernel]:
    """BFS over an object's attributes to find all Kernel instances."""
    from collections import deque
    found = set()
    queue = deque([obj])
    seen_ids = {id(obj)}
    while queue:
        cur = queue.popleft()
        if isinstance(cur, Kernel):
            found.add(cur)
        elif hasattr(cur, "__dict__"):
            for v in vars(cur).values():
                if id(v) not in seen_ids:
                    seen_ids.add(id(v))
                    queue.append(v)
        elif isinstance(cur, (list, tuple)):
            for item in cur:
                if id(item) not in seen_ids:
                    seen_ids.add(id(item))
                    queue.append(item)
    return found


def visualize_layer(g, layer_meta, extra_seeds=None,
                    path="dsv4_pro_graph_layer0.svg"):
    """Visualize a single layer's subgraph."""
    kernels = _collect_kernels(layer_meta)
    if extra_seeds:
        kernels.update(extra_seeds)
    # Include Spawn/Concat identity kernels adjacent to collected kernels
    frozen = frozenset(kernels)
    for k in g.topological_sort():
        if k in frozen:
            continue
        if type(k).__name__ not in ("Spawn", "Concat"):
            continue
        neighbors = set(g._dag.predecessors(k)) | set(g._dag.successors(k))
        if neighbors & frozen:
            kernels.add(k)
    figsize = (max(48, len(kernels) // 8), 32) if len(kernels) > 50 else (24, 16)
    export_graph(g, path, kernels=kernels, figsize=figsize)


def optimize_model_superchip(g, layers, hw, emb=None):
    """Place all kernels on the single fused GPU (no splits, no comms)."""
    gpu = [c for c in hw.nodes if isinstance(c, Compute)
           and "nvidia-b300" in c.name][0]
    p = Placement(hardware=hw, graph=g)
    for k in g.topological_sort():
        p.set_kernel_device(k, gpu)
    g.validate()
    p.validate(g)
    return g, p


def main():
    # A. Declaration
    hw = B300ClusterA(n_nodes=1)
    g, layers, decode_steps, emb, read_input, kv_cache_reads, pfx_out_head = \
        declare_model()

    visualize_layer(g, layers[0], extra_seeds={emb, read_input})

    # B. Optimization
    g, p = optimize_model(g, layers, hw, emb, read_input,
                          decode_steps, kv_cache_reads, pfx_out_head)

    # C. Simulation
    result = simulate(g, p, hw, "dsv4_pro_prefill.json")
    print(f"Prefill: {result.total_time_us:.1f} us "
          f"({result.total_time_us / 1000:.1f} ms)")

    # ── SuperChip (zero-comm) comparison ──
    hw_sc = B300SuperChipA()
    g_sc, layers_sc, _, emb_sc, _, _, _ = declare_model()

    # B. Optimization (no splits)
    g_sc, p_sc = optimize_model_superchip(g_sc, layers_sc, hw_sc, emb_sc)

    # C. Simulation
    result_sc = simulate(g_sc, p_sc, hw_sc, "dsv4_pro_superchip.json")
    print(f"Prefill (SuperChip): {result_sc.total_time_us:.1f} us "
          f"({result_sc.total_time_us / 1000:.1f} ms)")


if __name__ == "__main__":
    main()
