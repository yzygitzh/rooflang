"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 TP=8.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model() — split_kernel for TP, control edges, placement
  C. simulate() — DES execution + trace export
"""

from dataclasses import dataclass, field
from typing import List

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.forward import Gemm, RMSNorm, SparseAttn
from rooflang.language.kernels.identity import Spawn
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.optimization.split import column_split, head_split, row_split
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.language.utils import dtype_bytes
from rooflang.programs.presets.b300 import B300ClusterA
from rooflang.runtime.simulator import Simulator
from rooflang.runtime.trace_export import export_trace

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
TP = 8
EP = 8
N_LOCAL_EXPERTS = N_EXPERTS // EP
S_PREFILL = 8192
COMPRESS_RATIOS = [128, 128] + [v for _ in range(29) for v in (4, 128)] + [4]


# ── Kernel factories ────────────────────────────────────────────────────

def make_gemm(M, N, K, w_dtype, a_dtype="bf16", out_dtype="bf16"):
    k = Gemm(M, N, K, w_dtype, a_dtype, out_dtype)
    k.inputs = {"x": Tensor("bf16", (M * K,))}
    k.weights = {"w": Tensor(w_dtype, (K * N,))}
    k.outputs = {"y": Tensor("bf16", (M * N,))}
    return k


def make_norm(M, dim):
    k = RMSNorm(M, dim, "bf16")
    k.inputs = {"x": Tensor("bf16", (M * dim,))}
    k.weights = {"g": Tensor("bf16", (dim,))}
    k.outputs = {"y": Tensor("bf16", (M * dim,))}
    return k



# ── Per-layer metadata for optimization phase ───────────────────────────

@dataclass
class LayerMeta:
    bridge: Kernel = None
    attn_norm: Kernel = None
    wq_a: Kernel = None
    q_norm: Kernel = None
    wq_b: Kernel = None
    wkv: Kernel = None
    kv_norm: Kernel = None
    comp: Kernel = None
    comp_norm: Kernel = None
    sa: Kernel = None
    wo_a: Kernel = None
    wo_b: Kernel = None
    ffn_norm: Kernel = None
    gate: Kernel = None
    dispatch: Kernel = None
    combine: Kernel = None
    sw_up: Kernel = None
    sw_down: Kernel = None
    experts: List[List[Kernel]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# A. Declaration Phase
# ═══════════════════════════════════════════════════════════════════════

def declare_model():
    """Build logical compute graph using only add_kernel and add_data_edge."""
    g = ComputeGraph()
    M = S_PREFILL
    layers = []

    prev_out = None

    for layer_id in range(N_LAYERS):
        ratio = COMPRESS_RATIOS[layer_id]
        L = LayerMeta()

        # ── Attention ─────────────────────────────────────────────
        attn_norm = make_norm(M, D)
        attn_norm.outputs = {"y": Tensor("bf16", (M * D,)),
                             "y2": Tensor("bf16", (M * D,))}
        g.add_kernel(attn_norm)
        L.attn_norm = attn_norm

        # Residual fan-out: bridge feeds both attn_norm and comp
        if prev_out is not None:
            bridge = Spawn(world=2)
            bridge.inputs = {"x": Tensor("bf16", (M * D,))}
            bridge.outputs = {"y": Tensor("bf16", (M * D,)),
                              "y2": Tensor("bf16", (M * D,))}
            g.add_kernel(bridge)
            g.add_data_edge(prev_out, bridge, {"y": "x"})
            g.add_data_edge(bridge, attn_norm, {"y": "x"})
            L.bridge = bridge

        # Q path
        wq_a = make_gemm(M, Q_LORA, D, "fp8")
        g.add_kernel(wq_a)
        g.add_data_edge(attn_norm, wq_a, {"y": "x"})
        L.wq_a = wq_a

        q_norm = make_norm(M, Q_LORA)
        g.add_kernel(q_norm)
        g.add_data_edge(wq_a, q_norm, {"y": "x"})
        L.q_norm = q_norm

        wq_b = make_gemm(M, H * HD, Q_LORA, "fp8")
        g.add_kernel(wq_b)
        g.add_data_edge(q_norm, wq_b, {"y": "x"})
        L.wq_b = wq_b

        # KV path (branch from attn_norm)
        wkv = make_gemm(M, KV_DIM, D, "fp8")
        g.add_kernel(wkv)
        g.add_data_edge(attn_norm, wkv, {"y2": "x"})
        L.wkv = wkv

        kv_norm = make_norm(M, KV_DIM)
        g.add_kernel(kv_norm)
        g.add_data_edge(wkv, kv_norm, {"y": "x"})
        L.kv_norm = kv_norm

        # Compressor (reads from residual via bridge, or root at layer 0)
        if ratio in (128, 4):
            coff = 1 if ratio == 128 else 2
            comp = make_gemm(M, KV_DIM * coff, D, "fp32", "bf16")
            comp.outputs = {"y": Tensor("bf16", (M // ratio * KV_DIM,))}
            g.add_kernel(comp)
            if L.bridge is not None:
                g.add_data_edge(L.bridge, comp, {"y2": "x"})
            L.comp = comp

            comp_norm = make_norm(M // ratio, KV_DIM)
            g.add_kernel(comp_norm)
            g.add_data_edge(comp, comp_norm, {"y": "x"})
            L.comp_norm = comp_norm

        # Sparse attention
        k_sel = WINDOW
        if ratio == 128:
            k_sel = WINDOW + M // 128
        elif ratio == 4:
            k_sel = WINDOW + INDEX_TOPK

        sa = SparseAttn(1, H, 1, M, k_sel, HD)
        sa.inputs = {"x": Tensor("bf16", (M * H * HD,))}
        sa.outputs = {"y": Tensor("bf16", (M * H * HD,))}
        g.add_kernel(sa)
        g.add_data_edge(wq_b, sa, {"y": "x"})
        L.sa = sa

        # Output projection
        wo_a = Gemm(M, O_GROUPS * O_LORA, H * HD // O_GROUPS, "bf16", "bf16")
        wo_a.inputs = {"x": Tensor("bf16", (M * H * HD,))}
        wo_a.weights = {"w": Tensor("bf16",
                        (H * HD // O_GROUPS * O_GROUPS * O_LORA,))}
        wo_a.outputs = {"y": Tensor("bf16", (M * O_GROUPS * O_LORA,))}
        g.add_kernel(wo_a)
        g.add_data_edge(sa, wo_a, {"y": "x"})
        L.wo_a = wo_a

        wo_b = make_gemm(M, D, O_GROUPS * O_LORA, "fp8")
        g.add_kernel(wo_b)
        g.add_data_edge(wo_a, wo_b, {"y": "x"})
        L.wo_b = wo_b

        # ── FFN / MoE ─────────────────────────────────────────────
        ffn_norm = make_norm(M, D)
        g.add_kernel(ffn_norm)
        g.add_data_edge(wo_b, ffn_norm, {"y": "x"})
        L.ffn_norm = ffn_norm

        gate = make_gemm(M, N_EXPERTS, D, "fp32", "bf16")
        gate.outputs = {"y": Tensor("fp32", (M * N_EXPERTS,))}
        g.add_kernel(gate)
        g.add_data_edge(ffn_norm, gate, {"y": "x"})
        L.gate = gate

        # Dispatch bridge
        M_e = M * TOPK // N_LOCAL_EXPERTS
        dispatch = Kernel(
            inputs={"routing": Tensor("fp32", (M * N_EXPERTS,))},
            outputs={f"o{i}": Tensor("bf16", (M_e * D,)) for i in range(TP)})
        g.add_kernel(dispatch)
        g.add_data_edge(gate, dispatch, {"y": "routing"})
        L.dispatch = dispatch

        # Combine bridge
        combine = Kernel(
            inputs={f"i{i}": Tensor("bf16", (M_e * D,)) for i in range(TP)},
            outputs={"y": Tensor("bf16", (M * D,))})
        g.add_kernel(combine)
        L.combine = combine

        # Expert chains per GPU (up_proj + down_proj per expert, chained)
        L.experts = []
        for gpu_id in range(TP):
            gpu_experts = []
            exp_prev = None
            for eid in range(N_LOCAL_EXPERTS):
                up = Gemm(M_e, 2 * MOE_INTER, D, "fp4", "fp8")
                up.inputs = {"x": Tensor("bf16", (M_e * D,))}
                up.weights = {"w": Tensor("fp4", (2 * D * MOE_INTER,))}
                up.outputs = {"y": Tensor("bf16", (M_e * MOE_INTER,))}
                g.add_kernel(up)

                down = Gemm(M_e, D, MOE_INTER, "fp4", "fp8")
                down.inputs = {"x": Tensor("bf16", (M_e * MOE_INTER,))}
                down.weights = {"w": Tensor("fp4", (MOE_INTER * D,))}
                down.outputs = {"y": Tensor("bf16", (M_e * D,))}
                g.add_kernel(down)
                g.add_data_edge(up, down, {"y": "x"})

                if eid == 0:
                    g.add_data_edge(dispatch, up, {f"o{gpu_id}": "x"})
                else:
                    g.add_data_edge(exp_prev, up, {"y": "x"})

                gpu_experts.extend([up, down])
                exp_prev = down

            g.add_data_edge(exp_prev, combine, {"y": f"i{gpu_id}"})
            L.experts.append(gpu_experts)

        # Shared expert
        sw_up = Gemm(M, 2 * MOE_INTER, D, "fp8", "fp8")
        sw_up.inputs = {"x": Tensor("bf16", (M * D,))}
        sw_up.weights = {"w": Tensor("fp8", (2 * D * MOE_INTER,))}
        sw_up.outputs = {"y": Tensor("bf16", (M * MOE_INTER,))}
        g.add_kernel(sw_up)
        g.add_data_edge(combine, sw_up, {"y": "x"})
        L.sw_up = sw_up

        sw_down = Gemm(M, D, MOE_INTER, "fp8", "fp8")
        sw_down.inputs = {"x": Tensor("bf16", (M * MOE_INTER,))}
        sw_down.weights = {"w": Tensor("fp8", (MOE_INTER * D,))}
        sw_down.outputs = {"y": Tensor("bf16", (M * D,))}
        g.add_kernel(sw_down)
        g.add_data_edge(sw_up, sw_down, {"y": "x"})
        L.sw_down = sw_down

        prev_out = sw_down
        layers.append(L)

    g.validate()
    return g, layers


# ═══════════════════════════════════════════════════════════════════════
# B. Optimization Phase
# ═══════════════════════════════════════════════════════════════════════

def optimize_model(g, layers, hw):
    """Apply split_kernel for TP, add control edges, and place."""
    gpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and "nvidia-b300" in c.name],
        key=lambda c: c.name)

    # ── Graph transforms ──────────────────────────────────────────
    for L in layers:
        # TP splits
        _, wq_b_copies, _ = g.split_kernel(column_split, L.wq_b, TP)
        sa_prev, sa_copies, _ = g.split_kernel(head_split, L.sa, TP)
        _, wo_a_copies, _ = g.split_kernel(column_split, L.wo_a, TP)
        _, wo_b_copies, _ = g.split_kernel(row_split, L.wo_b, TP)

        # KV path ordering: attention waits for KV computation
        kv_end = L.comp_norm if L.comp_norm else L.kv_norm
        g.add_control_edge(kv_end, sa_prev)

        # Store copies for placement
        L._wq_b_copies = wq_b_copies
        L._sa_copies = sa_copies
        L._wo_a_copies = wo_a_copies
        L._wo_b_copies = wo_b_copies

    g.validate()

    # ── Placement ─────────────────────────────────────────────────
    p = Placement(hardware=hw, graph=g)

    for L in layers:
        # Non-split kernels → GPU0 (place in topological order)
        for k in [L.attn_norm, L.wq_a, L.q_norm, L.wkv, L.kv_norm,
                  L.ffn_norm, L.gate, L.dispatch]:
            p.set_kernel_device(k, gpus[0])
        if L.comp is not None:
            p.set_kernel_device(L.comp, gpus[0])
        if L.comp_norm is not None:
            p.set_kernel_device(L.comp_norm, gpus[0])

        # TP copies → GPU0-7
        for copies in [L._wq_b_copies, L._sa_copies,
                       L._wo_a_copies, L._wo_b_copies]:
            for i, c in enumerate(copies):
                p.set_kernel_device(c, gpus[i])

        # Expert kernels → respective GPUs (before combine)
        for gpu_id, gpu_experts in enumerate(L.experts):
            for k in gpu_experts:
                p.set_kernel_device(k, gpus[gpu_id])

        # Post-expert kernels → GPU0
        for k in [L.combine, L.sw_up, L.sw_down]:
            p.set_kernel_device(k, gpus[0])

    p.validate(g)
    return g, p


# ═══════════════════════════════════════════════════════════════════════
# C. Simulation Phase
# ═══════════════════════════════════════════════════════════════════════

def simulate(g, p, hw):
    """Run DES simulator and export trace."""
    result = Simulator(g, p, hw).run()
    export_trace(result, "dsv4_pro_prefill.json")
    return result


# ── Main ────────────────────────────────────────────────────────────────

def main():
    # A. Declaration
    hw = B300ClusterA(n_nodes=1)
    g, layers = declare_model()

    # B. Optimization
    g, p = optimize_model(g, layers, hw)

    # C. Simulation
    result = simulate(g, p, hw)
    print(f"Prefill: {result.total_time_us:.1f} us "
          f"({result.total_time_us / 1000:.1f} ms)")


if __name__ == "__main__":
    main()
