"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 TP=8.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model() — split_kernel for TP, control edges, placement
  C. simulate() — DES execution + trace export
"""

from dataclasses import dataclass, field
from typing import List, Set

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.forward import (
    Embedding, Gemm, ReadInput, RMSNorm, SparseAttn, StridedGemm,
    TokenCombine, TokenDispatch,
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
TP = 8
EP = 8
N_LOCAL_EXPERTS = N_EXPERTS // EP
V = 129280
BATCH = 8
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
    kv_concat: Kernel = None
    sa: Kernel = None
    wo_a: Kernel = None
    wo_b: Kernel = None
    ffn_norm: Kernel = None
    ffn_fan: Kernel = None
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
    S = S_PREFILL
    M = BATCH * S  # total tokens (all Gemm/Norm/MoE use M)
    layers = []

    # ReadInput: host-to-device transfer of token indices
    read_input = ReadInput(BATCH * S, "int32")
    read_input.inputs = {"tokens": Tensor("int32", (BATCH, S, 1))}
    read_input.outputs = {"tokens": Tensor("int32", (BATCH, S, 1))}
    g.add_kernel(read_input)

    # Embedding lookup
    emb = Embedding(M, V, D)
    emb.inputs = {"idx": Tensor("int32", (BATCH, S, 1))}
    emb.weights = {"emb": Tensor("bf16", (M, D))}
    emb.outputs = {"y": Tensor("bf16", (BATCH, S, D))}
    g.add_kernel(emb)
    g.add_data_edge(read_input, emb, {"tokens": "idx"})

    prev_out = emb

    for layer_id in range(N_LAYERS):
        ratio = COMPRESS_RATIOS[layer_id]
        L = LayerMeta()

        # ── Attention ─────────────────────────────────────────────
        attn_norm = make_norm(BATCH, S, D)
        g.add_kernel(attn_norm)
        L.attn_norm = attn_norm

        # Fan-out after norm: Q path + KV path
        attn_fan = Spawn(world=2)
        attn_fan.inputs = {"x": Tensor("bf16", (BATCH, S, D))}
        attn_fan.outputs = {"y": Tensor("bf16", (BATCH, S, D)),
                            "y2": Tensor("bf16", (BATCH, S, D))}
        g.add_kernel(attn_fan)
        g.add_data_edge(attn_norm, attn_fan, {"y": "x"})
        L.attn_fan = attn_fan

        # Residual fan-out: bridge feeds both attn_norm and comp
        bridge = Spawn(world=2)
        bridge.inputs = {"x": Tensor("bf16", (BATCH, S, D))}
        bridge.outputs = {"y": Tensor("bf16", (BATCH, S, D)),
                          "y2": Tensor("bf16", (BATCH, S, D))}
        g.add_kernel(bridge)
        g.add_data_edge(prev_out, bridge, {"y": "x"})
        g.add_data_edge(bridge, attn_norm, {"y": "x"})
        L.bridge = bridge

        # Q path
        wq_a = make_gemm(BATCH, S, Q_LORA, D, "fp8")
        g.add_kernel(wq_a)
        g.add_data_edge(attn_fan, wq_a, {"y": "x"})
        L.wq_a = wq_a

        q_norm = make_norm(BATCH, S, Q_LORA)
        g.add_kernel(q_norm)
        g.add_data_edge(wq_a, q_norm, {"y": "x"})
        L.q_norm = q_norm

        wq_b = make_gemm(BATCH, S, H * HD, Q_LORA, "fp8")
        g.add_kernel(wq_b)
        g.add_data_edge(q_norm, wq_b, {"y": "x"})
        L.wq_b = wq_b

        # KV path (branch from attn fan-out)
        wkv = make_gemm(BATCH, S, KV_DIM, D, "fp8")
        g.add_kernel(wkv)
        g.add_data_edge(attn_fan, wkv, {"y2": "x"})
        L.wkv = wkv

        kv_norm = make_norm(BATCH, S, KV_DIM)
        g.add_kernel(kv_norm)
        g.add_data_edge(wkv, kv_norm, {"y": "x"})
        L.kv_norm = kv_norm

        # Compressor (reads from residual via bridge, or root at layer 0)
        if ratio in (128, 4):
            coff = 1 if ratio == 128 else 2
            S_comp = S // ratio  # compressed seq len per sequence
            M_comp = BATCH * S_comp  # total compressed tokens
            comp_out_elems = M_comp * KV_DIM
            comp = StridedGemm(M, KV_DIM * coff, D, "fp32", "bf16",
                               in_elems=M * D, out_elems=comp_out_elems)
            comp.inputs = {"x": Tensor("bf16", (BATCH, S, D))}
            comp.weights = {"w": Tensor("fp32", (D, KV_DIM * coff))}
            comp.outputs = {"y": Tensor("bf16", (BATCH, S_comp, KV_DIM))}
            g.add_kernel(comp)
            if L.bridge is not None:
                g.add_data_edge(L.bridge, comp, {"y2": "x"})
            L.comp = comp

            comp_norm = make_norm(BATCH, S_comp, KV_DIM)
            g.add_kernel(comp_norm)
            g.add_data_edge(comp, comp_norm, {"y": "x"})
            L.comp_norm = comp_norm

        # Sparse attention (per-sequence dimensions)
        k_sel = WINDOW
        if ratio == 128:
            k_sel = WINDOW + S // 128
        elif ratio == 4:
            k_sel = WINDOW + INDEX_TOPK

        S_kv = S + S // ratio
        sa = SparseAttn(BATCH, H, 1, S, k_sel, S_kv, HD, "bf16", kv_factor=1)
        sa.inputs = {"q": Tensor("bf16", (BATCH, S, H * HD)),
                     "kv": Tensor("bf16", (BATCH, S_kv, HD))}
        sa.outputs = {"y": Tensor("bf16", (BATCH, S, H * HD))}
        g.add_kernel(sa)
        g.add_data_edge(wq_b, sa, {"y": "q"})

        # KV cache = concat(window_kv, compressed_kv)
        kv_concat = Concat()
        kv_concat.inputs = {"a": Tensor("bf16", (BATCH, S, KV_DIM)),
                            "b": Tensor("bf16", (BATCH, S_comp, KV_DIM))}
        kv_concat.outputs = {"y": Tensor("bf16", (BATCH, S_kv, KV_DIM))}
        g.add_kernel(kv_concat)
        g.add_data_edge(kv_norm, kv_concat, {"y": "a"})
        g.add_data_edge(comp_norm, kv_concat, {"y": "b"})
        g.add_data_edge(kv_concat, sa, {"y": "kv"})
        L.kv_concat = kv_concat
        L.sa = sa

        # Output projection (grouped linear: O_GROUPS independent Gemms)
        wo_a = StridedGemm(M, O_GROUPS * O_LORA, H * HD // O_GROUPS,
                           "bf16", "bf16", in_elems=M * H * HD)
        wo_a.inputs = {"x": Tensor("bf16", (BATCH, S, H * HD))}
        wo_a.weights = {"w": Tensor("bf16",
                        (H * HD // O_GROUPS, O_GROUPS * O_LORA))}
        wo_a.outputs = {"y": Tensor("bf16", (BATCH, S, O_GROUPS * O_LORA))}
        g.add_kernel(wo_a)
        g.add_data_edge(sa, wo_a, {"y": "x"})
        L.wo_a = wo_a

        wo_b = make_gemm(BATCH, S, D, O_GROUPS * O_LORA, "fp8")
        g.add_kernel(wo_b)
        g.add_data_edge(wo_a, wo_b, {"y": "x"})
        L.wo_b = wo_b

        # ── FFN / MoE ─────────────────────────────────────────────
        ffn_norm = make_norm(BATCH, S, D)
        g.add_kernel(ffn_norm)
        g.add_data_edge(wo_b, ffn_norm, {"y": "x"})
        L.ffn_norm = ffn_norm

        # FFN fan-out: gate needs routing scores, dispatch needs hidden states
        ffn_fan = Spawn(world=2)
        ffn_fan.inputs = {"x": Tensor("bf16", (BATCH, S, D))}
        ffn_fan.outputs = {"y": Tensor("bf16", (BATCH, S, D)),
                           "y2": Tensor("bf16", (BATCH, S, D))}
        g.add_kernel(ffn_fan)
        g.add_data_edge(ffn_norm, ffn_fan, {"y": "x"})
        L.ffn_fan = ffn_fan

        gate = make_gemm(BATCH, S, N_EXPERTS, D, "bf16", "bf16", "fp32")
        g.add_kernel(gate)
        g.add_data_edge(ffn_fan, gate, {"y": "x"})
        L.gate = gate

        # Dispatch: softmax routing + token scatter to experts
        S_e = S * TOPK // N_EXPERTS
        M_e = BATCH * S_e
        dispatch = TokenDispatch(M, D, N_EXPERTS, TOPK)
        dispatch.inputs = {"x": Tensor("bf16", (BATCH, S, D)),
                           "routing": Tensor("fp32", (BATCH, S, N_EXPERTS))}
        dispatch.outputs = {f"o{i}": Tensor("bf16", (BATCH, S_e, D))
                            for i in range(N_EXPERTS)}
        g.add_kernel(dispatch)
        g.add_data_edge(gate, dispatch, {"y": "routing"})
        g.add_data_edge(ffn_fan, dispatch, {"y2": "x"})
        L.dispatch = dispatch

        # Combine: weighted sum of expert outputs
        combine = TokenCombine(M, D, N_EXPERTS, TOPK)
        combine.inputs = {f"i{i}": Tensor("bf16", (BATCH, S_e, D))
                          for i in range(N_EXPERTS)}
        combine.outputs = {"y": Tensor("bf16", (BATCH, S, D))}
        g.add_kernel(combine)
        L.combine = combine

        # Expert kernels per GPU (up_proj + down_proj per expert, independent)
        L.experts = []
        for gpu_id in range(TP):
            gpu_experts = []
            for eid in range(N_LOCAL_EXPERTS):
                global_eid = gpu_id * N_LOCAL_EXPERTS + eid
                up = make_gated_up(BATCH, S_e, MOE_INTER, D, "fp4", "bf16")
                g.add_kernel(up)

                down = make_gemm(BATCH, S_e, D, MOE_INTER, "fp4", "bf16")
                g.add_kernel(down)
                g.add_data_edge(up, down, {"y": "x"})
                g.add_data_edge(dispatch, up, {f"o{global_eid}": "x"})
                g.add_data_edge(down, combine, {"y": f"i{global_eid}"})

                gpu_experts.extend([up, down])

            L.experts.append(gpu_experts)

        # Shared expert
        sw_up = make_gated_up(BATCH, S, MOE_INTER, D, "fp8", "bf16")
        g.add_kernel(sw_up)
        g.add_data_edge(combine, sw_up, {"y": "x"})
        L.sw_up = sw_up

        sw_down = make_gemm(BATCH, S, D, MOE_INTER, "fp8", "bf16")
        g.add_kernel(sw_down)
        g.add_data_edge(sw_up, sw_down, {"y": "x"})
        L.sw_down = sw_down

        prev_out = sw_down
        layers.append(L)

    g.validate()
    return g, layers, emb, read_input


# ═══════════════════════════════════════════════════════════════════════
# B. Optimization Phase
# ═══════════════════════════════════════════════════════════════════════

def optimize_model(g, layers, hw, emb=None, read_input=None):
    """Apply split_kernel for TP, add control edges, and place."""
    gpus = sorted(
        [c for c in hw.nodes if isinstance(c, Compute)
         and "nvidia-b300" in c.name],
        key=lambda c: c.name)

    # ── Phase 1: DP splits (batch dim) ───────────────────────────────
    emb_copies = None
    if emb is not None:
        _, emb_copies, _ = g.split_kernel(batch_split, emb, TP)

    for L in layers:
        _, L._bridge_copies, _ = g.split_kernel(batch_split, L.bridge, TP)
        _, L._attn_norm_copies, _ = g.split_kernel(batch_split, L.attn_norm, TP)
        _, L._attn_fan_copies, _ = g.split_kernel(batch_split, L.attn_fan, TP)
        if L.comp is not None:
            _, L._comp_copies, _ = g.split_kernel(batch_split, L.comp, TP)
        if L.comp_norm is not None:
            _, L._comp_norm_copies, _ = g.split_kernel(
                batch_split, L.comp_norm, TP)
        _, L._wq_a_copies, _ = g.split_kernel(batch_split, L.wq_a, TP)
        _, L._q_norm_copies, _ = g.split_kernel(batch_split, L.q_norm, TP)
        _, L._wq_b_copies, _ = g.split_kernel(batch_split, L.wq_b, TP)
        _, L._wkv_copies, _ = g.split_kernel(batch_split, L.wkv, TP)
        _, L._kv_norm_copies, _ = g.split_kernel(batch_split, L.kv_norm, TP)
        _, L._kv_concat_copies, _ = g.split_kernel(
            batch_split, L.kv_concat, TP)
        _, L._sa_copies, _ = g.split_kernel(batch_split, L.sa, TP)
        _, L._wo_a_copies, _ = g.split_kernel(batch_split, L.wo_a, TP)
        _, L._wo_b_copies, _ = g.split_kernel(batch_split, L.wo_b, TP)
        _, L._ffn_norm_copies, _ = g.split_kernel(batch_split, L.ffn_norm, TP)
        _, L._ffn_fan_copies, _ = g.split_kernel(batch_split, L.ffn_fan, TP)
        _, L._gate_copies, _ = g.split_kernel(batch_split, L.gate, TP)
        _, L._dispatch_copies, _ = g.split_kernel(batch_split, L.dispatch, TP)
        _, L._combine_copies, _ = g.split_kernel(
            batch_split, L.combine, TP)
        _, L._sw_up_copies, _ = g.split_kernel(batch_split, L.sw_up, TP)
        _, L._sw_down_copies, _ = g.split_kernel(batch_split, L.sw_down, TP)

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

    for L in layers:
        # DP copies → GPU0-7
        for copies in [L._bridge_copies, L._attn_norm_copies,
                       L._attn_fan_copies, L._wq_a_copies,
                       L._q_norm_copies, L._wq_b_copies,
                       L._wkv_copies, L._kv_norm_copies,
                       L._kv_concat_copies, L._sa_copies,
                       L._wo_a_copies, L._wo_b_copies,
                       L._ffn_norm_copies, L._ffn_fan_copies,
                       L._gate_copies, L._dispatch_copies,
                       L._combine_copies,
                       L._sw_up_copies, L._sw_down_copies]:
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
            for gpu_id in range(TP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    p.set_tensor_memory(
                        copy.outputs[f"o{global_eid}"], local_mem)

        # Combine RDMA: each copy reads expert outputs from source GPU's HBM
        for copy in L._combine_copies:
            for gpu_id in range(TP):
                local_mem = hw.find_local_memory(gpus[gpu_id])
                for local_eid in range(N_LOCAL_EXPERTS):
                    global_eid = gpu_id * N_LOCAL_EXPERTS + local_eid
                    p.set_tensor_memory(
                        copy.inputs[f"i{global_eid}"], local_mem)

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
    g, layers, emb, read_input = declare_model()

    visualize_layer(g, layers[0], extra_seeds={emb, read_input})

    # B. Optimization
    g, p = optimize_model(g, layers, hw, emb, read_input)

    # C. Simulation
    result = simulate(g, p, hw, "dsv4_pro_prefill.json")
    print(f"Prefill: {result.total_time_us:.1f} us "
          f"({result.total_time_us / 1000:.1f} ms)")

    # ── SuperChip (zero-comm) comparison ──
    hw_sc = B300SuperChipA()
    g_sc, layers_sc, emb_sc, read_input_sc = declare_model()

    # B. Optimization (no splits)
    g_sc, p_sc = optimize_model_superchip(g_sc, layers_sc, hw_sc, emb_sc)

    # C. Simulation
    result_sc = simulate(g_sc, p_sc, hw_sc, "dsv4_pro_superchip.json")
    print(f"Prefill (SuperChip): {result_sc.total_time_us:.1f} us "
          f"({result_sc.total_time_us / 1000:.1f} ms)")


if __name__ == "__main__":
    main()
