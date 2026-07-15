"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 TP=8.

Parallelism is expressed through data dependencies, kernel copies, and device
placement. No explicit CommKernels — cross-device data edges naturally model
communication at NVLink bandwidth via the placement memory-tracking mechanism.

Chained TP splits (wq_b → sparse_attn → wo_a → wo_b) keep data local on each
GPU (control edges between same-GPU copies). Only the final row-split gather
(after wo_b) incurs cross-device reads, modeled via data edges from remote
copies to the gather kernel on GPU0.
"""

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import Gemm, RMSNorm, SparseAttn
from rooflang.language.kernels.kernel import Kernel
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


# ── Helpers ─────────────────────────────────────────────────────────────

def make_gemm(M, N, K, w_dtype, a_dtype, out_dtype="bf16"):
    k = Gemm(M, N, K, w_dtype, a_dtype, out_dtype)
    k.inputs = {"x": Tensor(a_dtype, (M * K,))}
    k.weights = {"w": Tensor(w_dtype, (K * N,))}
    return k


def make_norm(M, dim, dtype="bf16"):
    k = RMSNorm(M, dim, dtype)
    k.inputs = {"x": Tensor(dtype, (M * dim,))}
    k.weights = {"g": Tensor(dtype, (dim,))}
    return k


# ── Build prefill graph ─────────────────────────────────────────────────

def build_prefill(hw, gpus):
    g = ComputeGraph()
    p = Placement(hardware=hw, graph=g)
    M = S_PREFILL

    def place(kernel, gpu=gpus[0]):
        g.add_kernel(kernel)
        p.set_kernel_device(kernel, gpu)
        return kernel

    prev = None
    for layer_id in range(N_LAYERS):
        ratio = COMPRESS_RATIOS[layer_id]

        # ── Attention (GPU0 sequential) ────────────────────────────
        attn_norm = place(make_norm(M, D))
        wq_a = place(make_gemm(M, Q_LORA, D, "fp8", "fp8"))
        q_norm = place(make_norm(M, Q_LORA))
        wkv = place(make_gemm(M, KV_DIM, D, "fp8", "fp8"))
        kv_norm = place(make_norm(M, KV_DIM))

        seq = [attn_norm, wq_a, q_norm]
        if prev is not None:
            seq = [prev] + seq
        for i in range(len(seq) - 1):
            g.add_control_edge(seq[i], seq[i + 1])

        # ── TP copies (chained locally per GPU) ────────────────────
        # wq_b: column split on N
        shard_n_wqb = H * HD // TP
        wq_b_copies = []
        for i in range(TP):
            c = Gemm(M, shard_n_wqb, Q_LORA, "fp8", "fp8")
            c.inputs = {"x": Tensor("fp8", (M * Q_LORA,))}
            c.weights = {"w": Tensor("fp8", (Q_LORA * shard_n_wqb,))}
            wq_b_copies.append(place(c, gpus[i]))
            g.add_control_edge(q_norm, c)

        # KV path (sequential after wq_b for ordering)
        g.add_control_edge(wq_b_copies[0], wkv)
        g.add_control_edge(wkv, kv_norm)

        # Compressor
        if ratio in (128, 4):
            coff = 1 if ratio == 128 else 2
            comp = place(make_gemm(M, KV_DIM * coff, D, "fp32", "bf16", "bf16"))
            comp_norm = place(make_norm(M // ratio, KV_DIM))
            g.add_control_edge(kv_norm, comp)
            g.add_control_edge(comp, comp_norm)
            sa_after = comp_norm
            k_sel = WINDOW + (M // 128 if ratio == 128 else INDEX_TOPK)
        else:
            sa_after = kv_norm
            k_sel = WINDOW

        # Sparse attention: head split
        shard_h = H // TP
        in_elems = (1 * shard_h * M * HD
                    + 2 * 1 * 1 * M * k_sel * HD)
        out_elems = 1 * shard_h * M * HD
        sa_copies = []
        for i in range(TP):
            c = SparseAttn(1, shard_h, 1, M, k_sel, HD)
            c.inputs = {"x": Tensor("bf16", (in_elems,))}
            sa_copies.append(place(c, gpus[i]))
            g.add_control_edge(wq_b_copies[i], c)
            g.add_control_edge(sa_after, c)

        # wo_a: column split
        wo_a_K = H * HD // O_GROUPS
        wo_a_N = O_GROUPS * O_LORA
        shard_n_woa = wo_a_N // TP
        wo_a_copies = []
        for i in range(TP):
            c = Gemm(M, shard_n_woa, wo_a_K, "bf16", "bf16")
            c.inputs = {"x": Tensor("bf16", (M * wo_a_K,))}
            c.weights = {"w": Tensor("bf16", (wo_a_K * shard_n_woa,))}
            wo_a_copies.append(place(c, gpus[i]))
            g.add_control_edge(sa_copies[i], c)

        # wo_b: row split — gather at the end via data edges
        wo_b_K = O_GROUPS * O_LORA
        shard_k_wob = wo_b_K // TP
        wo_b_copies = []
        for i in range(TP):
            c = Gemm(M, D, shard_k_wob, "fp8", "fp8")
            c.inputs = {"x": Tensor("fp8", (M * shard_k_wob,))}
            c.weights = {"w": Tensor("fp8", (shard_k_wob * D,))}
            c.outputs = {"y": Tensor("bf16", (M * D,))}
            wo_b_copies.append(place(c, gpus[i]))
            g.add_control_edge(wo_a_copies[i], c)

        # Gather after wo_b: reads from all GPUs → models AllReduce cost
        wo_b_gather = Kernel(
            inputs={f"i{i}": Tensor("bf16", (M * D,)) for i in range(TP)})
        place(wo_b_gather)
        for i in range(TP):
            g.add_data_edge(wo_b_copies[i], wo_b_gather, {"y": f"i{i}"})

        # ── FFN / MoE ──────────────────────────────────────────────
        ffn_norm = place(make_norm(M, D))
        gate = place(make_gemm(M, N_EXPERTS, D, "fp32", "bf16", "fp32"))
        g.add_control_edge(wo_b_gather, ffn_norm)
        g.add_control_edge(ffn_norm, gate)

        # Dispatch bridge: scatters token batches to each GPU
        M_e = M * TOPK // N_LOCAL_EXPERTS
        dispatch = Kernel(
            outputs={f"o{i}": Tensor("fp8", (M_e * D,)) for i in range(TP)})
        place(dispatch)
        g.add_control_edge(gate, dispatch)

        # Combine bridge: gathers expert results from all GPUs
        combine = Kernel(
            inputs={f"i{i}": Tensor("bf16", (M_e * D,)) for i in range(TP)},
            outputs={"y": Tensor("fp8", (M * D,))})
        place(combine)

        # Experts: 48 per GPU × 8 GPUs
        for gpu_id in range(TP):
            exp_prev = None
            for eid in range(N_LOCAL_EXPERTS):
                w1 = place(make_gemm(M_e, MOE_INTER, D, "fp4", "fp8"),
                           gpus[gpu_id])
                w3 = place(make_gemm(M_e, MOE_INTER, D, "fp4", "fp8"),
                           gpus[gpu_id])
                w2 = make_gemm(M_e, D, MOE_INTER, "fp4", "fp8")
                if eid == N_LOCAL_EXPERTS - 1:
                    w2.outputs = {"y": Tensor("bf16", (M_e * D,))}
                place(w2, gpus[gpu_id])
                if eid == 0:
                    g.add_data_edge(dispatch, w1, {f"o{gpu_id}": "x"})
                else:
                    g.add_control_edge(exp_prev, w1)
                g.add_control_edge(w1, w3)
                g.add_control_edge(w3, w2)
                exp_prev = w2
            g.add_data_edge(exp_prev, combine, {"y": f"i{gpu_id}"})

        # Shared expert
        sw1 = place(make_gemm(M, MOE_INTER, D, "fp8", "fp8"))
        sw3 = place(make_gemm(M, MOE_INTER, D, "fp8", "fp8"))
        sw2 = place(make_gemm(M, D, MOE_INTER, "fp8", "fp8"))
        g.add_data_edge(combine, sw1, {"y": "x"})
        g.add_control_edge(sw1, sw3)
        g.add_control_edge(sw3, sw2)
        prev = sw2

    return g, p


# ── Main ────────────────────────────────────────────────────────────────

def main():
    hw = B300ClusterA(n_nodes=1)
    gpus = [c for c in hw.nodes
            if isinstance(c, Compute) and "nvidia-b300" in c.name]
    gpus.sort(key=lambda c: c.name)

    g, p = build_prefill(hw, gpus)
    result = Simulator(g, p, hw).run()
    export_trace(result, "dsv4_pro_prefill.json")
    print(f"Prefill: {result.total_time_us:.1f} us "
          f"({result.total_time_us / 1000:.1f} ms)")


if __name__ == "__main__":
    from rooflang.language.hardware.component import Compute
    main()
