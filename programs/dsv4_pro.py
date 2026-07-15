"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 TP=8."""

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.comm import AllReduce, AllToAll, Broadcast, Gather
from rooflang.language.kernels.forward import Gemm, RMSNorm, SparseAttn
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


# ── Split classes ───────────────────────────────────────────────────────

def column_split(kernel, n):
    shard_n = kernel.N // n
    prev = Broadcast(bytes_per_rank=0, world=n)
    prev.outputs = {f"o{i}": Tensor(kernel.a_dtype, (kernel.M * kernel.K,))
                    for i in range(n)}
    copies = []
    for _ in range(n):
        c = Gemm(kernel.M, shard_n, kernel.K, kernel.w_dtype,
                 kernel.a_dtype, kernel.out_dtype)
        c.inputs = {"x": Tensor(kernel.a_dtype, (kernel.M * kernel.K,))}
        c.weights = {"w": Tensor(kernel.w_dtype, (kernel.K * shard_n,))}
        c.outputs = {"y": Tensor(kernel.out_dtype, (kernel.M * shard_n,))}
        copies.append(c)
    next_ = Gather(bytes_per_rank=0, world=n)
    next_.inputs = {f"i{i}": Tensor(kernel.out_dtype, (kernel.M * shard_n,))
                    for i in range(n)}
    return prev, copies, next_


def row_split_allreduce(kernel, n):
    shard_k = kernel.K // n
    prev = Broadcast(bytes_per_rank=0, world=n)
    prev.outputs = {f"o{i}": Tensor(kernel.a_dtype, (kernel.M * shard_k,))
                    for i in range(n)}
    copies = []
    for _ in range(n):
        c = Gemm(kernel.M, kernel.N, shard_k, kernel.w_dtype,
                 kernel.a_dtype, kernel.out_dtype)
        c.inputs = {"x": Tensor(kernel.a_dtype, (kernel.M * shard_k,))}
        c.weights = {"w": Tensor(kernel.w_dtype, (shard_k * kernel.N,))}
        c.outputs = {"y": Tensor(kernel.out_dtype, (kernel.M * kernel.N,))}
        copies.append(c)
    ar_bytes = kernel.M * kernel.N * dtype_bytes(kernel.out_dtype)
    next_ = AllReduce(bytes_per_rank=ar_bytes, world=n, dtype=kernel.out_dtype)
    next_.inputs = {f"i{i}": Tensor(kernel.out_dtype, (kernel.M * kernel.N,))
                    for i in range(n)}
    return prev, copies, next_


def head_split(kernel, n):
    shard_h = kernel.H // n
    in_elems = (kernel.B * shard_h * kernel.S_q * kernel.Hd
                + 2 * kernel.B * kernel.H_kv * kernel.S_q
                * kernel.k_sel * kernel.Hd)
    out_elems = kernel.B * shard_h * kernel.S_q * kernel.Hd
    prev = Broadcast(bytes_per_rank=0, world=n)
    prev.outputs = {f"o{i}": Tensor(kernel.dtype_, (in_elems,))
                    for i in range(n)}
    copies = []
    for _ in range(n):
        c = SparseAttn(kernel.B, shard_h, kernel.H_kv, kernel.S_q,
                       kernel.k_sel, kernel.Hd, kernel.dtype_)
        c.inputs = {"x": Tensor(kernel.dtype_, (in_elems,))}
        c.outputs = {"y": Tensor(kernel.dtype_, (out_elems,))}
        copies.append(c)
    next_ = Gather(bytes_per_rank=0, world=n)
    next_.inputs = {f"i{i}": Tensor(kernel.dtype_, (out_elems,))
                    for i in range(n)}
    return prev, copies, next_


# ── Build prefill graph ─────────────────────────────────────────────────

def build_prefill(hw, gpus):
    g = ComputeGraph()
    p = Placement(hardware=hw, graph=g)
    M = S_PREFILL

    def place(kernel, gpu=gpus[0]):
        g.add_kernel(kernel)
        p.set_kernel_device(kernel, gpu)
        return kernel

    def split_tp(kernel, split_fn, after=None):
        g.add_kernel(kernel)
        prev_c, copies, next_c = g.split_kernel(split_fn, kernel, TP)
        for i, c in enumerate(copies):
            p.set_kernel_device(c, gpus[i])
        if after is not None:
            g.add_control_edge(after, prev_c)
        return next_c

    prev = None
    for layer_id in range(N_LAYERS):
        ratio = COMPRESS_RATIOS[layer_id]

        # ── Attention ───────────────────────────────────────────────
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

        # wq_b: ColumnParallel split on N
        wq_b_end = split_tp(
            Gemm(M, H * HD, Q_LORA, "fp8", "fp8"), column_split, after=q_norm)

        # KV path (sequential after wq_b split for V1 simplicity)
        g.add_control_edge(wq_b_end, wkv)
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
        sa_end = split_tp(
            SparseAttn(1, H, 1, M, k_sel, HD), head_split, after=sa_after)

        # Output projection
        wo_a_end = split_tp(
            Gemm(M, O_GROUPS * O_LORA, H * HD // O_GROUPS, "bf16", "bf16"),
            column_split, after=sa_end)
        wo_b_end = split_tp(
            Gemm(M, D, O_GROUPS * O_LORA, "fp8", "fp8"),
            row_split_allreduce, after=wo_a_end)

        # ── FFN / MoE ──────────────────────────────────────────────
        ffn_norm = place(make_norm(M, D))
        gate = place(make_gemm(M, N_EXPERTS, D, "fp32", "bf16", "fp32"))
        g.add_control_edge(wo_b_end, ffn_norm)
        g.add_control_edge(ffn_norm, gate)

        # AllToAll dispatch + combine
        dispatch = AllToAll(M * D * dtype_bytes("bf16"), EP)
        combine = AllToAll(M * D * dtype_bytes("bf16"), EP)
        g.add_kernel(dispatch)
        g.add_kernel(combine)
        g.add_control_edge(gate, dispatch)

        # Experts: 48 per GPU × 8 GPUs
        M_e = M * TOPK // N_LOCAL_EXPERTS
        for gpu_id in range(TP):
            exp_prev = dispatch
            for eid in range(N_LOCAL_EXPERTS):
                w1 = place(make_gemm(M_e, MOE_INTER, D, "fp4", "fp8"),
                           gpus[gpu_id])
                w3 = place(make_gemm(M_e, MOE_INTER, D, "fp4", "fp8"),
                           gpus[gpu_id])
                w2 = place(make_gemm(M_e, D, MOE_INTER, "fp4", "fp8"),
                           gpus[gpu_id])
                g.add_control_edge(exp_prev, w1)
                g.add_control_edge(w1, w3)
                g.add_control_edge(w3, w2)
                exp_prev = w2
            g.add_control_edge(exp_prev, combine)

        # Shared expert (replicated on GPU0)
        sw1 = place(make_gemm(M, MOE_INTER, D, "fp8", "fp8"))
        sw3 = place(make_gemm(M, MOE_INTER, D, "fp8", "fp8"))
        sw2 = place(make_gemm(M, D, MOE_INTER, "fp8", "fp8"))
        g.add_control_edge(combine, sw1)
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
