"""Split callables for ComputeGraph.split_kernel().

Each callable takes (kernel, n) and returns (prev_comm, copies, next_comm).

| Split dimension            | Pattern                       |
|----------------------------|-------------------------------|
| Non-contracting input dim  | Scatter → Kernels → Gather    |
| Non-contracting output dim | Broadcast → Kernels → Gather  |
| Contracting dim            | Scatter → Kernels → Reduce    |

When the prev_comm is a zero-cost placeholder (no actual wire traffic),
Spawn is used instead of a real CommKernel.
"""

from rooflang.language.kernels.comm import Broadcast, Gather, Reduce, Scatter
from rooflang.language.kernels.identity import Spawn
from rooflang.language.kernels.forward import Gemm, SparseAttn
from rooflang.language.tensor import Tensor
from rooflang.language.utils import dtype_bytes


def column_split(kernel, n):
    """Non-contracting output dim (N): Broadcast → Kernels → Gather.

    Input is replicated (Broadcast) to all ranks; output shards are
    concatenated (Gather).
    """
    shard_n = kernel.N // n
    input_bytes = kernel.M * kernel.K * dtype_bytes("bf16")
    prev = Broadcast(bytes_per_rank=input_bytes, world=n)
    prev.inputs = dict(kernel.inputs)
    prev.outputs = {f"o{i}": Tensor("bf16", (kernel.M * kernel.K,))
                    for i in range(n)}
    copies = []
    for _ in range(n):
        c = Gemm(kernel.M, shard_n, kernel.K, kernel.w_dtype,
                 kernel.a_dtype, kernel.out_dtype)
        c.inputs = {"x": Tensor("bf16", (kernel.M * kernel.K,))}
        c.weights = {"w": Tensor(kernel.w_dtype, (kernel.K * shard_n,))}
        c.outputs = {"y": Tensor("bf16", (kernel.M * shard_n,))}
        copies.append(c)
    output_bytes = kernel.M * kernel.N * dtype_bytes("bf16")
    next_ = Gather(bytes_per_rank=output_bytes, world=n)
    next_.inputs = {f"i{i}": Tensor("bf16", (kernel.M * shard_n,))
                    for i in range(n)}
    next_.outputs = dict(kernel.outputs)
    return prev, copies, next_


def row_split(kernel, n):
    """Contracting dim (K): Scatter → Kernels → Reduce.

    Input is partitioned (Scatter) across ranks; partial sums are
    reduced (Reduce).
    """
    shard_k = kernel.K // n
    input_bytes = kernel.M * kernel.K * dtype_bytes("bf16")
    prev = Scatter(bytes_per_rank=input_bytes, world=n)
    prev.inputs = dict(kernel.inputs)
    prev.outputs = {f"o{i}": Tensor("bf16", (kernel.M * shard_k,))
                    for i in range(n)}
    copies = []
    for _ in range(n):
        c = Gemm(kernel.M, kernel.N, shard_k, kernel.w_dtype,
                 kernel.a_dtype, kernel.out_dtype)
        c.inputs = {"x": Tensor("bf16", (kernel.M * shard_k,))}
        c.weights = {"w": Tensor(kernel.w_dtype, (shard_k * kernel.N,))}
        c.outputs = {"y": Tensor("bf16", (kernel.M * kernel.N,))}
        copies.append(c)
    reduce_bytes = kernel.M * kernel.N * dtype_bytes("bf16")
    next_ = Reduce(bytes_per_rank=reduce_bytes, world=n)
    next_.inputs = {f"i{i}": Tensor("bf16", (kernel.M * kernel.N,))
                    for i in range(n)}
    next_.outputs = dict(kernel.outputs)
    return prev, copies, next_


def head_split(kernel, n):
    """Non-contracting input dim (H): Scatter → Kernels → Gather.

    Input is partitioned (Scatter) by head; output shards are
    concatenated (Gather).
    """
    shard_h = kernel.H // n
    q_elems = kernel.B * shard_h * kernel.S_q * kernel.Hd
    kv_elems = 2 * kernel.B * kernel.H_kv * kernel.S_q * kernel.k_sel * kernel.Hd
    in_elems = q_elems + kv_elems
    out_elems = kernel.B * shard_h * kernel.S_q * kernel.Hd
    in_bytes = kernel.B * kernel.H * kernel.S_q * kernel.Hd * dtype_bytes(kernel.dtype_)
    prev = Scatter(bytes_per_rank=in_bytes, world=n)
    prev.inputs = dict(kernel.inputs)
    prev.outputs = {f"o{i}": Tensor(kernel.dtype_, (in_elems,))
                    for i in range(n)}
    copies = []
    for _ in range(n):
        c = SparseAttn(kernel.B, shard_h, kernel.H_kv, kernel.S_q,
                       kernel.k_sel, kernel.Hd, kernel.dtype_)
        c.inputs = {"x": Tensor(kernel.dtype_, (in_elems,))}
        c.outputs = {"y": Tensor(kernel.dtype_, (out_elems,))}
        copies.append(c)
    out_bytes = kernel.B * kernel.H * kernel.S_q * kernel.Hd * dtype_bytes(kernel.dtype_)
    next_ = Gather(bytes_per_rank=out_bytes, world=n)
    next_.inputs = {f"i{i}": Tensor(kernel.dtype_, (out_elems,))
                    for i in range(n)}
    next_.outputs = dict(kernel.outputs)
    return prev, copies, next_
