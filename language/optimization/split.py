"""Split callables for ComputeGraph.split_kernel().

Each callable takes (kernel, n) and returns (prev_comms, copies, next_comms):
  - prev_comms: Dict[str, Kernel] — one comm kernel per input port
  - copies: List[Kernel] — the split compute kernels
  - next_comms: Dict[str, Kernel] — one comm kernel per output port

Each prev_comm has 1 input ("x") and n outputs ("o0".."o{n-1}").
Each next_comm has n inputs ("i0".."i{n-1}") and 1 output ("y").

| Split dimension            | prev_comm type | next_comm type |
|----------------------------|----------------|----------------|
| Non-contracting input dim  | Scatter        | Gather         |
| Non-contracting output dim | Broadcast      | Gather         |
| Contracting dim            | Scatter        | Reduce         |
"""

from rooflang.language.kernels.comm import Broadcast, Gather, Reduce, Scatter
from rooflang.language.kernels.forward import (
    ElementwiseOp, Embedding, Gemm, ReadInput, RMSNorm, Sampling, SparseAttn,
    Slice, StridedGemm, TokenCombine, TokenDispatch,
)
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.tensor import Tensor
from rooflang.language.utils import dtype_bytes, gemm_scale_bytes


# ── Shared helpers ─────────────────────────────────────────────────────


def _shard_shape(shape, n, dim=0):
    """Divide shape[dim] by n."""
    if dim < 0:
        dim = len(shape) + dim
    lst = list(shape)
    lst[dim] = lst[dim] // n
    return tuple(lst)


def _make_scatter(tensor, n, dim=0):
    """Create a Scatter comm: 1 input "x", n outputs "o0".."o{n-1}"."""
    if dim < 0:
        dim = len(tensor.shape) + dim
    shard = _shard_shape(tensor.shape, n, dim)
    comm = Scatter(total_bytes=tensor.size_bytes, world=n, dim=dim)
    comm.inputs = {"x": Tensor(tensor.dtype, tensor.shape)}
    comm.outputs = {f"o{i}": Tensor(tensor.dtype, shard) for i in range(n)}
    return comm


def _make_gather(tensor, n, dim=0):
    """Create a Gather comm: n inputs "i0".."i{n-1}", 1 output "y"."""
    if dim < 0:
        dim = len(tensor.shape) + dim
    shard = _shard_shape(tensor.shape, n, dim)
    comm = Gather(total_bytes=tensor.size_bytes, world=n, dim=dim)
    comm.inputs = {f"i{i}": Tensor(tensor.dtype, shard) for i in range(n)}
    comm.outputs = {"y": Tensor(tensor.dtype, tensor.shape)}
    return comm


def _make_broadcast(tensor, n):
    """Create a Broadcast comm: 1 input "x", n outputs "o0".."o{n-1}" (unchanged shape)."""
    comm = Broadcast(total_bytes=tensor.size_bytes, world=n)
    comm.inputs = {"x": Tensor(tensor.dtype, tensor.shape)}
    comm.outputs = {f"o{i}": Tensor(tensor.dtype, tensor.shape)
                    for i in range(n)}
    return comm


def _make_reduce(tensor, n, dtype="bf16"):
    """Create a Reduce comm: n inputs "i0".."i{n-1}", 1 output "y" (unchanged shape)."""
    comm = Reduce(total_bytes=tensor.size_bytes, world=n, dtype=dtype)
    comm.inputs = {f"i{i}": Tensor(tensor.dtype, tensor.shape)
                   for i in range(n)}
    comm.outputs = {"y": Tensor(tensor.dtype, tensor.shape)}
    return comm


# ── column_split / row_split / head_split ──────────────────────────────


def _propagate_weight_id(orig_weights, copy_weights, shard_idx, split_type):
    """Propagate weight_id with shard tag for TP splits."""
    for port, t in copy_weights.items():
        orig = orig_weights.get(port)
        if orig is not None and orig.weight_id is not None:
            t.weight_id = f"{orig.weight_id}/{split_type}:{shard_idx}"


def column_split(kernel, n):
    """Non-contracting output dim (N): Broadcast → Kernels → Gather.

    Input is replicated (Broadcast) to all ranks; output shards are
    concatenated (Gather). Works for both Gemm and StridedGemm.
    """
    shard_n = kernel.N // n
    in_tensor = kernel.inputs["x"]
    out_tensor = kernel.outputs["y"]

    prev_comms = {"x": _make_broadcast(in_tensor, n)}

    copies = []
    if isinstance(kernel, StridedGemm):
        shard_out_elems = kernel._out_elems // n
        shard_out_shape = _shard_shape(out_tensor.shape, n, dim=-1)
        for i in range(n):
            c = StridedGemm(kernel.M, shard_n, kernel.K, kernel.w_dtype,
                            kernel.a_dtype, kernel.out_dtype,
                            in_elems=kernel._in_elems, out_elems=shard_out_elems)
            c.inputs = {"x": Tensor(in_tensor.dtype, in_tensor.shape)}
            c.weights = {"w": Tensor(kernel.w_dtype, (kernel.K, shard_n))}
            scale_bytes = gemm_scale_bytes(shard_n, kernel.K, kernel.w_dtype)
            if scale_bytes > 0:
                c.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            _propagate_weight_id(kernel.weights, c.weights, i, "col")
            c.outputs = {"y": Tensor(kernel.out_dtype, shard_out_shape)}
            copies.append(c)
    else:
        shard_out_shape = _shard_shape(out_tensor.shape, n, dim=-1)
        for i in range(n):
            c = Gemm(kernel.M, shard_n, kernel.K, kernel.w_dtype,
                     kernel.a_dtype, kernel.out_dtype)
            c.inputs = {"x": Tensor(in_tensor.dtype, in_tensor.shape)}
            c.weights = {"w": Tensor(kernel.w_dtype, (kernel.K, shard_n))}
            scale_bytes = gemm_scale_bytes(shard_n, kernel.K, kernel.w_dtype)
            if scale_bytes > 0:
                c.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            _propagate_weight_id(kernel.weights, c.weights, i, "col")
            c.outputs = {"y": Tensor(kernel.out_dtype, shard_out_shape)}
            copies.append(c)

    next_comms = {"y": _make_gather(out_tensor, n, dim=-1)}

    return prev_comms, copies, next_comms


def row_split(kernel, n):
    """Contracting dim (K): Scatter → Kernels → Reduce.

    Input is partitioned (Scatter) across ranks; partial sums are
    reduced (Reduce).
    """
    shard_k = kernel.K // n
    in_tensor = kernel.inputs["x"]
    out_tensor = kernel.outputs["y"]
    shard_in_shape = _shard_shape(in_tensor.shape, n, dim=-1)

    prev_comms = {"x": _make_scatter(in_tensor, n, dim=-1)}

    copies = []
    if isinstance(kernel, StridedGemm):
        for i in range(n):
            c = StridedGemm(kernel.M, kernel.N, shard_k, kernel.w_dtype,
                            kernel.a_dtype, kernel.out_dtype,
                            in_elems=kernel._in_elems // n,
                            out_elems=kernel._out_elems)
            c.inputs = {"x": Tensor(in_tensor.dtype, shard_in_shape)}
            c.weights = {"w": Tensor(kernel.w_dtype, (shard_k, kernel.N))}
            scale_bytes = gemm_scale_bytes(kernel.N, shard_k, kernel.w_dtype)
            if scale_bytes > 0:
                c.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            _propagate_weight_id(kernel.weights, c.weights, i, "row")
            c.outputs = {"y": Tensor(out_tensor.dtype, out_tensor.shape)}
            copies.append(c)
    else:
        for i in range(n):
            c = Gemm(kernel.M, kernel.N, shard_k, kernel.w_dtype,
                     kernel.a_dtype, kernel.out_dtype)
            c.inputs = {"x": Tensor(in_tensor.dtype, shard_in_shape)}
            c.weights = {"w": Tensor(kernel.w_dtype, (shard_k, kernel.N))}
            scale_bytes = gemm_scale_bytes(kernel.N, shard_k, kernel.w_dtype)
            if scale_bytes > 0:
                c.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            _propagate_weight_id(kernel.weights, c.weights, i, "row")
            c.outputs = {"y": Tensor(out_tensor.dtype, out_tensor.shape)}
            copies.append(c)

    next_comms = {"y": _make_reduce(out_tensor, n, dtype=kernel.out_dtype)}

    return prev_comms, copies, next_comms


def head_split(kernel, n):
    """Non-contracting input dim (H): per-port Scatter/Broadcast → Kernels → Gather.

    Q input is partitioned (Scatter) by head; KV input is broadcast
    (same KV cache to each rank). Output shards are concatenated (Gather).
    """
    shard_h = kernel.H // n
    q_tensor = kernel.inputs["q"]
    kv_tensor = kernel.inputs["kv"]
    out_tensor = kernel.outputs["y"]
    shard_q_shape = _shard_shape(q_tensor.shape, n, dim=-1)

    prev_comms = {
        "q": _make_scatter(q_tensor, n, dim=-1),
        "kv": _make_broadcast(kv_tensor, n),
    }

    copies = []
    for _ in range(n):
        c = SparseAttn(kernel.B, shard_h, kernel.H_kv, kernel.S_q,
                       kernel.k_sel, kernel.S_kv, kernel.Hd, kernel.dtype_,
                       kernel.kv_factor)
        c.inputs = {"q": Tensor(kernel.dtype_, shard_q_shape),
                    "kv": Tensor(kernel.dtype_, kv_tensor.shape)}
        c.outputs = {"y": Tensor(kernel.dtype_, shard_q_shape)}
        copies.append(c)

    next_comms = {"y": _make_gather(out_tensor, n, dim=-1)}

    return prev_comms, copies, next_comms


# ── batch_split ────────────────────────────────────────────────────────


def batch_split(kernel, n):
    """Non-contracting batch dim (M/token count): Scatter → Kernels → Gather.

    Splits the leading token dimension (M) across n ranks. Each copy
    processes M/n tokens independently. Weights are replicated.
    """
    prev_comms = {port: _make_scatter(tensor, n)
                  for port, tensor in kernel.inputs.items()}
    copies = [_make_batch_copy(kernel, n) for _ in range(n)]
    next_comms = {port: _make_gather(tensor, n)
                  for port, tensor in kernel.outputs.items()}
    return prev_comms, copies, next_comms


def _make_batch_copy(kernel, n):
    """Construct a single batch-sharded copy of kernel (M → M/n)."""
    if isinstance(kernel, StridedGemm):
        c = StridedGemm(kernel.M // n, kernel.N, kernel.K, kernel.w_dtype,
                        kernel.a_dtype, kernel.out_dtype,
                        in_elems=kernel._in_elems // n,
                        out_elems=kernel._out_elems // n)
    elif isinstance(kernel, Gemm):
        c = Gemm(kernel.M // n, kernel.N, kernel.K, kernel.w_dtype,
                 kernel.a_dtype, kernel.out_dtype)
    elif isinstance(kernel, RMSNorm):
        c = RMSNorm(kernel.M // n, kernel.D, kernel.dtype_)
    elif isinstance(kernel, Embedding):
        c = Embedding(kernel.M // n, kernel.V, kernel.D, kernel.w_dtype,
                      kernel.idx_dtype, kernel.out_dtype)
    elif isinstance(kernel, ReadInput):
        c = ReadInput(kernel.n_elements // n, kernel.dtype_)
    elif isinstance(kernel, TokenDispatch):
        c = TokenDispatch(kernel.M // n, kernel.D, kernel.N_experts,
                          kernel.topk, kernel.a_dtype)
    elif isinstance(kernel, TokenCombine):
        c = TokenCombine(kernel.M // n, kernel.D, kernel.N_experts,
                         kernel.topk, kernel.a_dtype)
    elif isinstance(kernel, SparseAttn):
        c = SparseAttn(kernel.B // n, kernel.H, kernel.H_kv, kernel.S_q,
                       kernel.k_sel, kernel.S_kv, kernel.Hd, kernel.dtype_,
                       kernel.kv_factor)
    elif isinstance(kernel, Spawn):
        c = Spawn(world=kernel.world)
    elif isinstance(kernel, Concat):
        c = Concat()
    elif isinstance(kernel, Slice):
        c = Slice()
    elif isinstance(kernel, Sampling):
        c = Sampling(kernel.M // n, kernel.V, kernel.dtype_, kernel.out_dtype)
    elif isinstance(kernel, ElementwiseOp):
        c = ElementwiseOp(kernel.M // n, kernel.D, kernel.dtype_, kernel.op)
    else:
        raise TypeError(
            f"batch_split: unsupported kernel type {type(kernel).__name__}")

    c.inputs = {k: Tensor(t.dtype, _shard_shape(t.shape, n))
                for k, t in kernel.inputs.items()}
    c.outputs = {k: Tensor(t.dtype, _shard_shape(t.shape, n))
                 for k, t in kernel.outputs.items()}
    if kernel.weights:
        if isinstance(kernel, Embedding):
            c.weights = {"emb": Tensor(kernel.w_dtype, (kernel.V, kernel.D),
                                       weight_id=kernel.weights["emb"].weight_id)}
        else:
            c.weights = {k: Tensor(t.dtype, t.shape, weight_id=t.weight_id)
                         for k, t in kernel.weights.items()}
    return c
