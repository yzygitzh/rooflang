"""Split and duplication callables for semantics-preserving graph rewrites.

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

from copy import deepcopy

from rooflang.language.kernels.comm import Broadcast, Gather, Reduce, Scatter
from rooflang.language.kernels.forward import (
    Attn, ElementwiseOp, Embedding, Gemm, Nop, ReadInput, RMSNorm, Sampling,
    SparseAttn, Slice, StridedGemm, TokenCombine, TokenDispatch,
)
from rooflang.language.kernels.identity import Concat, Move, Spawn
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


def _make_dependency_gather(tensor, n):
    """Join replicated dummy outputs from split dependency-only kernels."""
    comm = Gather(total_bytes=tensor.size_bytes, world=n, dim=0)
    comm.inputs = {
        f"i{i}": Tensor(tensor.dtype, tensor.shape) for i in range(n)}
    comm.outputs = {"y": Tensor(tensor.dtype, tensor.shape)}
    return comm


def decode_attention_context_split(kernel, n):
    """Split one decode attention over equal-size persistent KV shards."""
    if not isinstance(kernel, SparseAttn):
        raise TypeError("decode attention split requires SparseAttn")
    if kernel.S_kv % n != 0:
        raise ValueError(
            f"attention S_kv={kernel.S_kv} must be divisible by {n}")
    if kernel.k_sel % n != 0:
        raise ValueError(
            f"attention k_sel={kernel.k_sel} must be divisible by {n}")

    q_tensor = kernel.inputs["q"]
    kv_tensor = kernel.inputs["kv"]
    out_tensor = kernel.outputs["y"]
    local_kv = kernel.S_kv // n
    prev_comms = {
        "q": _make_broadcast(q_tensor, n),
        "kv": _make_scatter(kv_tensor, n, dim=1),
    }
    copies = []
    for _ in range(n):
        copy = SparseAttn(
            kernel.B, kernel.H, kernel.H_kv, kernel.S_q,
            kernel.k_sel // n, local_kv, kernel.Hd, kernel.dtype_,
            kernel.kv_factor)
        copy.inputs = {
            "q": Tensor(q_tensor.dtype, q_tensor.shape),
            "kv": Tensor(
                kv_tensor.dtype, _shard_shape(kv_tensor.shape, n, dim=1)),
        }
        copy.outputs = {
            "y": Tensor(out_tensor.dtype, out_tensor.shape)}
        copies.append(copy)
    next_comms = {
        "y": _make_reduce(out_tensor, n, dtype=out_tensor.dtype)}
    return prev_comms, copies, next_comms


def decode_persistence_split(kernel, n):
    """Split a decode KV barrier while replicating its scalar completion."""
    if not isinstance(kernel, Nop) or "decode_output" not in kernel.inputs:
        raise TypeError("decode persistence split requires its barrier Nop")

    prev_comms = {}
    for port, tensor in kernel.inputs.items():
        if port == "decode_output":
            prev_comms[port] = _make_broadcast(tensor, n)
        else:
            prev_comms[port] = _make_scatter(tensor, n, dim=1)

    copies = []
    for _ in range(n):
        copy = Nop()
        copy.inputs = {
            port: Tensor(
                tensor.dtype,
                tensor.shape if port == "decode_output"
                else _shard_shape(tensor.shape, n, dim=1))
            for port, tensor in kernel.inputs.items()
        }
        copy.outputs = {
            port: Tensor(tensor.dtype, tensor.shape)
            for port, tensor in kernel.outputs.items()
        }
        copies.append(copy)
    next_comms = {
        port: _make_dependency_gather(tensor, n)
        for port, tensor in kernel.outputs.items()
    }
    return prev_comms, copies, next_comms


def replicate_before(kernel, n):
    """Dup callable that moves a one-input Broadcast before the kernel."""
    if len(kernel.inputs) != 1 or len(kernel.outputs) != 1:
        raise ValueError("replicate_before requires one input and one output")
    input_name, input_tensor = next(iter(kernel.inputs.items()))
    broadcast = _make_broadcast(input_tensor, n)
    broadcast.inputs = {
        input_name: Tensor(input_tensor.dtype, input_tensor.shape)}
    copies = [deepcopy(kernel) for _ in range(n)]
    return broadcast, copies


def batch_split_comm(kernel, n):
    """Replicate one collective across independent batch/DP groups.

    The original collective operates on the full batch. Each copy operates on
    one batch shard; per-port Scatter/Gather wrappers preserve the original
    graph interface until adjacent compute kernels receive the same DP split.
    """
    if not isinstance(kernel, (Broadcast, Gather, Reduce, Scatter)):
        raise TypeError(
            "batch collective split requires a primitive communication "
            "kernel")
    tensors = (*kernel.inputs.values(), *kernel.outputs.values())
    if any(not tensor.shape or tensor.shape[0] % n != 0
           for tensor in tensors):
        raise ValueError(
            "every collective tensor batch dimension must be divisible by "
            f"{n}")

    prev_comms = {
        port: _make_scatter(tensor, n)
        for port, tensor in kernel.inputs.items()
    }
    copies = []
    for _ in range(n):
        copy = deepcopy(kernel)
        copy.total_bytes /= n
        copy.inputs = {
            port: Tensor(tensor.dtype, _shard_shape(tensor.shape, n))
            for port, tensor in kernel.inputs.items()
        }
        copy.outputs = {
            port: Tensor(tensor.dtype, _shard_shape(tensor.shape, n))
            for port, tensor in kernel.outputs.items()
        }
        copies.append(copy)
    next_comms = {
        port: _make_gather(tensor, n)
        for port, tensor in kernel.outputs.items()
    }
    return prev_comms, copies, next_comms


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


# ── context_split ─────────────────────────────────────────────────────


def context_split(kernel, n):
    """Shard token work along sequence dim 1 while preserving batch size.

    Expert-token tensors produced by TokenDispatch and consumed by
    TokenCombine are flattened as (M_e, D), so those ports shard dim 0.
    Attention receives a logical full-KV input through AllGather.  Simulator
    passthrough keeps its physical storage backed by the distributed source
    shards, while each rank accounts for processing all KV blocks in the ring.
    """
    if isinstance(kernel, (Attn, SparseAttn)):
        return _context_split_attn(kernel, n)

    prev_comms = {
        port: _make_scatter(
            tensor, n, dim=_context_port_dim(kernel, port, is_input=True))
        for port, tensor in kernel.inputs.items()
    }
    copies = [_make_context_copy(kernel, n) for _ in range(n)]
    if isinstance(kernel, Nop):
        next_comms = {
            port: _make_dependency_gather(tensor, n)
            for port, tensor in kernel.outputs.items()
        }
    else:
        next_comms = {
            port: _make_gather(
                tensor, n,
                dim=_context_port_dim(kernel, port, is_input=False))
            for port, tensor in kernel.outputs.items()
        }
    return prev_comms, copies, next_comms


def _context_port_dim(kernel, port, *, is_input):
    """Return the physical tensor axis carrying sequence-sharded tokens."""
    if isinstance(kernel, TokenDispatch) and not is_input:
        return 0
    if isinstance(kernel, TokenCombine) and is_input:
        return 0
    return 1


def _context_split_attn(kernel, n):
    """Context/ring split for dense and sparse attention."""
    q_tensor = kernel.inputs["q"]
    kv_tensor = kernel.inputs["kv"]
    out_tensor = kernel.outputs["y"]
    q_shard = _shard_shape(q_tensor.shape, n, dim=1)
    out_shard = _shard_shape(out_tensor.shape, n, dim=1)

    prev_comms = {
        "q": _make_scatter(q_tensor, n, dim=1),
        "kv": _make_broadcast(kv_tensor, n),
    }

    copies = []
    for _ in range(n):
        if isinstance(kernel, SparseAttn):
            c = SparseAttn(
                kernel.B, kernel.H, kernel.H_kv, kernel.S_q // n,
                kernel.k_sel, kernel.S_kv, kernel.Hd, kernel.dtype_,
                kernel.kv_factor)
        else:
            c = Attn(
                kernel.B, kernel.H, kernel.H_kv, kernel.S_q // n,
                kernel.S_kv, kernel.Hd, kernel.dtype_, kernel.causal)
        c.inputs = {
            "q": Tensor(q_tensor.dtype, q_shard),
            "kv": Tensor(kv_tensor.dtype, kv_tensor.shape),
        }
        c.outputs = {"y": Tensor(out_tensor.dtype, out_shard)}
        copies.append(c)

    next_comms = {"y": _make_gather(out_tensor, n, dim=1)}
    return prev_comms, copies, next_comms


def _make_context_copy(kernel, n):
    """Construct one context-sharded copy of a non-attention kernel."""
    if isinstance(kernel, StridedGemm):
        c = StridedGemm(
            kernel.M // n, kernel.N, kernel.K, kernel.w_dtype,
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
    elif isinstance(kernel, Spawn):
        c = Spawn(world=kernel.world)
    elif isinstance(kernel, Concat):
        c = Concat()
    elif isinstance(kernel, Slice):
        c = Slice()
    elif isinstance(kernel, Move):
        c = Move()
    elif isinstance(kernel, Sampling):
        c = Sampling(kernel.M // n, kernel.V, kernel.dtype_, kernel.out_dtype)
    elif isinstance(kernel, ElementwiseOp):
        c = ElementwiseOp(kernel.M // n, kernel.D, kernel.dtype_, kernel.op)
    elif isinstance(kernel, Nop):
        c = Nop()
    else:
        raise TypeError(
            f"context_split: unsupported kernel type "
            f"{type(kernel).__name__}")

    c.inputs = {
        port: Tensor(
            tensor.dtype,
            _shard_shape(
                tensor.shape, n,
                _context_port_dim(kernel, port, is_input=True)))
        for port, tensor in kernel.inputs.items()
    }
    if isinstance(kernel, Nop):
        c.outputs = {
            port: Tensor(tensor.dtype, tensor.shape)
            for port, tensor in kernel.outputs.items()
        }
    else:
        c.outputs = {
            port: Tensor(
                tensor.dtype,
                _shard_shape(
                    tensor.shape, n,
                    _context_port_dim(kernel, port, is_input=False)))
            for port, tensor in kernel.outputs.items()
        }
    if kernel.weights:
        if isinstance(kernel, Embedding):
            c.weights = {
                "emb": Tensor(
                    kernel.w_dtype, (kernel.V, kernel.D),
                    weight_id=kernel.weights["emb"].weight_id)}
        else:
            c.weights = {
                port: Tensor(
                    tensor.dtype, tensor.shape, weight_id=tensor.weight_id)
                for port, tensor in kernel.weights.items()
            }
    return c


# ── batch_split ────────────────────────────────────────────────────────


def batch_split(kernel, n):
    """Non-contracting batch dim (M/token count): Scatter → Kernels → Gather.

    Splits the leading token dimension (M) across n ranks. Each copy
    processes M/n tokens independently. Weights are replicated.
    """
    prev_comms = {port: _make_scatter(tensor, n)
                  for port, tensor in kernel.inputs.items()}
    copies = [_make_batch_copy(kernel, n) for _ in range(n)]
    if isinstance(kernel, Nop):
        next_comms = {
            port: _make_dependency_gather(tensor, n)
            for port, tensor in kernel.outputs.items()
        }
    else:
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
    elif isinstance(kernel, Move):
        c = Move()
    elif isinstance(kernel, Sampling):
        c = Sampling(kernel.M // n, kernel.V, kernel.dtype_, kernel.out_dtype)
    elif isinstance(kernel, ElementwiseOp):
        c = ElementwiseOp(kernel.M // n, kernel.D, kernel.dtype_, kernel.op)
    elif isinstance(kernel, Nop):
        c = Nop()
    else:
        raise TypeError(
            f"batch_split: unsupported kernel type {type(kernel).__name__}")

    c.inputs = {k: Tensor(t.dtype, _shard_shape(t.shape, n))
                for k, t in kernel.inputs.items()}
    if isinstance(kernel, Nop):
        c.outputs = {k: Tensor(t.dtype, t.shape)
                     for k, t in kernel.outputs.items()}
    else:
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
