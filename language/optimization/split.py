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
from fractions import Fraction

from rooflang.language.kernels.comm import Broadcast, Gather, Reduce, Scatter
from rooflang.language.kernels.forward import (
    Attn, AttnRes, DpskV4SparseAttn, ElementwiseOp, Embedding, Gemm,
    Glm52SparseAttn, KdaStateStore, KimiK3DeltaAttn, KimiK3MlaAttn, LayerNorm,
    Nop,
    PartialRMSNorm, ReadInput, RMSNorm, Sampling, Slice, StridedGemm,
    TokenCombine, TokenDispatch,
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
    if isinstance(lst[dim], Fraction):
        lst[dim] /= n
    else:
        lst[dim] //= n
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


def _context_split_attn_decode(kernel, n):
    """Split one decode attention over equal-size persistent KV shards."""
    if not isinstance(
            kernel, (DpskV4SparseAttn, Glm52SparseAttn, KimiK3MlaAttn)):
        raise TypeError(
            "decode attention split requires an MLA/sparse-attention kernel")
    if kernel.S_kv % n != 0:
        raise ValueError(
            f"attention S_kv={kernel.S_kv} must be divisible by {n}")
    if hasattr(kernel, "k_sel") and kernel.k_sel % n != 0:
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
    index_tensor = kernel.inputs.get("index_kv")
    if index_tensor is not None:
        prev_comms["index_kv"] = _make_scatter(index_tensor, n, dim=1)
    for port in ("index_q", "index_weights"):
        tensor = kernel.inputs.get(port)
        if tensor is not None:
            prev_comms[port] = _make_broadcast(tensor, n)
    copies = []
    for _ in range(n):
        if isinstance(kernel, KimiK3MlaAttn):
            copy = KimiK3MlaAttn(
                kernel.B, kernel.H, kernel.S_q, local_kv,
                kernel.qk_head_dim, kernel.v_head_dim,
                kernel.kv_cache_dim, kernel.kv_lora_rank,
                kernel.qk_nope_head_dim, kernel.dtype_,
                kernel.kv_transform_dtype,
                q_dtype=kernel.q_dtype, kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype, causal=kernel.causal,
                selected_pairs=kernel.selected_pairs / n,
                kv_transform_tokens=kernel.kv_transform_tokens / n)
        elif isinstance(kernel, Glm52SparseAttn):
            copy = Glm52SparseAttn(
                kernel.B, kernel.H, kernel.S_q, kernel.k_sel // n,
                local_kv, kernel.qk_head_dim, kernel.v_head_dim,
                kernel.kv_cache_dim, kernel.dtype_,
                kv_lora_rank=kernel.kv_lora_rank,
                qk_nope_head_dim=kernel.qk_nope_head_dim,
                kv_transform_dtype=kernel.kv_transform_dtype,
                indexer_mode=kernel.indexer_mode,
                indexer_s_kv=(kernel.indexer_s_kv // n
                              if kernel.indexer_mode == "full" else 0),
                indexer_h=kernel.indexer_h,
                indexer_hd=kernel.indexer_hd,
                indexer_dtype=kernel.indexer_dtype,
                indexer_compute_dtype=kernel.indexer_compute_dtype,
                indexer_reduce_dtype=kernel.indexer_reduce_dtype,
                q_dtype=kernel.q_dtype,
                kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype,
                index_q_dtype=kernel.index_q_dtype,
                index_weight_dtype=kernel.index_weight_dtype,
                causal=kernel.causal,
                selected_pairs=kernel.selected_pairs / n,
                indexer_pairs=kernel.indexer_pairs / n,
                kv_transform_tokens=kernel.kv_transform_tokens / n)
        else:
            copy = DpskV4SparseAttn(
                kernel.B, kernel.H, kernel.H_kv, kernel.S_q,
                kernel.k_sel // n, local_kv, kernel.Hd, kernel.dtype_,
                kernel.kv_factor,
                indexer_s_kv=kernel.indexer_s_kv // n,
                indexer_h=kernel.indexer_h,
                indexer_hd=kernel.indexer_hd,
                indexer_dtype=kernel.indexer_dtype,
                indexer_compute_dtype=kernel.indexer_compute_dtype,
                q_dtype=kernel.q_dtype,
                kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype,
                causal=kernel.causal,
                causal_k_sel=kernel.causal_k_sel // n)
        copy.inputs = {
            "q": Tensor(q_tensor.dtype, q_tensor.shape),
            "kv": Tensor(
                kv_tensor.dtype, _shard_shape(kv_tensor.shape, n, dim=1)),
        }
        if index_tensor is not None:
            copy.inputs["index_kv"] = Tensor(
                index_tensor.dtype,
                _shard_shape(index_tensor.shape, n, dim=1))
        for port in ("index_q", "index_weights"):
            tensor = kernel.inputs.get(port)
            if tensor is not None:
                copy.inputs[port] = Tensor(tensor.dtype, tensor.shape)
        copy.weights = {
            port: Tensor(tensor.dtype, tensor.shape, tensor.weight_id)
            for port, tensor in kernel.weights.items()
        }
        copy.outputs = {
            "y": Tensor(out_tensor.dtype, out_tensor.shape)}
        copies.append(copy)
    next_comms = {
        "y": _make_reduce(out_tensor, n, dtype=out_tensor.dtype)}
    return prev_comms, copies, next_comms


def kv_persistence_split(kernel, n):
    """Shard KV barrier inputs while replicating the stage output token."""
    output_ports = {
        port for port in kernel.inputs
        if port in ("prefill_output", "decode_output")
    } if isinstance(kernel, Nop) else set()
    if len(output_ports) != 1:
        raise TypeError("KV persistence split requires its barrier Nop")
    output_port = next(iter(output_ports))

    prev_comms = {}
    for port, tensor in kernel.inputs.items():
        if port == output_port:
            prev_comms[port] = _make_broadcast(tensor, n)
        else:
            prev_comms[port] = _make_scatter(tensor, n, dim=1)

    copies = []
    for _ in range(n):
        copy = Nop()
        copy.inputs = {
            port: Tensor(
                tensor.dtype,
                tensor.shape if port == output_port
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


def general_dup(kernel, n):
    """Dup callable that moves a one-input Broadcast before the kernel."""
    if len(kernel.inputs) != 1 or len(kernel.outputs) != 1:
        raise ValueError("general_dup requires one input and one output")
    input_name, input_tensor = next(iter(kernel.inputs.items()))
    broadcast = _make_broadcast(input_tensor, n)
    broadcast.inputs = {
        input_name: Tensor(input_tensor.dtype, input_tensor.shape)}
    copies = [deepcopy(kernel) for _ in range(n)]
    return broadcast, copies


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
                     kernel.a_dtype, kernel.out_dtype,
                     kernel.compute_dtype)
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
                     kernel.a_dtype, kernel.out_dtype,
                     kernel.compute_dtype)
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
    index_tensor = kernel.inputs.get("index_kv")
    if index_tensor is not None:
        prev_comms["index_kv"] = _make_broadcast(index_tensor, n)
    index_q_tensor = kernel.inputs.get("index_q")
    if index_q_tensor is not None:
        prev_comms["index_q"] = _make_scatter(
            index_q_tensor, n, dim=-1)
    index_weight_tensor = kernel.inputs.get("index_weights")
    if index_weight_tensor is not None:
        prev_comms["index_weights"] = _make_scatter(
            index_weight_tensor, n, dim=-1)

    copies = []
    for i in range(n):
        if isinstance(kernel, Glm52SparseAttn):
            c = Glm52SparseAttn(
                kernel.B, shard_h, kernel.S_q, kernel.k_sel, kernel.S_kv,
                kernel.qk_head_dim, kernel.v_head_dim, kernel.kv_cache_dim,
                kernel.dtype_, kv_lora_rank=kernel.kv_lora_rank,
                qk_nope_head_dim=kernel.qk_nope_head_dim,
                kv_transform_dtype=kernel.kv_transform_dtype,
                indexer_mode=kernel.indexer_mode,
                indexer_s_kv=kernel.indexer_s_kv,
                indexer_h=(kernel.indexer_h // n
                           if kernel.indexer_mode == "full" else 0),
                indexer_hd=kernel.indexer_hd,
                indexer_dtype=kernel.indexer_dtype,
                indexer_compute_dtype=kernel.indexer_compute_dtype,
                indexer_reduce_dtype=kernel.indexer_reduce_dtype,
                q_dtype=kernel.q_dtype,
                kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype,
                index_q_dtype=kernel.index_q_dtype,
                index_weight_dtype=kernel.index_weight_dtype,
                causal=kernel.causal,
                selected_pairs=kernel.selected_pairs,
                indexer_pairs=kernel.indexer_pairs,
                kv_transform_tokens=kernel.kv_transform_tokens)
        else:
            c = DpskV4SparseAttn(
                kernel.B, shard_h, kernel.H_kv, kernel.S_q,
                kernel.k_sel, kernel.S_kv, kernel.Hd, kernel.dtype_,
                kernel.kv_factor,
                indexer_s_kv=kernel.indexer_s_kv,
                indexer_h=(kernel.indexer_h // n
                           if kernel.indexer_h else 0),
                indexer_hd=kernel.indexer_hd,
                indexer_dtype=kernel.indexer_dtype,
                indexer_compute_dtype=kernel.indexer_compute_dtype,
                q_dtype=kernel.q_dtype,
                kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype,
                causal=kernel.causal,
                causal_k_sel=kernel.causal_k_sel)
        c.inputs = {"q": Tensor(q_tensor.dtype, shard_q_shape),
                    "kv": Tensor(kv_tensor.dtype, kv_tensor.shape)}
        if index_tensor is not None:
            c.inputs["index_kv"] = Tensor(
                index_tensor.dtype, index_tensor.shape)
        if index_q_tensor is not None:
            c.inputs["index_q"] = Tensor(
                index_q_tensor.dtype,
                _shard_shape(index_q_tensor.shape, n, dim=-1))
        if index_weight_tensor is not None:
            c.inputs["index_weights"] = Tensor(
                index_weight_tensor.dtype,
                _shard_shape(index_weight_tensor.shape, n, dim=-1))
        if isinstance(kernel, Glm52SparseAttn) and kernel.weights:
            kv_out = shard_h * (
                kernel.qk_nope_head_dim + kernel.v_head_dim)
            c.weights = {
                "kv_b": Tensor(
                    kernel.kv_transform_dtype,
                    (kernel.kv_lora_rank, kv_out)),
            }
            scale_bytes = gemm_scale_bytes(
                kv_out, kernel.kv_lora_rank, kernel.kv_transform_dtype)
            if scale_bytes > 0:
                c.weights["kv_b_scale"] = Tensor(
                    "ue8m0", (int(scale_bytes),))
            _propagate_weight_id(kernel.weights, c.weights, i, "column")
        c.outputs = {
            "y": Tensor(
                out_tensor.dtype,
                _shard_shape(out_tensor.shape, n, dim=-1))}
        copies.append(c)

    next_comms = {"y": _make_gather(out_tensor, n, dim=-1)}

    return prev_comms, copies, next_comms


# ── context_split ─────────────────────────────────────────────────────


def context_split_prefill(kernel, n):
    """Apply context parallelism using the prefill attention strategy."""
    return _context_split(kernel, n, is_prefill=True)


def context_split_decode(kernel, n):
    """Apply context parallelism using the decode attention strategy."""
    return _context_split(kernel, n, is_prefill=False)


def _context_split(kernel, n, *, is_prefill):
    """Shard token work along sequence dim 1 while preserving batch size.

    Expert-token tensors produced by TokenDispatch and consumed by
    TokenCombine are flattened as (M_e, D), so those ports shard dim 0.
    Prefill attention shards Q and receives logical full KV through AllGather;
    decode attention broadcasts Q and keeps KV sharded.
    """
    if isinstance(
            kernel, (Attn, DpskV4SparseAttn, Glm52SparseAttn,
                     KimiK3MlaAttn)):
        if is_prefill:
            return _context_split_attn_prefill(kernel, n)
        else:
            return _context_split_attn_decode(kernel, n)

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


def _context_split_attn_prefill(kernel, n):
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
    index_tensor = kernel.inputs.get("index_kv")
    if index_tensor is not None:
        prev_comms["index_kv"] = _make_broadcast(index_tensor, n)
    for port in ("index_q", "index_weights"):
        tensor = kernel.inputs.get(port)
        if tensor is not None:
            prev_comms[port] = _make_scatter(tensor, n, dim=1)

    copies = []
    for _ in range(n):
        if isinstance(kernel, KimiK3MlaAttn):
            c = KimiK3MlaAttn(
                kernel.B, kernel.H, kernel.S_q // n, kernel.S_kv,
                kernel.qk_head_dim, kernel.v_head_dim,
                kernel.kv_cache_dim, kernel.kv_lora_rank,
                kernel.qk_nope_head_dim, kernel.dtype_,
                kernel.kv_transform_dtype,
                q_dtype=kernel.q_dtype, kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype, causal=kernel.causal,
                selected_pairs=kernel.selected_pairs / n,
                kv_transform_tokens=kernel.kv_transform_tokens / n)
        elif isinstance(kernel, Glm52SparseAttn):
            c = Glm52SparseAttn(
                kernel.B, kernel.H, kernel.S_q // n, kernel.k_sel,
                kernel.S_kv, kernel.qk_head_dim, kernel.v_head_dim,
                kernel.kv_cache_dim, kernel.dtype_,
                kv_lora_rank=kernel.kv_lora_rank,
                qk_nope_head_dim=kernel.qk_nope_head_dim,
                kv_transform_dtype=kernel.kv_transform_dtype,
                indexer_mode=kernel.indexer_mode,
                indexer_s_kv=kernel.indexer_s_kv,
                indexer_h=kernel.indexer_h,
                indexer_hd=kernel.indexer_hd,
                indexer_dtype=kernel.indexer_dtype,
                indexer_compute_dtype=kernel.indexer_compute_dtype,
                indexer_reduce_dtype=kernel.indexer_reduce_dtype,
                q_dtype=kernel.q_dtype,
                kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype,
                index_q_dtype=kernel.index_q_dtype,
                index_weight_dtype=kernel.index_weight_dtype,
                causal=kernel.causal,
                selected_pairs=kernel.selected_pairs / n,
                indexer_pairs=kernel.indexer_pairs / n,
                kv_transform_tokens=kernel.kv_transform_tokens / n)
        elif isinstance(kernel, DpskV4SparseAttn):
            c = DpskV4SparseAttn(
                kernel.B, kernel.H, kernel.H_kv, kernel.S_q // n,
                kernel.k_sel, kernel.S_kv, kernel.Hd, kernel.dtype_,
                kernel.kv_factor,
                indexer_s_kv=kernel.indexer_s_kv,
                indexer_h=kernel.indexer_h,
                indexer_hd=kernel.indexer_hd,
                indexer_dtype=kernel.indexer_dtype,
                indexer_compute_dtype=kernel.indexer_compute_dtype,
                q_dtype=kernel.q_dtype,
                kv_dtype=kernel.kv_dtype,
                out_dtype=kernel.out_dtype,
                causal=kernel.causal,
                causal_k_sel=kernel.causal_k_sel)
        else:
            c = Attn(
                kernel.B, kernel.H, kernel.H_kv, kernel.S_q // n,
                kernel.S_kv, kernel.Hd, kernel.dtype_, kernel.causal,
                triangular=kernel.triangular)
        c.inputs = {
            "q": Tensor(q_tensor.dtype, q_shard),
            "kv": Tensor(kv_tensor.dtype, kv_tensor.shape),
        }
        if index_tensor is not None:
            c.inputs["index_kv"] = Tensor(
                index_tensor.dtype, index_tensor.shape)
        for port in ("index_q", "index_weights"):
            tensor = kernel.inputs.get(port)
            if tensor is not None:
                c.inputs[port] = Tensor(
                    tensor.dtype, _shard_shape(tensor.shape, n, dim=1))
        c.weights = {
            port: Tensor(tensor.dtype, tensor.shape, tensor.weight_id)
            for port, tensor in kernel.weights.items()
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
                 kernel.a_dtype, kernel.out_dtype, kernel.compute_dtype)
    elif isinstance(kernel, RMSNorm):
        c = RMSNorm(kernel.M // n, kernel.D, kernel.dtype_)
    elif isinstance(kernel, PartialRMSNorm):
        c = PartialRMSNorm(
            kernel.M // n, kernel.input_dim, kernel.norm_dim,
            kernel.dtype_)
    elif isinstance(kernel, LayerNorm):
        c = LayerNorm(kernel.M // n, kernel.D, kernel.dtype_)
    elif isinstance(kernel, AttnRes):
        c = AttnRes(
            kernel.B, kernel.S // n, kernel.D, kernel.R,
            kernel.storage_dtype, kernel.dtype_)
    elif isinstance(kernel, KimiK3DeltaAttn):
        c = KimiK3DeltaAttn(
            kernel.B, kernel.H, kernel.S // n, kernel.K, kernel.V,
            kernel.mode, kernel.chunk_size, kernel.conv_size,
            kernel.dtype_, kernel.state_dtype)
    elif isinstance(kernel, KdaStateStore):
        c = KdaStateStore(
            kernel.B, kernel.H // n, kernel.S // n,
            kernel.K, kernel.V, kernel.conv_size, kernel.state_dtype)
    elif isinstance(kernel, Embedding):
        c = Embedding(kernel.M // n, kernel.V, kernel.D, kernel.w_dtype,
                      kernel.idx_dtype, kernel.out_dtype)
    elif isinstance(kernel, ReadInput):
        c = ReadInput(kernel.n_elements // n, kernel.dtype_)
    elif isinstance(kernel, TokenDispatch):
        c = TokenDispatch(kernel.M // n, kernel.D, kernel.N_experts,
                          kernel.topk, kernel.a_dtype, kernel.scoring_func)
    elif isinstance(kernel, TokenCombine):
        c = TokenCombine(kernel.M // n, kernel.D, kernel.N_experts,
                         kernel.topk, kernel.a_dtype)
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
                 kernel.a_dtype, kernel.out_dtype, kernel.compute_dtype)
    elif isinstance(kernel, RMSNorm):
        c = RMSNorm(kernel.M // n, kernel.D, kernel.dtype_)
    elif isinstance(kernel, PartialRMSNorm):
        c = PartialRMSNorm(
            kernel.M // n, kernel.input_dim, kernel.norm_dim,
            kernel.dtype_)
    elif isinstance(kernel, LayerNorm):
        c = LayerNorm(kernel.M // n, kernel.D, kernel.dtype_)
    elif isinstance(kernel, AttnRes):
        c = AttnRes(
            kernel.B // n, kernel.S, kernel.D, kernel.R,
            kernel.storage_dtype, kernel.dtype_)
    elif isinstance(kernel, KimiK3MlaAttn):
        c = KimiK3MlaAttn(
            kernel.B // n, kernel.H, kernel.S_q, kernel.S_kv,
            kernel.qk_head_dim, kernel.v_head_dim, kernel.kv_cache_dim,
            kernel.kv_lora_rank, kernel.qk_nope_head_dim, kernel.dtype_,
            kernel.kv_transform_dtype,
            q_dtype=kernel.q_dtype, kv_dtype=kernel.kv_dtype,
            out_dtype=kernel.out_dtype, causal=kernel.causal,
            selected_pairs=kernel.selected_pairs,
            kv_transform_tokens=kernel.kv_transform_tokens)
    elif isinstance(kernel, KimiK3DeltaAttn):
        c = KimiK3DeltaAttn(
            kernel.B // n, kernel.H, kernel.S, kernel.K, kernel.V,
            kernel.mode, kernel.chunk_size, kernel.conv_size,
            kernel.dtype_, kernel.state_dtype)
    elif isinstance(kernel, KdaStateStore):
        c = KdaStateStore(
            kernel.B // n, kernel.H, kernel.S,
            kernel.K, kernel.V, kernel.conv_size, kernel.state_dtype)
    elif isinstance(kernel, Embedding):
        c = Embedding(kernel.M // n, kernel.V, kernel.D, kernel.w_dtype,
                      kernel.idx_dtype, kernel.out_dtype)
    elif isinstance(kernel, ReadInput):
        c = ReadInput(kernel.n_elements // n, kernel.dtype_)
    elif isinstance(kernel, TokenDispatch):
        c = TokenDispatch(kernel.M // n, kernel.D, kernel.N_experts,
                          kernel.topk, kernel.a_dtype, kernel.scoring_func)
    elif isinstance(kernel, TokenCombine):
        c = TokenCombine(kernel.M // n, kernel.D, kernel.N_experts,
                         kernel.topk, kernel.a_dtype)
    elif isinstance(kernel, DpskV4SparseAttn):
        c = DpskV4SparseAttn(kernel.B // n, kernel.H, kernel.H_kv, kernel.S_q,
                       kernel.k_sel, kernel.S_kv, kernel.Hd, kernel.dtype_,
                       kernel.kv_factor,
                       indexer_s_kv=kernel.indexer_s_kv,
                       indexer_h=kernel.indexer_h,
                       indexer_hd=kernel.indexer_hd,
                       indexer_dtype=kernel.indexer_dtype,
                       indexer_compute_dtype=kernel.indexer_compute_dtype,
                       q_dtype=kernel.q_dtype,
                       kv_dtype=kernel.kv_dtype,
                       out_dtype=kernel.out_dtype,
                       causal=kernel.causal,
                       causal_k_sel=kernel.causal_k_sel)
    elif isinstance(kernel, Glm52SparseAttn):
        c = Glm52SparseAttn(
            kernel.B // n, kernel.H, kernel.S_q, kernel.k_sel, kernel.S_kv,
            kernel.qk_head_dim, kernel.v_head_dim, kernel.kv_cache_dim,
            kernel.dtype_, kv_lora_rank=kernel.kv_lora_rank,
            qk_nope_head_dim=kernel.qk_nope_head_dim,
            kv_transform_dtype=kernel.kv_transform_dtype,
            indexer_mode=kernel.indexer_mode,
            indexer_s_kv=kernel.indexer_s_kv,
            indexer_h=kernel.indexer_h,
            indexer_hd=kernel.indexer_hd,
            indexer_dtype=kernel.indexer_dtype,
            indexer_compute_dtype=kernel.indexer_compute_dtype,
            indexer_reduce_dtype=kernel.indexer_reduce_dtype,
            q_dtype=kernel.q_dtype,
            kv_dtype=kernel.kv_dtype,
            out_dtype=kernel.out_dtype,
            index_q_dtype=kernel.index_q_dtype,
            index_weight_dtype=kernel.index_weight_dtype,
            causal=kernel.causal,
            selected_pairs=kernel.selected_pairs,
            indexer_pairs=kernel.indexer_pairs,
            kv_transform_tokens=kernel.kv_transform_tokens)
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
