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
from rooflang.language.kernels.forward import Gemm, SparseAttn, StridedGemm
from rooflang.language.tensor import Tensor
from rooflang.language.utils import dtype_bytes, gemm_scale_bytes


def column_split(kernel, n):
    """Non-contracting output dim (N): Broadcast → Kernels → Gather.

    Input is replicated (Broadcast) to all ranks; output shards are
    concatenated (Gather). Works for both Gemm and StridedGemm.
    """
    shard_n = kernel.N // n

    # Read actual tensor shapes from kernel
    in_tensor = kernel.inputs["x"]
    in_shape = in_tensor.shape
    in_dtype = in_tensor.dtype
    in_bytes = in_tensor.size_bytes

    # prev_comms: Broadcast the input to all ranks (unchanged)
    prev_comms = {}
    prev_comms["x"] = Broadcast(total_bytes=in_bytes, world=n)
    prev_comms["x"].inputs = {"x": Tensor(in_dtype, in_shape)}
    prev_comms["x"].outputs = {f"o{i}": Tensor(in_dtype, in_shape)
                               for i in range(n)}

    # copies: split along N dimension
    copies = []
    if isinstance(kernel, StridedGemm):
        shard_out_elems = kernel._out_elems // n
        for _ in range(n):
            c = StridedGemm(kernel.M, shard_n, kernel.K, kernel.w_dtype,
                            kernel.a_dtype, kernel.out_dtype,
                            in_elems=kernel._in_elems, out_elems=shard_out_elems)
            c.inputs = {"x": Tensor(in_dtype, in_shape)}
            c.weights = {"w": Tensor(kernel.w_dtype, (kernel.K * shard_n,))}
            scale_bytes = gemm_scale_bytes(shard_n, kernel.K, kernel.w_dtype)
            if scale_bytes > 0:
                c.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            c.outputs = {"y": Tensor(kernel.out_dtype, (shard_out_elems,))}
            copies.append(c)
    else:
        for _ in range(n):
            c = Gemm(kernel.M, shard_n, kernel.K, kernel.w_dtype,
                     kernel.a_dtype, kernel.out_dtype)
            c.inputs = {"x": Tensor(in_dtype, in_shape)}
            c.weights = {"w": Tensor(kernel.w_dtype, (kernel.K * shard_n,))}
            scale_bytes = gemm_scale_bytes(shard_n, kernel.K, kernel.w_dtype)
            if scale_bytes > 0:
                c.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
            c.outputs = {"y": Tensor(kernel.out_dtype, (kernel.M * shard_n,))}
            copies.append(c)

    # next_comms: Gather output shards
    out_tensor = kernel.outputs["y"]
    out_bytes = out_tensor.size_bytes
    out_dtype = out_tensor.dtype
    shard_shape = copies[0].outputs["y"].shape
    next_comms = {}
    next_comms["y"] = Gather(total_bytes=out_bytes, world=n)
    next_comms["y"].inputs = {f"i{i}": Tensor(out_dtype, shard_shape)
                              for i in range(n)}
    next_comms["y"].outputs = {"y": Tensor(out_dtype, out_tensor.shape)}

    return prev_comms, copies, next_comms


def row_split(kernel, n):
    """Contracting dim (K): Scatter → Kernels → Reduce.

    Input is partitioned (Scatter) across ranks; partial sums are
    reduced (Reduce).
    """
    shard_k = kernel.K // n

    # Read actual tensor shapes from kernel
    in_tensor = kernel.inputs["x"]
    in_shape = in_tensor.shape
    in_dtype = in_tensor.dtype
    in_bytes = in_tensor.size_bytes

    # Shard input along K
    in_elems = in_shape[0]
    shard_in_elems = in_elems // n

    # prev_comms: Scatter input into K-shards
    prev_comms = {}
    prev_comms["x"] = Scatter(total_bytes=in_bytes, world=n)
    prev_comms["x"].inputs = {"x": Tensor(in_dtype, in_shape)}
    prev_comms["x"].outputs = {f"o{i}": Tensor(in_dtype, (shard_in_elems,))
                               for i in range(n)}

    # copies
    out_tensor = kernel.outputs["y"]
    out_dtype = out_tensor.dtype
    out_shape = out_tensor.shape
    copies = []
    for _ in range(n):
        c = Gemm(kernel.M, kernel.N, shard_k, kernel.w_dtype,
                 kernel.a_dtype, kernel.out_dtype)
        c.inputs = {"x": Tensor(in_dtype, (shard_in_elems,))}
        c.weights = {"w": Tensor(kernel.w_dtype, (shard_k * kernel.N,))}
        scale_bytes = gemm_scale_bytes(kernel.N, shard_k, kernel.w_dtype)
        if scale_bytes > 0:
            c.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
        c.outputs = {"y": Tensor(out_dtype, out_shape)}
        copies.append(c)

    # next_comms: Reduce partial sums
    out_bytes = out_tensor.size_bytes
    next_comms = {}
    next_comms["y"] = Reduce(total_bytes=out_bytes, world=n)
    next_comms["y"].inputs = {f"i{i}": Tensor(out_dtype, out_shape)
                              for i in range(n)}
    next_comms["y"].outputs = {"y": Tensor(out_dtype, out_shape)}

    return prev_comms, copies, next_comms


def head_split(kernel, n):
    """Non-contracting input dim (H): per-port Scatter/Broadcast → Kernels → Gather.

    Q input is partitioned (Scatter) by head; KV input is broadcast
    (same KV cache to each rank). Output shards are concatenated (Gather).
    """
    shard_h = kernel.H // n
    q_elems = kernel.B * shard_h * kernel.S_q * kernel.Hd
    kv_elems = kernel.kv_factor * kernel.B * kernel.H_kv * kernel.S_kv * kernel.Hd
    out_elems = kernel.B * shard_h * kernel.S_q * kernel.Hd
    b = dtype_bytes(kernel.dtype_)

    # Read actual tensor shapes
    q_tensor = kernel.inputs["q"]
    kv_tensor = kernel.inputs["kv"]
    full_q_elems = q_tensor.shape[0]

    # prev_comms: Scatter for Q, Broadcast for KV
    prev_comms = {}
    prev_comms["q"] = Scatter(total_bytes=q_tensor.size_bytes, world=n)
    prev_comms["q"].inputs = {"x": Tensor(kernel.dtype_, q_tensor.shape)}
    prev_comms["q"].outputs = {f"o{i}": Tensor(kernel.dtype_, (q_elems,))
                               for i in range(n)}

    prev_comms["kv"] = Broadcast(total_bytes=kv_tensor.size_bytes, world=n)
    prev_comms["kv"].inputs = {"x": Tensor(kernel.dtype_, kv_tensor.shape)}
    prev_comms["kv"].outputs = {f"o{i}": Tensor(kernel.dtype_, (kv_elems,))
                                for i in range(n)}

    # copies
    copies = []
    for _ in range(n):
        c = SparseAttn(kernel.B, shard_h, kernel.H_kv, kernel.S_q,
                       kernel.k_sel, kernel.S_kv, kernel.Hd, kernel.dtype_,
                       kernel.kv_factor)
        c.inputs = {"q": Tensor(kernel.dtype_, (q_elems,)),
                    "kv": Tensor(kernel.dtype_, (kv_elems,))}
        c.outputs = {"y": Tensor(kernel.dtype_, (out_elems,))}
        copies.append(c)

    # next_comms: Gather output
    out_tensor = kernel.outputs["y"]
    next_comms = {}
    next_comms["y"] = Gather(total_bytes=out_tensor.size_bytes, world=n)
    next_comms["y"].inputs = {f"i{i}": Tensor(kernel.dtype_, (out_elems,))
                              for i in range(n)}
    next_comms["y"].outputs = {"y": Tensor(kernel.dtype_, out_tensor.shape)}

    return prev_comms, copies, next_comms
