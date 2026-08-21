"""kernels — Kernel classes for closed-form FLOPs / HBM-byte formulas,
split by phase:

  kernel.py     — Kernel base class + TensorDesc
  forward.py    — Slice, Gemm, StridedGemm, RMSNorm, LayerNorm, RoPE,
                  Attn, DpskV4SparseAttn
  backward.py   — GemmDX, GemmDW, StridedGemmDX, StridedGemmDW,
                   RMSNorm, LayerNorm, RoPE, Attn, DpskV4SparseAttn
  optimizer.py  — AdamWStep
  comm.py       — AllReduce, ReduceScatter, AllGather, AllToAll, Broadcast, SendRecv
  identity.py   — Spawn, Concat

Forward and backward share class names; disambiguate via module:
    from rooflang.language.kernels import forward, backward
    fwd = forward.Gemm(4096, 4096, 4096, 'fp8', 'fp8')
    bwd_dx = backward.GemmDX(4096, 4096, 4096, 'fp8', 'fp8')
    bwd_norm = backward.RMSNorm(4096, 7168)
"""

from rooflang.language.kernels.kernel import Kernel  # noqa: F401
from rooflang.language.tensor import Tensor  # noqa: F401
