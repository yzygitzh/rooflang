"""kernels — Kernel classes for closed-form FLOPs / HBM-byte formulas,
split by phase:

  kernel.py     — Kernel base class + TensorDesc
  forward.py    — Gemm, RMSNorm, LayerNorm, RoPE, Attn, SparseAttn
  backward.py   — GemmDX, GemmDW, RMSNorm, LayerNorm, RoPE, Attn, SparseAttn
  optimizer.py  — AdamWStep
  comm.py       — AllReduce, ReduceScatter, AllGather, AllToAll, Broadcast, SendRecv
  identity.py   — Move (data movement between memory tiers)

Forward and backward share class names; disambiguate via module:
    from rooflang.language.kernels import forward, backward
    fwd = forward.Gemm(4096, 4096, 4096, 'fp8', 'fp8')
    bwd_dx = backward.GemmDX(4096, 4096, 4096, 'fp8', 'fp8')
    bwd_norm = backward.RMSNorm(4096, 7168)
"""

from rooflang.language.kernels.kernel import Kernel, TensorDesc  # noqa: F401
from rooflang.language.kernels.identity import Move  # noqa: F401
