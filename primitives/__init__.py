"""primitives — Kernel classes for closed-form FLOPs / HBM-byte formulas,
split by phase:

  kernel.py     — Kernel base class + FusedKernel / OverlappedKernel compositors
  forward.py    — Gemm, RMSNorm, LayerNorm, RoPE, Attn, SparseAttn
  backward.py   — gradient kernels (dX, dW, attention backward, …) [dict, pending migration]
  optimizer.py  — AdamW step [dict, pending migration]

Import directly from the submodule of interest:
    from primitives.kernel  import Kernel, FusedKernel, OverlappedKernel
    from primitives.forward import Gemm, Attn
"""
