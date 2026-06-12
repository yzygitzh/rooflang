"""primitives — Kernel classes for closed-form FLOPs / HBM-byte formulas,
split by phase:

  kernel.py     — Kernel base class + FusedKernel / OverlappedKernel compositors
  forward.py    — Gemm, RMSNorm, LayerNorm, RoPE, Attn, SparseAttn
  backward.py   — GemmDX, GemmDW, RMSNorm, LayerNorm, RoPE, Attn, SparseAttn
  optimizer.py  — AdamWStep

Forward and backward share class names; disambiguate via module:
    from primitives import forward, backward
    fwd = forward.Gemm(4096, 4096, 4096, 'fp8', 'fp8')
    bwd_dx = backward.GemmDX(4096, 4096, 4096, 'fp8', 'fp8')
    bwd_norm = backward.RMSNorm(4096, 7168)
"""
