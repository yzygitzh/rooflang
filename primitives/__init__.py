"""primitives — closed-form FLOPs / HBM-byte formulas split by phase:

  forward.py    — forward kernels (GEMM, norms, RoPE, attention, sparse-attn)
  backward.py   — gradient kernels (dX, dW, attention backward, …)
  optimizer.py  — optimizer-step kernels (AdamW, …)  [pending]

Import directly from the submodule of interest:
    from primitives.forward  import gemm_flops_bytes, attn_flops_bytes
    from primitives.backward import gemm_dx_flops_bytes, attn_backward_flops_bytes
"""
