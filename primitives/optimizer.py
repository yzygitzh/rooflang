"""Closed-form FLOPs and HBM-byte formulas for optimizer-step kernels,
paired with primitives/forward.py and primitives/backward.py.

Returns the same five-key dict as the forward / backward primitives,
built via the shared `_kernel_result` helper:
  {flops, transferred_bytes, input_bytes, weight_bytes, output_bytes}.

The optimizer step is per-parameter — `n_param` is the total parameter
count owned by this rank (after TP / ZeRO / EP sharding).

Bytes are accounted with three dtype knobs:
  - param_dtype   : storage of the optimizer's master parameter copy and
                    the write-back. Standard mixed-precision recipes hold
                    master weights in fp32 here even when the forward
                    weight is bf16 / fp8 / fp4.
  - grad_dtype    : dtype of the incoming gradient (read-only).
  - moment_dtype  : storage of AdamW's m, v buffers. Standard recipes use
                    fp32; "8-bit Adam" / fp8 Adam variants pass "fp8".

Categorisation:
  - input_bytes : the gradient `g` (the upstream signal consumed by the step).
  - weight_bytes: persistent state reads (p, m, v).
  - output_bytes: persistent state writes (p, m, v).

Defaults reflect the conservative recipe (everything in fp32). The op
enumerator passes recipe-specific dtypes per parameter group.
"""

from typing import Dict
from .forward import dtype_bytes, _kernel_result


def adamw_step_flops_bytes(n_param: int,
                           param_dtype:  str = "fp32",
                           grad_dtype:   str = "fp32",
                           moment_dtype: str = "fp32",
                           ) -> Dict[str, float]:
    """One AdamW step over n_param parameters owned by this rank.

    AdamW update (Loshchilov & Hutter 2017, decoupled weight decay):

        g²        = g · g
        m         = β1·m + (1 - β1)·g
        v         = β2·v + (1 - β2)·g²
        denom     = sqrt(v) + ε
        update    = m / denom
        p         = (1 - lr·wd)·p − lr_t·update

      Bias-correction scalars (1−β1^t, 1−β2^t) are folded into a single
      pre-computed lr_t at step time — they amortize to O(1) over n_param.

    flops (per parameter, by direct count of the formula above):
      - g²:                                       1 mult
      - m update:    β1·m + (1-β1)·g              2 mults + 1 add = 3
      - v update:    β2·v + (1-β2)·g²             2 mults + 1 add = 3
      - sqrt(v) + ε:                              1 sqrt + 1 add  = 2
      - update = m / denom:                       1 div           = 1
      - decoupled decay step:                     2 mults + 1 sub = 3
                    (1 - lr·wd)·p − lr_t·update
      Total = 13 flops per parameter.

      This is a direct count of the formula above, not a literature
      citation; published references vary widely (~9–16 flops/param)
      depending on FMA fusion and how sqrt / div are counted. The exact
      constant barely matters here: AdamW is memory-bound by ~10×, so
      the binding constraint on the roofline is HBM bandwidth, not peak
      TFLOPS. (AI ≈ 13 / 24 ≈ 0.54 flops/byte → roofline ceiling caps at
      AI · HBM_BW, far below peak TFLOPS; expect MFU < 1% at perfect
      efficiency. Read the optimizer-step row by its memory utilization,
      not its MFU.)

    bytes (per parameter, fused single-pass):
      input_bytes  = n_param · sizeof(grad_dtype)        (read g)
      weight_bytes = n_param · (sizeof(param) + 2·sizeof(moment))
                     (read p, m, v — persistent optimizer state)
      output_bytes = n_param · (sizeof(param) + 2·sizeof(moment))
                     (write p, m, v back)
      transferred_bytes = n_param · (sizeof(grad) + 2·sizeof(param) + 4·sizeof(moment)).

      Optimizer state is dominant in bytes: with the default fp32-everything
      recipe, AdamW state alone touches 28 B per parameter — comparable to
      forward HBM traffic for a transformer block at moderate batch size,
      which is why optimizer ZeRO-1 / fused kernels matter for training MFU.
    """
    p_b = dtype_bytes(param_dtype)
    g_b = dtype_bytes(grad_dtype)
    m_b = dtype_bytes(moment_dtype)
    state_bytes  = n_param * (p_b + 2.0 * m_b)        # one read or one write
    input_bytes  = n_param * g_b
    weight_bytes = state_bytes
    output_bytes = state_bytes
    return _kernel_result(13.0 * n_param, input_bytes, weight_bytes, output_bytes)
