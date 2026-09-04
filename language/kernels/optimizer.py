# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Optimizer-step Kernel subclasses, paired with forward.py and backward.py.

The optimizer step is per-parameter — `n_param` is the total parameter
count owned by this rank (after TP / ZeRO / EP sharding).

Bytes are accounted with three dtype knobs:
  - param_dtype   : storage of the optimizer's master parameter copy and
                    the write-back.
  - grad_dtype    : dtype of the incoming gradient (read-only).
  - moment_dtype  : storage of AdamW's m, v buffers.

Categorisation:
  - input_bytes : the gradient `g` (the upstream signal consumed by the step).
  - weight_bytes: persistent state reads (p, m, v).
  - output_bytes: persistent state writes (p, m, v).

Defaults reflect the conservative recipe (everything in fp32). The op
enumerator passes recipe-specific dtypes per parameter group.
"""

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.utils import dtype_bytes


class AdamWStep(Kernel):
    """One AdamW step over n_param parameters owned by this rank.

    AdamW update (Loshchilov & Hutter 2017, decoupled weight decay):

        g²        = g · g
        m         = β1·m + (1 - β1)·g
        v         = β2·v + (1 - β2)·g²
        denom     = sqrt(v) + ε
        update    = m / denom
        p         = (1 - lr·wd)·p − lr_t·update

    flops (per parameter, by direct count of the formula above):
      - g²:                                       1 mult
      - m update:    β1·m + (1-β1)·g              2 mults + 1 add = 3
      - v update:    β2·v + (1-β2)·g²             2 mults + 1 add = 3
      - sqrt(v) + ε:                              1 sqrt + 1 add  = 2
      - update = m / denom:                       1 div           = 1
      - decoupled decay step:                     2 mults + 1 sub = 3
                    (1 - lr·wd)·p − lr_t·update
      Total = 13 flops per parameter.

    bytes (per parameter, fused single-pass):
      input_bytes  = n_param · sizeof(grad_dtype)        (read g)
      weight_bytes = n_param · (sizeof(param) + 2·sizeof(moment))
                     (read p, m, v)
      output_bytes = n_param · (sizeof(param) + 2·sizeof(moment))
                     (write p, m, v)
    """

    def __init__(self, n_param: int,
                 param_dtype: str = "fp32",
                 grad_dtype: str = "fp32",
                 moment_dtype: str = "fp32"):
        self.n_param = n_param
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.moment_dtype = moment_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 13.0 * self.n_param

    @property
    def input_bytes(self) -> float:
        return self.n_param * dtype_bytes(self.grad_dtype)

    @property
    def weight_bytes(self) -> float:
        return self.n_param * (dtype_bytes(self.param_dtype)
                               + 2.0 * dtype_bytes(self.moment_dtype))

    @property
    def output_bytes(self) -> float:
        return self.n_param * (dtype_bytes(self.param_dtype)
                               + 2.0 * dtype_bytes(self.moment_dtype))
