"""Identity kernels — explicit data movement between memory tiers."""

from __future__ import annotations

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor


class Move(Kernel):
    """Move a tensor to a different memory location.

    Covers all data movement: HBM→NVMe (offload), NVMe→HBM (prefetch),
    HBM→DRAM, DRAM→HBM, etc.
    """

    def __init__(self, tensor: Tensor, dst_location) -> None:
        dst = Tensor(dtype=tensor.dtype, shape=tensor.shape,
                     location=dst_location)
        super().__init__(
            inputs={"src": tensor},
            outputs={"dst": dst},
        )
