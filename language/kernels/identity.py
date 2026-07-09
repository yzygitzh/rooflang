"""Identity kernels — explicit data movement between memory tiers."""

from __future__ import annotations

from rooflang.language.kernels.kernel import Kernel, TensorDesc


class Move(Kernel):
    """Move a tensor to a different memory location. Side-effect kernel.

    Covers all data movement: HBM→NVMe (offload), NVMe→HBM (prefetch),
    HBM→DRAM, DRAM→HBM, etc.
    """

    def __init__(self, tensor: TensorDesc, dst_location: str) -> None:
        dst = TensorDesc(dtype=tensor.dtype, shape=tensor.shape,
                         location=dst_location)
        super().__init__(
            inputs={"src": tensor},
            outputs={"dst": dst},
        )
