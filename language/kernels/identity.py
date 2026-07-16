"""Identity kernels — zero-cost structural nodes and data movement."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor

if TYPE_CHECKING:
    from rooflang.language.hardware.component import Memory


class Spawn(Kernel):
    """One-to-many data dependency placeholder with zero cost.

    Used by split_kernel prev_comm when no actual communication is needed
    (e.g., input already replicated/sharded from a prior operation).
    The simulator treats this as an instant pass-through.
    """

    _requires_placement = False

    def __init__(self, world: int):
        self.world = world
        super().__init__()

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def transferred_bytes(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return 0.0

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 0.0


class Move(Kernel):
    """Move a tensor to a different memory location.

    Covers all data movement: HBM→NVMe (offload), NVMe→HBM (prefetch),
    HBM→DRAM, DRAM→HBM, etc.
    """

    def __init__(self, tensor: Tensor, dst_location: Memory) -> None:
        self.dst_location = dst_location
        dst = Tensor(dtype=tensor.dtype, shape=tensor.shape)
        super().__init__(
            inputs={"src": tensor},
            outputs={"dst": dst},
        )
