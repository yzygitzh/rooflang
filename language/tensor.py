"""Tensor — a shaped, typed array with optional memory location."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

from rooflang.language.utils import dtype_bytes

if TYPE_CHECKING:
    from rooflang.language.hardware.component import Memory


@dataclass
class Tensor:
    """Descriptor for a tensor slot (input, weight, or output).

    Before placement, location is None. After placement, it points to
    the Memory node where this tensor resides.
    """
    dtype: str
    shape: Tuple[int, ...]
    location: Memory | None = None

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def size_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype)
