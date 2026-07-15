"""Tensor — a shaped, typed array descriptor."""

from __future__ import annotations

from typing import Tuple

from rooflang.language.utils import dtype_bytes


class Tensor:
    """Descriptor for a tensor slot (input, weight, or output)."""

    def __init__(self, dtype: str, shape: Tuple[int, ...]) -> None:
        self.dtype = dtype
        self.shape = shape

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def size_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype)
