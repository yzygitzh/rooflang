# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Tensor — a shaped, typed array descriptor."""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple

from rooflang.language.utils import dtype_bytes


class Tensor:
    """Descriptor for a tensor slot (input, weight, or output)."""

    def __init__(self, dtype: str, shape: Tuple[int | Fraction, ...],
                 weight_id: str = None) -> None:
        self.dtype = dtype
        self.shape = shape
        self.weight_id = weight_id

    @property
    def n_elements(self) -> int | Fraction:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def size_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype)
