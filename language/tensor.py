"""Tensor — a named, shaped, typed array located in a specific Memory."""

from dataclasses import dataclass
from typing import Tuple

from rooflang.language.hardware.component import Memory
from rooflang.language.primitives.utils import dtype_bytes


@dataclass
class Tensor:
    """A tensor living in a specific memory node.

    Kernels consume and produce Tensors. The location determines which
    Memory node's capacity is consumed.
    """
    name: str
    shape: Tuple[int, ...]
    dtype: str
    location: Memory

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def size_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype)
