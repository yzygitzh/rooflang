"""Base Kernel class for the model-roofline DSL.

Every primitive (forward, backward, optimizer, comm, identity) is a Kernel
subclass carrying roofline metrics as @property methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from rooflang.language.kernels.utils import dtype_bytes


@dataclass
class TensorDesc:
    """Descriptor for a named tensor slot (input, weight, or output)."""
    dtype: str
    shape: Tuple[int, ...]
    location: str | None = None

    @property
    def size_bytes(self) -> float:
        n = 1
        for d in self.shape:
            n *= d
        return n * dtype_bytes(self.dtype)


class Kernel:
    """Base: one primitive or composite with its roofline metrics.

    Byte properties (input_bytes, weight_bytes, output_bytes, transferred_bytes)
    are computed from the inputs/weights/outputs dicts by default. Subclasses
    can override these properties for custom accounting.
    """

    def __init__(
        self,
        inputs: Dict[str, TensorDesc] | None = None,
        weights: Dict[str, TensorDesc] | None = None,
        outputs: Dict[str, TensorDesc] | None = None,
        has_side_effect: bool = False,
    ) -> None:
        self.inputs: Dict[str, TensorDesc] = inputs or {}
        self.weights: Dict[str, TensorDesc] = weights or {}
        self.outputs: Dict[str, TensorDesc] = outputs or {}
        self.has_side_effect = has_side_effect

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return sum(t.size_bytes for t in self.inputs.values())

    @property
    def weight_bytes(self) -> float:
        return sum(t.size_bytes for t in self.weights.values())

    @property
    def output_bytes(self) -> float:
        return sum(t.size_bytes for t in self.outputs.values())

    @property
    def transferred_bytes(self) -> float:
        return self.input_bytes + self.weight_bytes + self.output_bytes

    def to_dict(self) -> dict:
        d = {
            "flops":             self.flops,
            "transferred_bytes": self.transferred_bytes,
            "input_bytes":       self.input_bytes,
            "weight_bytes":      self.weight_bytes,
            "output_bytes":      self.output_bytes,
        }
        if self.has_side_effect:
            d["has_side_effect"] = True
        if self.inputs:
            d["inputs"] = {
                k: {"dtype": v.dtype, "shape": v.shape, "location": v.location}
                for k, v in self.inputs.items()
            }
        if self.weights:
            d["weights"] = {
                k: {"dtype": v.dtype, "shape": v.shape, "location": v.location}
                for k, v in self.weights.items()
            }
        if self.outputs:
            d["outputs"] = {
                k: {"dtype": v.dtype, "shape": v.shape, "location": v.location}
                for k, v in self.outputs.items()
            }
        return d


