"""Base Kernel class for the model-roofline DSL.

Every primitive (forward, backward, optimizer, comm, identity) is a Kernel
subclass carrying roofline metrics as @property methods.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Dict

from rooflang.language.utils import dtype_bytes

if TYPE_CHECKING:
    from rooflang.language.tensor import Tensor


class Kernel:
    """Base: one primitive or composite with its roofline metrics.

    Byte properties (input_bytes, weight_bytes, output_bytes, transferred_bytes)
    are computed from the inputs/weights/outputs dicts by default. Subclasses
    can override these properties for custom accounting.  input_tensor_bytes
    describes the full logical input tensors, while input_bytes reflects the
    bytes actually read after per-port reuse or sparse access.
    """

    _requires_placement = True

    def __init__(
        self,
        inputs: Dict[str, Tensor] | None = None,
        weights: Dict[str, Tensor] | None = None,
        outputs: Dict[str, Tensor] | None = None,
        has_side_effect: bool = False,
    ) -> None:
        self.inputs: Dict[str, Tensor] = inputs or {}
        self.weights: Dict[str, Tensor] = weights or {}
        self.outputs: Dict[str, Tensor] = outputs or {}
        self.has_side_effect = has_side_effect
        # Physical weights remain resident in full. This factor controls only
        # the expected fraction read by one logical kernel invocation.
        self.weight_read_fraction = Fraction(1, 1)

    @property
    def flops(self) -> float:
        return 0.0

    def input_read_fraction(self, port: str) -> float:
        """Return the fraction of one input tensor read by this invocation."""
        return 1.0

    @property
    def input_tensor_bytes(self) -> float:
        return sum(t.size_bytes for t in self.inputs.values())

    @property
    def input_bytes(self) -> float:
        return sum(
            t.size_bytes * self.input_read_fraction(port)
            for port, t in self.inputs.items()
        )

    @property
    def weight_bytes(self) -> float:
        return sum(t.size_bytes for t in self.weights.values())

    @property
    def loaded_weight_bytes(self) -> float:
        return float(self.weight_bytes * self.weight_read_fraction)

    @property
    def output_bytes(self) -> float:
        return sum(t.size_bytes for t in self.outputs.values())

    @property
    def transferred_bytes(self) -> float:
        return self.input_bytes + self.loaded_weight_bytes + self.output_bytes

    def to_dict(self) -> dict:
        d = {
            "flops":             self.flops,
            "transferred_bytes": self.transferred_bytes,
            "input_bytes":       self.input_bytes,
            "weight_bytes":      self.weight_bytes,
            "output_bytes":      self.output_bytes,
        }
        if self.input_tensor_bytes != self.input_bytes:
            d["input_tensor_bytes"] = self.input_tensor_bytes
        if self.weight_read_fraction != 1:
            d["loaded_weight_bytes"] = self.loaded_weight_bytes
            d["weight_read_fraction"] = float(self.weight_read_fraction)
        if self.has_side_effect:
            d["has_side_effect"] = True
        if self.inputs:
            d["inputs"] = {
                k: {"dtype": v.dtype, "shape": v.shape}
                for k, v in self.inputs.items()
            }
        if self.weights:
            d["weights"] = {
                k: {"dtype": v.dtype, "shape": v.shape}
                for k, v in self.weights.items()
            }
        if self.outputs:
            d["outputs"] = {
                k: {"dtype": v.dtype, "shape": v.shape}
                for k, v in self.outputs.items()
            }
        return d
