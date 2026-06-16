"""Base Kernel class and composition primitives (FusedKernel, OverlappedKernel).

Every primitive (forward, backward, optimizer) is a Kernel subclass carrying
roofline metrics. Composition rules live here so they're defined once:

  FusedKernel(a, b):
    A's output stays in SMEM/registers → never hits HBM.
    Eliminates A.output_bytes and B.input_bytes from transferred_bytes.
    flops, weight_bytes sum normally.

  OverlappedKernel(a, b):
    Both run concurrently (e.g. compute overlapped with comm).
    All resource fields sum. Wall-clock time = max(time_A, time_B),
    handled by the scheduler layer — not part of the Kernel contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rooflang.language.parallelism import ShardingSpec


@dataclass
class Kernel:
    """Base: one primitive or composite with its roofline metrics.

    Attributes beyond roofline numbers:
      sharding:   per-primitive ShardingSpec (set by the enumerator).
      recompute:  whether this primitive is recomputed during backward
                  (adds one extra forward pass of FLOPs/bytes).
    """
    flops: float
    transferred_bytes: float
    input_bytes: float
    weight_bytes: float
    output_bytes: float
    sharding: Optional[ShardingSpec] = field(default=None, repr=False)
    recompute: Optional[bool] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {
            "flops":             self.flops,
            "transferred_bytes": self.transferred_bytes,
            "input_bytes":       self.input_bytes,
            "weight_bytes":      self.weight_bytes,
            "output_bytes":      self.output_bytes,
        }
        if self.recompute is not None:
            d["recompute"] = self.recompute
        if self.sharding is not None:
            d["sharding"] = {
                "tp": self.sharding.tp,
                "dp": self.sharding.dp,
                "cp": self.sharding.cp,
                "ep": self.sharding.ep,
                "zero": self.sharding.zero,
            }
        return d


class FusedKernel(Kernel):
    """A fused with B: A's output + B's input eliminated from HBM.

    Assumptions:
      - A's output is B's input (same intermediate tensor, held in SMEM/regs).
      - If A.output_bytes != B.input_bytes, the smaller of the two is
        eliminated (conservative: only the shared portion cancels).
    """

    def __init__(self, a: Kernel, b: Kernel):
        self.children = (a, b)
        self.flops = a.flops + b.flops
        self.input_bytes = a.input_bytes
        self.weight_bytes = a.weight_bytes + b.weight_bytes
        self.output_bytes = b.output_bytes
        self.transferred_bytes = (self.input_bytes + self.weight_bytes
                                  + self.output_bytes)


class OverlappedKernel(Kernel):
    """A overlapped with B: both consume resources simultaneously.

    All resource fields sum (total bandwidth / compute demand).
    Wall-clock time = max(time_A, time_B), handled externally.
    """

    def __init__(self, a: Kernel, b: Kernel):
        self.children = (a, b)
        self.flops = a.flops + b.flops
        self.input_bytes = a.input_bytes + b.input_bytes
        self.weight_bytes = a.weight_bytes + b.weight_bytes
        self.output_bytes = a.output_bytes + b.output_bytes
        self.transferred_bytes = a.transferred_bytes + b.transferred_bytes
