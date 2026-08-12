"""Hardware component nodes: Compute and Memory.

Nodes in the hardware graph. Compute represents processing units (GPU, CPU,
NIC, switch). Memory represents storage (HBM, DRAM, SSD).
"""

from __future__ import annotations

from typing import Dict


class HardwareComponent:
    """Base node in the hardware graph (identity semantics for hashing)."""

    def __init__(self, name: str, kind: str | None = None) -> None:
        self.name = name
        self.kind = kind


class Compute(HardwareComponent):
    """A compute node: GPU, CPU, NIC, switch, etc."""

    def __init__(self, name: str, tflops: Dict[str, float] | None = None,
                 kind: str | None = None) -> None:
        super().__init__(name, kind)
        self.tflops = tflops if tflops is not None else {}


class Memory(HardwareComponent):
    """A memory node: HBM, DRAM, SSD, etc."""

    def __init__(self, name: str, capacity_gb: float = 0.0,
                 kind: str | None = None) -> None:
        super().__init__(name, kind)
        self.capacity_gb = capacity_gb
