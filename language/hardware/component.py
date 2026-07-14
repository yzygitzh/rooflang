"""Hardware component nodes: Compute and Memory.

Nodes in the hardware graph. Compute represents processing units (GPU, CPU,
NIC, switch). Memory represents storage (HBM, DRAM, SSD).
"""

from __future__ import annotations

from typing import Dict


class HardwareComponent:
    """Base node in the hardware graph (identity semantics for hashing)."""

    def __init__(self, name: str) -> None:
        self.name = name


class Compute(HardwareComponent):
    """A compute node: GPU, CPU, NIC, switch, etc."""

    def __init__(self, name: str, tflops: Dict[str, float] | None = None) -> None:
        super().__init__(name)
        self.tflops = tflops if tflops is not None else {}


class Memory(HardwareComponent):
    """A memory node: HBM, DRAM, SSD, etc."""

    def __init__(self, name: str, capacity_gb: float = 0.0) -> None:
        super().__init__(name)
        self.capacity_gb = capacity_gb
