"""Placement — assigns kernels to physical devices with resource allocation.

The placement pass produces a Placement object that maps each kernel in the
compute graph to a (device, stream, resource_cap) triple. The simulator
consumes this mapping to schedule kernels on devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet

if TYPE_CHECKING:
    from rooflang.language.graph import ComputeGraph
    from rooflang.language.hardware.component import Compute
    from rooflang.language.kernels.kernel import Kernel


@dataclass
class DeviceAssignment:
    """A kernel's placement on a physical device.

    Attributes:
        device: the Compute node this kernel runs on.
        stream: execution lane index (same device + same stream = serial).
        resource_cap: fraction of device resource allocated, in (0, 1].
    """
    device: Compute
    stream: int = 0
    resource_cap: float = 1.0


class Placement:
    """Placement mapping: kernel -> DeviceAssignment.

    Created by the placement pass, consumed by the simulator.
    """

    def __init__(self) -> None:
        self._mapping: Dict[Kernel, DeviceAssignment] = {}

    def set(self, kernel: Kernel, device: Compute, stream: int = 0,
            resource_cap: float = 1.0) -> None:
        """Assign a kernel to a device/stream with resource allocation."""
        if resource_cap <= 0.0 or resource_cap > 1.0:
            raise ValueError(
                f"resource_cap must be in (0, 1], got {resource_cap}")
        self._mapping[kernel] = DeviceAssignment(device, stream, resource_cap)

    def get(self, kernel: Kernel) -> DeviceAssignment:
        """Get placement for a kernel. Raises KeyError if not placed."""
        if kernel not in self._mapping:
            raise KeyError(f"Kernel not placed: {kernel}")
        return self._mapping[kernel]

    @property
    def placed_kernels(self) -> FrozenSet[Kernel]:
        """All kernels that have been assigned a placement."""
        return frozenset(self._mapping.keys())

    def validate(self, graph: ComputeGraph) -> None:
        """Verify placement and graph have exactly the same kernel set.

        Raises ValueError if any kernel is unplaced or extraneous.
        """
        unplaced = graph.kernels - self.placed_kernels
        if unplaced:
            raise ValueError(f"Unplaced kernels: {unplaced}")
        extraneous = self.placed_kernels - graph.kernels
        if extraneous:
            raise ValueError(f"Extraneous placements (not in graph): {extraneous}")
