"""Placement — assigns kernels to physical devices with resource allocation.

The placement pass produces a Placement object that maps each compute kernel
in the graph to a (device, stream, resource_cap) triple. Communication kernels
do not require placement — their cost is attributed to adjacent compute kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, Optional

from rooflang.language.graph import HardwareGraph

if TYPE_CHECKING:
    from rooflang.language.graph import ComputeGraph
    from rooflang.language.hardware.component import Compute
    from rooflang.language.kernels.kernel import Kernel


@dataclass
class DeviceAssignment:
    """A kernel's placement on a single physical device.

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

    Only compute kernels (where _requires_placement is True) need entries.
    Communication kernels are exempt.
    """

    def __init__(self, hardware: Optional[HardwareGraph] = None) -> None:
        self._hardware = hardware
        self._mapping: Dict[Kernel, DeviceAssignment] = {}

    def set(self, kernel: Kernel, device: Compute, stream: int = 0,
            resource_cap: float = 1.0) -> None:
        """Assign a kernel to a single device/stream.

        Automatically sets tensor.location to the device's local memory
        for any tensor where location is still None (requires hardware).
        """
        if resource_cap <= 0.0 or resource_cap > 1.0:
            raise ValueError(
                f"resource_cap must be in (0, 1], got {resource_cap}")
        self._mapping[kernel] = DeviceAssignment(device, stream, resource_cap)
        if self._hardware is None:
            return
        mem = self._hardware.find_local_memory(device)
        for t in kernel.inputs.values():
            if t.location is None:
                t.location = mem
        for t in kernel.weights.values():
            if t.location is None:
                t.location = mem
        for t in kernel.outputs.values():
            if t.location is None:
                t.location = mem

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
        """Verify placement and graph are consistent.

        Checks that every kernel requiring placement is placed, and that
        no extraneous placements exist for kernels not in the graph.
        """
        required = {k for k in graph.kernels if k._requires_placement}
        unplaced = required - self.placed_kernels
        if unplaced:
            raise ValueError(f"Unplaced kernels: {unplaced}")
        extraneous = self.placed_kernels - graph.kernels
        if extraneous:
            raise ValueError(
                f"Extraneous placements (not in graph): {extraneous}")
