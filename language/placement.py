# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Placement — assigns kernels to devices and tensors to memory nodes.

The placement pass produces a Placement object that maps each compute kernel
to a (device, stream, resource_cap) triple, and each tensor to a Memory node.
Communication kernels do not require device placement. Their participant
devices are inferred from the memory placement of their input/output tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional

from rooflang.language.graph import HardwareGraph

if TYPE_CHECKING:
    from rooflang.language.graph import ComputeGraph
    from rooflang.language.hardware.component import Compute, Memory
    from rooflang.language.kernels.kernel import Kernel
    from rooflang.language.tensor import Tensor


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


@dataclass(frozen=True)
class MemoryFootprint:
    """Tagged memory usage metadata for one simulated graph wave."""

    memory: Memory
    size_bytes: float
    role: str


class Placement:
    """Placement mapping: kernel -> DeviceAssignment, tensor -> Memory.

    Only compute kernels (where _requires_placement is True) need entries.
    Communication kernels are exempt.
    """

    def __init__(self, hardware: Optional[HardwareGraph] = None,
                 graph: Optional[ComputeGraph] = None) -> None:
        self._hardware = hardware
        self._graph = graph
        self._mapping: Dict[Kernel, DeviceAssignment] = {}
        self._memory: Dict[Tensor, Memory] = {}
        self._memory_footprints: Dict[tuple[Memory, str], float] = {}

    # ── Kernel device placement ──────────────────────────────────────

    def set_kernel_device(self, kernel: Kernel, device: Compute,
                          stream: int = 0, resource_cap: float = 1.0) -> None:
        """Assign a kernel to a device/stream and auto-assign tensor memory."""
        if resource_cap <= 0.0 or resource_cap > 1.0:
            raise ValueError(
                f"resource_cap must be in (0, 1], got {resource_cap}")
        self._mapping[kernel] = DeviceAssignment(device, stream, resource_cap)
        if self._hardware is None:
            return
        self._assign_tensor_memory(kernel, device)

    def get_kernel_device(self, kernel: Kernel) -> DeviceAssignment:
        """Get placement for a kernel. Raises KeyError if not placed."""
        if kernel not in self._mapping:
            raise KeyError(f"Kernel not placed: {kernel}")
        return self._mapping[kernel]

    # ── Tensor memory placement ──────────────────────────────────────

    def set_tensor_memory(self, tensor: Tensor, memory: Memory) -> None:
        """Explicitly set the memory location for a tensor."""
        self._memory[tensor] = memory

    def get_tensor_memory(self, tensor: Tensor) -> Optional[Memory]:
        """Get the memory location for a tensor, or None if unset."""
        return self._memory.get(tensor)

    def record_memory_footprint(
        self, memory: Memory, size_bytes: float, role: str,
    ) -> None:
        """Record tagged usage without changing simulated memory accounting."""
        if size_bytes < 0:
            raise ValueError(
                f"memory footprint must be non-negative, got {size_bytes}")
        if size_bytes == 0:
            return
        key = (memory, role)
        self._memory_footprints[key] = \
            self._memory_footprints.get(key, 0.0) + size_bytes

    def infer_comm_devices(self, kernel: Kernel) -> List[Compute]:
        """Infer comm participants solely from its placed tensor ports."""
        if self._hardware is None:
            raise ValueError("Cannot infer comm devices without hardware")

        devices = []
        seen = set()
        for tensor in (*kernel.inputs.values(), *kernel.outputs.values()):
            memory = self.get_tensor_memory(tensor)
            if memory is None:
                raise ValueError(
                    f"Cannot infer device for {type(kernel).__name__}: "
                    "an input/output tensor has no memory placement")
            device = self._hardware.find_local_device(memory)
            if device not in seen:
                seen.add(device)
                devices.append(device)
        if not devices:
            raise ValueError(
                f"Cannot infer device for {type(kernel).__name__}: "
                "kernel has no input/output tensors")
        return devices

    def get_tensor_device(self, tensor: Tensor) -> Compute:
        """Return the local compute device for one placed tensor."""
        if self._hardware is None:
            raise ValueError("Cannot infer tensor device without hardware")
        memory = self.get_tensor_memory(tensor)
        if memory is None:
            raise ValueError("Tensor has no memory placement")
        return self._hardware.find_local_device(memory)

    # ── Query ────────────────────────────────────────────────────────

    @property
    def placed_kernels(self) -> FrozenSet[Kernel]:
        """All kernels that have been assigned a placement."""
        return frozenset(self._mapping.keys())

    @property
    def memory_footprints(self) -> tuple[MemoryFootprint, ...]:
        """Tagged per-memory usage metadata attached to this placement."""
        return tuple(
            MemoryFootprint(memory, size_bytes, role)
            for (memory, role), size_bytes in self._memory_footprints.items()
        )

    # ── Validation ───────────────────────────────────────────────────

    def validate(self, graph: ComputeGraph) -> None:
        """Verify placement, tensor memory, and data-edge consistency."""
        from rooflang.language.kernels.forward import Slice
        from rooflang.language.kernels.identity import Concat, Spawn

        required = {k for k in graph.kernels if k._requires_placement}
        unplaced = required - self.placed_kernels
        if unplaced:
            raise ValueError(f"Unplaced kernels: {unplaced}")
        extraneous = self.placed_kernels - graph.kernels
        if extraneous:
            raise ValueError(
                f"Extraneous placements (not in graph): {extraneous}")

        for kernel in graph.kernels:
            for name, t in kernel.inputs.items():
                if t not in self._memory:
                    raise ValueError(
                        f"Tensor '{name}' (input of {kernel}) has no memory")
            for name, t in kernel.weights.items():
                if t not in self._memory:
                    raise ValueError(
                        f"Tensor '{name}' (weight of {kernel}) has no memory")
            for name, t in kernel.outputs.items():
                if t not in self._memory:
                    raise ValueError(
                        f"Tensor '{name}' (output of {kernel}) has no memory")

        same_memory_types = (Spawn, Concat, Slice)
        for kernel in graph.kernels:
            if not isinstance(kernel, same_memory_types):
                continue
            ports = list(kernel.inputs.items()) + list(kernel.outputs.items())
            memories = []
            for name, tensor in ports:
                memory = self._memory.get(tensor)
                if memory is None:
                    raise ValueError(
                        f"Tensor '{name}' of {type(kernel).__name__} "
                        "has no memory")
                memories.append(memory)
            if memories and any(m is not memories[0] for m in memories[1:]):
                names = ", ".join(sorted({m.name for m in memories}))
                raise ValueError(
                    f"{type(kernel).__name__} input/output tensors must "
                    f"share one memory, found: {names}")

        for src in graph.kernels:
            for edge in graph._out_edges(src):
                for out_name, in_name in edge.mapping.items():
                    src_t = edge.src.outputs[out_name]
                    dst_t = edge.dst.inputs[in_name]
                    src_mem = self._memory.get(src_t)
                    dst_mem = self._memory.get(dst_t)
                    if src_mem is not None and dst_mem is not None \
                       and src_mem is not dst_mem:
                        raise ValueError(
                            f"Memory mismatch on edge {out_name}->{in_name}: "
                            f"{src_mem.name} vs {dst_mem.name}")

    # ── Internal ─────────────────────────────────────────────────────

    def _assign_tensor_memory(self, kernel: Kernel, device: Compute) -> None:
        from rooflang.language.kernels.identity import Spawn

        mem = self._hardware.find_local_memory(device)
        predecessor_memories = {}
        if self._graph is not None:
            for src, _, attr in self._graph._dag.in_edges(
                    kernel, data=True):
                for out_name, in_name in attr["mapping"].items():
                    if in_name not in predecessor_memories:
                        predecessor_memories[in_name] = self._memory.get(
                            src.outputs[out_name])

        # Spawn is a zero-cost aliasing fan-out. Its outputs remain in the
        # input allocation even when downstream compute is placed remotely.
        if isinstance(kernel, Spawn):
            source_memory = next(
                (memory for memory in predecessor_memories.values()
                 if memory is not None),
                mem,
            )
            for tensor in (*kernel.inputs.values(), *kernel.outputs.values()):
                self._memory[tensor] = source_memory
            return

        for t in kernel.weights.values():
            if t in self._memory:
                continue
            self._memory[t] = mem

        for name, t in kernel.inputs.items():
            if t in self._memory:
                continue
            pred_mem = predecessor_memories.get(name)
            self._memory[t] = pred_mem if pred_mem else mem

        for t in kernel.outputs.values():
            if t in self._memory:
                continue
            self._memory[t] = mem
