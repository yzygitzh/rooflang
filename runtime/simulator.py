"""Roofline simulator — DES with resource contention.

Overlapping kernels on the same device share compute/memory by weighted
cap; overlapping comms on the same fabric share link BW equally (bottleneck).
Stream semantics: kernels on the same (device, stream) execute serially.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import AllToAll, CommKernel, Recv, Send
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor


class Bound(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"


@dataclass
class TensorInfo:
    """A tensor alive in memory at the time of OOM."""
    tensor: Tensor
    size_bytes: float
    role: str  # "weight", "output", "root_input"


class OOMError(Exception):
    """Raised when a Memory node exceeds capacity during simulation."""

    def __init__(self, memory: Memory, used_bytes: float,
                 capacity_bytes: float, alive_tensors: List[TensorInfo],
                 trigger_kernel: Kernel) -> None:
        self.memory = memory
        self.used_bytes = used_bytes
        self.capacity_bytes = capacity_bytes
        self.alive_tensors = alive_tensors
        self.trigger_kernel = trigger_kernel
        super().__init__(
            f"OOM on '{memory.name}': {used_bytes / 1e9:.2f} GB used, "
            f"{capacity_bytes / 1e9:.2f} GB capacity")


class TraceEntry:
    __slots__ = ("kernel", "device", "stream", "start_us", "end_us", "bound")

    def __init__(self, kernel, device, stream, start_us, end_us, bound):
        self.kernel = kernel
        self.device = device
        self.stream = stream
        self.start_us = start_us
        self.end_us = end_us
        self.bound = bound


class SimulationResult:
    def __init__(self, trace: List[TraceEntry], total_time_us: float,
                 peak_memory: Dict[Memory, float]):
        self.trace = trace
        self.total_time_us = total_time_us
        self.peak_memory = peak_memory


class RunningKernel:
    """Running kernel state in the DES."""

    __slots__ = ("kernel", "device", "stream", "resource_cap",
                 "compute_time", "memory_time",
                 "network_alpha", "network_transfer_time",
                 "fabric_edges", "start_us",
                 "cp", "mp", "tp", "alpha_remaining",
                 "seg_start", "dev_share", "net_share")

    def __init__(self, kernel, device, stream, cap, ct, mt,
                 alpha, xfer, fabs, t0):
        self.kernel = kernel
        self.device = device
        self.stream = stream
        self.resource_cap = cap
        self.compute_time = ct
        self.memory_time = mt
        self.network_alpha = alpha
        self.network_transfer_time = xfer
        self.fabric_edges = fabs
        self.start_us = t0
        self.cp = self.mp = self.tp = 0.0
        self.alpha_remaining = alpha
        self.seg_start = t0
        self.dev_share = self.net_share = 1.0

    def advance_to(self, now):
        dt = now - self.seg_start
        if dt <= 0:
            return
        if self.compute_time > 0:
            self.cp += dt * self.dev_share / self.compute_time
        if self.memory_time > 0:
            self.mp += dt * self.dev_share / self.memory_time
        if self.alpha_remaining > 0:
            consumed = min(dt, self.alpha_remaining)
            self.alpha_remaining -= consumed
            net_dt = dt - consumed
        else:
            net_dt = dt
        if self.network_transfer_time > 0 and net_dt > 0:
            self.tp += net_dt * self.net_share / self.network_transfer_time
        self.seg_start = now

    def eta(self) -> float:
        worst = 0.0
        if self.compute_time > 0 and self.cp < 1.0:
            worst = max(worst, (1.0 - self.cp) * self.compute_time / self.dev_share)
        if self.memory_time > 0 and self.mp < 1.0:
            worst = max(worst, (1.0 - self.mp) * self.memory_time / self.dev_share)
        net_remaining = self.alpha_remaining
        if self.network_transfer_time > 0 and self.tp < 1.0:
            net_remaining += (1.0 - self.tp) * self.network_transfer_time / self.net_share
        worst = max(worst, net_remaining)
        return self.seg_start + worst

    def bound(self) -> Bound:
        nt = self.network_alpha + self.network_transfer_time
        if nt >= self.compute_time and nt >= self.memory_time and nt > 0:
            return Bound.NETWORK
        if self.compute_time >= self.memory_time and self.compute_time > 0:
            return Bound.COMPUTE
        return Bound.MEMORY


class Simulator:
    """Discrete-event roofline simulator with resource contention."""

    def __init__(self, graph: ComputeGraph, placement: Placement,
                 hardware: HardwareGraph) -> None:
        self._graph = graph
        self._placement = placement
        self._hardware = hardware

    def run(self) -> SimulationResult:
        """Execute DES and return trace + total time."""
        self._kernel_end: Dict[Kernel, float] = {}
        self._stream_end: Dict[Tuple[Compute, int], float] = {}
        self._on_dev: Dict[Compute, List[RunningKernel]] = {}
        self._on_fab: Dict[FabricEdge, List[RunningKernel]] = {}
        self._trace: List[TraceEntry] = []
        self._completed: Set[Kernel] = set()
        self._eid = 0
        self._eq: list = []
        self._stream_active: Dict[Tuple[Compute, int], RunningKernel] = {}
        self._stream_pending: Dict[Tuple[Compute, int], List[Kernel]] = {}

        # ── Memory tracking init ────────────────────────────────────
        self._mem_usage: Dict[Memory, float] = defaultdict(float)
        self._mem_peak: Dict[Memory, float] = defaultdict(float)
        self._out_refcount: Dict[Tensor, int] = defaultdict(int)
        self._root_inputs: Dict[Kernel, List[Tensor]] = {}
        self._alive: Dict[Memory, Set[Tuple[Tensor, str]]] = defaultdict(set)

        for kernel in self._graph.kernels:
            for edge in self._graph._out_edges(kernel):
                for out_name in edge.mapping:
                    t = kernel.outputs[out_name]
                    self._out_refcount[t] += 1

        for kernel in self._graph.kernels:
            for t in kernel.weights.values():
                mem = self._placement.get_tensor_memory(t)
                if mem is not None:
                    self._mem_usage[mem] += t.size_bytes
                    self._alive[mem].add((t, "weight"))
            has_data_preds = bool(self._graph._in_edges(kernel))
            if not has_data_preds:
                inputs_for_kernel = []
                for t in kernel.inputs.values():
                    mem = self._placement.get_tensor_memory(t)
                    if mem is not None:
                        self._mem_usage[mem] += t.size_bytes
                        self._alive[mem].add((t, "root_input"))
                        inputs_for_kernel.append(t)
                if inputs_for_kernel:
                    self._root_inputs[kernel] = inputs_for_kernel

        for mem, usage in self._mem_usage.items():
            self._mem_peak[mem] = usage
            self._check_oom(mem, None)

        # ── Schedule root kernels ───────────────────────────────────
        for kernel in self._graph.topological_sort():
            if not list(self._graph._dag.predecessors(kernel)):
                dev, stream, _, _ = self._resolve(kernel)
                self._push(0.0, "start", kernel, dev, stream)

        while self._eq:
            now, _, typ, kernel, dev, stream = heapq.heappop(self._eq)
            if kernel in self._completed:
                continue
            if typ == "start":
                key = (dev, stream)
                if key in self._stream_active:
                    self._stream_pending.setdefault(key, []).append(kernel)
                else:
                    self._start_kernel(kernel, now)
            else:
                rk = self._stream_active.get((dev, stream))
                if rk is None or rk.kernel is not kernel:
                    continue
                rk.advance_to(now)
                if rk.eta() > now + 1e-9:
                    self._push(rk.eta(), "end", kernel, dev, stream)
                    continue
                self._completed.add(kernel)
                self._kernel_end[kernel] = now
                self._stream_end[(dev, stream)] = now
                self._trace.append(TraceEntry(
                    kernel, dev, stream, rk.start_us, now, rk.bound()))
                self._complete_kernel_memory(kernel)
                self._finish_kernel(rk, now)
                for succ in self._graph._dag.successors(kernel):
                    if succ in self._completed:
                        continue
                    preds = list(self._graph._dag.predecessors(succ))
                    if all(p in self._completed for p in preds):
                        t = max(self._kernel_end[p] for p in preds)
                        s_dev, s_strm, _, _ = self._resolve(succ)
                        sr = self._stream_end.get((s_dev, s_strm), 0.0)
                        self._push(max(t, sr), "start", succ, s_dev, s_strm)

        total = max(self._kernel_end.values()) if self._kernel_end else 0.0
        return SimulationResult(self._trace, total, dict(self._mem_peak))

    # ── Event queue ─────────────────────────────────────────────────

    def _push(self, t, typ, kernel, dev, stream):
        heapq.heappush(self._eq, (t, self._eid, typ, kernel, dev, stream))
        self._eid += 1

    # ── Kernel lifecycle ────────────────────────────────────────────

    def _start_kernel(self, kernel: Kernel, now: float):
        dev, stream, cap, parts = self._resolve(kernel)
        ct, mt, alpha, xfer, fabs = self._base_times(kernel, dev, parts)
        rk = RunningKernel(kernel, dev, stream, cap, ct, mt,
                           alpha, xfer, fabs, now)
        key = (dev, stream)
        self._stream_active[key] = rk
        self._on_dev.setdefault(dev, []).append(rk)
        for fab in fabs:
            self._on_fab.setdefault(fab, []).append(rk)
        self._allocate_outputs(kernel)
        self._advance_peers(rk, now)
        self._recompute_shares(rk)
        self._push(rk.eta(), "end", kernel, dev, stream)
        self._resched_peers(rk)

    def _finish_kernel(self, rk: RunningKernel, now: float):
        key = (rk.device, rk.stream)
        del self._stream_active[key]
        self._on_dev[rk.device].remove(rk)
        for fab in rk.fabric_edges:
            self._on_fab[fab].remove(rk)
        self._advance_peers(rk, now)
        if self._on_dev.get(rk.device):
            self._recompute_shares(self._on_dev[rk.device][0])
        for fab in rk.fabric_edges:
            if self._on_fab.get(fab):
                self._recompute_shares(self._on_fab[fab][0])
        self._resched_peers(rk)
        # Schedule next pending kernel on this stream
        pending = self._stream_pending.get(key, [])
        while pending:
            next_k = pending.pop(0)
            if next_k not in self._completed:
                self._push(now, "start", next_k, rk.device, rk.stream)
                break

    # ── Memory tracking ────────────────────────────────────────────

    def _allocate_outputs(self, kernel: Kernel) -> None:
        for t in kernel.outputs.values():
            mem = self._placement.get_tensor_memory(t)
            if mem is None:
                continue
            self._mem_usage[mem] += t.size_bytes
            self._alive[mem].add((t, "output"))
            if self._mem_usage[mem] > self._mem_peak[mem]:
                self._mem_peak[mem] = self._mem_usage[mem]
            self._check_oom(mem, kernel)

    def _complete_kernel_memory(self, kernel: Kernel) -> None:
        for edge in self._graph._in_edges(kernel):
            for out_name in edge.mapping:
                src_t = edge.src.outputs[out_name]
                self._out_refcount[src_t] -= 1
                if self._out_refcount[src_t] == 0:
                    mem = self._placement.get_tensor_memory(src_t)
                    if mem is not None:
                        self._mem_usage[mem] -= src_t.size_bytes
                        self._alive[mem].discard((src_t, "output"))
        if kernel in self._root_inputs:
            for t in self._root_inputs[kernel]:
                mem = self._placement.get_tensor_memory(t)
                if mem is not None:
                    self._mem_usage[mem] -= t.size_bytes
                    self._alive[mem].discard((t, "root_input"))

    def _check_oom(self, mem: Memory, trigger_kernel: Kernel | None) -> None:
        capacity_bytes = mem.capacity_gb * 1e9
        if self._mem_usage[mem] > capacity_bytes:
            alive = [TensorInfo(t, t.size_bytes, role)
                     for t, role in self._alive[mem]]
            raise OOMError(mem, self._mem_usage[mem], capacity_bytes,
                           alive, trigger_kernel)

    def _recompute_shares(self, rk: RunningKernel):
        peers = self._on_dev.get(rk.device, [])
        tc = sum(p.resource_cap for p in peers)
        s = 1.0 / max(1.0, tc)
        for p in peers:
            p.dev_share = p.resource_cap * s
        # Collect all kernels affected by fabric changes
        affected: Set[RunningKernel] = set()
        for fab in rk.fabric_edges:
            for p in self._on_fab.get(fab, []):
                affected.add(p)
        for p in affected:
            min_share = 1.0
            for fab in p.fabric_edges:
                fp = self._on_fab.get(fab, [])
                fab_share = 1.0 / max(1, len(fp))
                min_share = min(min_share, fab_share)
            p.net_share = min_share

    def _advance_peers(self, rk: RunningKernel, now: float):
        for p in self._on_dev.get(rk.device, []):
            p.advance_to(now)
        for fab in rk.fabric_edges:
            for p in self._on_fab.get(fab, []):
                p.advance_to(now)

    def _resched_peers(self, rk: RunningKernel):
        seen: Set[RunningKernel] = set()
        for p in self._on_dev.get(rk.device, []):
            if p is not rk:
                seen.add(p)
        for fab in rk.fabric_edges:
            for p in self._on_fab.get(fab, []):
                if p is not rk:
                    seen.add(p)
        for p in seen:
            self._push(p.eta(), "end", p.kernel, p.device, p.stream)

    # ── Resolution helpers ──────────────────────────────────────────

    def _resolve(self, kernel: Kernel):
        if kernel._requires_placement:
            a = self._placement.get_kernel_device(kernel)
            return a.device, a.stream, a.resource_cap, []
        preds = list(self._graph._dag.predecessors(kernel))
        succs = list(self._graph._dag.successors(kernel))
        pd = [self._placement.get_kernel_device(p).device
              for p in preds if p._requires_placement]
        sd = [self._placement.get_kernel_device(s).device
              for s in succs if s._requires_placement]
        seen: Set[Compute] = set()
        devs: List[Compute] = []
        for d in pd + sd:
            if d not in seen:
                seen.add(d)
                devs.append(d)
        primary = pd[0] if pd else sd[0] if sd else None
        if primary is None:
            raise ValueError(
                f"Cannot resolve device for kernel: no placed neighbors")
        stream = 0
        for p in preds:
            if p._requires_placement:
                stream = self._placement.get_kernel_device(p).stream
                break
        return primary, stream, 1.0, devs

    def _base_times(self, kernel: Kernel, device: Compute,
                    participants: List[Compute]):
        dtype = self._infer_dtype(kernel)
        peak = device.tflops.get(dtype, 0.0) * 1e6
        if kernel.flops > 0 and peak == 0:
            raise ValueError(
                f"Device '{device.name}' has no compute for dtype '{dtype}' "
                f"but kernel has {kernel.flops} flops")
        ct = kernel.flops / peak if peak > 0 else 0.0

        mt = 0.0
        if kernel._requires_placement:
            for t in list(kernel.inputs.values()) + list(kernel.weights.values()):
                mem = self._placement.get_tensor_memory(t) \
                    if self._placement.get_tensor_memory(t) \
                    else self._hardware.find_local_memory(device)
                fab = self._hardware.find_fabric(device, mem)
                bw = fab.dst_to_src_bandwidth_gbs * 1e3
                if t.size_bytes > 0 and bw <= 0:
                    raise ValueError(
                        f"Zero read bandwidth on fabric '{fab.name}' "
                        f"for tensor with {t.size_bytes} bytes")
                if bw > 0:
                    mt += t.size_bytes / bw
            for t in kernel.outputs.values():
                mem = self._placement.get_tensor_memory(t) \
                    if self._placement.get_tensor_memory(t) \
                    else self._hardware.find_local_memory(device)
                fab = self._hardware.find_fabric(device, mem)
                bw = fab.src_to_dst_bandwidth_gbs * 1e3
                if t.size_bytes > 0 and bw <= 0:
                    raise ValueError(
                        f"Zero write bandwidth on fabric '{fab.name}' "
                        f"for tensor with {t.size_bytes} bytes")
                if bw > 0:
                    mt += t.size_bytes / bw

        alpha = 0.0
        xfer = 0.0
        fab_edges: List[FabricEdge] = []
        if isinstance(kernel, CommKernel) and len(participants) >= 2 \
           and kernel.transferred_bytes > 0:
            seen_fab: Set[FabricEdge] = set()
            for i, d1 in enumerate(participants):
                for d2 in participants[i + 1:]:
                    for fab in self._hardware.find_fabric_path(d1, d2):
                        if fab not in seen_fab:
                            seen_fab.add(fab)
                            fab_edges.append(fab)
            if isinstance(kernel, (AllToAll, Send, Recv)):
                eff_bw = min(
                    min(self._hardware.find_fabric(d1, d2).src_to_dst_bandwidth_gbs,
                        self._hardware.find_fabric(d1, d2).dst_to_src_bandwidth_gbs)
                    for i, d1 in enumerate(participants)
                    for d2 in participants[i + 1:])
            else:
                eff_bw = self._hardware.find_aggregate_bandwidth(participants)
            alpha = max(
                self._hardware.find_fabric(d1, d2).alpha_us
                for i, d1 in enumerate(participants)
                for d2 in participants[i + 1:])
            xfer = kernel.transferred_bytes / (eff_bw * 1e3)

        return ct, mt, alpha, xfer, fab_edges

    @staticmethod
    def _infer_dtype(kernel: Kernel) -> str:
        if hasattr(kernel, "w_dtype"):
            return kernel.w_dtype
        if hasattr(kernel, "dtype_"):
            return kernel.dtype_
        if kernel.inputs:
            return next(iter(kernel.inputs.values())).dtype
        return "bf16"
