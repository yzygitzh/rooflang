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
from typing import Dict, List, Optional, Set, Tuple

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.comm import AllToAll, CommKernel, Recv, Send
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor

FabricKey = Tuple[FabricEdge, Optional[str]]


def _fabric_key(edge: FabricEdge, direction: str) -> FabricKey:
    return (edge, direction) if edge.is_full_duplex else (edge, None)


def _direction_bw(edge: FabricEdge, direction: str) -> float:
    return edge.src_to_dst_bandwidth_gbs if direction == 'fwd' \
        else edge.dst_to_src_bandwidth_gbs


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
                 peak_memory: Dict[Memory, float],
                 measurement_start_us: float = 0.0):
        self.trace = trace
        self.total_time_us = total_time_us
        self.peak_memory = peak_memory
        self.measurement_start_us = measurement_start_us
        self.measured_time_us = total_time_us - measurement_start_us


class RunningKernel:
    """Running kernel state in the DES."""

    __slots__ = ("kernel", "device", "stream", "resource_cap",
                 "compute_time", "memory_time",
                 "network_alpha", "network_transfer_time",
                 "link_data", "fabric_keys", "participants", "start_us",
                 "cp", "mp", "tp", "alpha_remaining",
                 "seg_start", "dev_share", "net_share")

    def __init__(self, kernel, device, stream, cap, ct, mt,
                 alpha, xfer, link_data, parts, t0):
        self.kernel = kernel
        self.device = device
        self.stream = stream
        self.resource_cap = cap
        self.compute_time = ct
        self.memory_time = mt
        self.network_alpha = alpha
        self.network_transfer_time = xfer
        self.link_data = link_data
        self.fabric_keys = list(link_data.keys()) if link_data else []
        self.participants = parts
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
                 hardware: HardwareGraph,
                 measurement_start: Kernel | None = None) -> None:
        self._graph = graph
        self._placement = placement
        self._hardware = hardware
        self._measurement_start = measurement_start

    def run(self) -> SimulationResult:
        """Execute DES and return trace + total time."""
        self._kernel_end: Dict[Kernel, float] = {}
        self._stream_end: Dict[Tuple[Compute, int], float] = {}
        self._on_dev: Dict[Compute, List[RunningKernel]] = {}
        self._on_fab: Dict[FabricKey, List[RunningKernel]] = {}
        self._trace: List[TraceEntry] = []
        self._completed: Set[Kernel] = set()
        self._eid = 0
        self._eq: list = []
        self._stream_active: Dict[Tuple[Compute, int], RunningKernel] = {}
        self._stream_pending: Dict[Tuple[Compute, int], List[Kernel]] = {}
        self._multi_stream_waiting: List[Tuple[float, Kernel]] = []

        # ── Memory tracking init ────────────────────────────────────
        self._mem_usage: Dict[Memory, float] = defaultdict(float)
        self._mem_peak: Dict[Memory, float] = defaultdict(float)
        self._out_refcount: Dict[Tensor, int] = defaultdict(int)
        self._root_inputs: Dict[Kernel, List[Tensor]] = {}
        self._alive: Dict[Memory, Set[Tuple[Tensor, str]]] = defaultdict(set)
        self._passthrough = self._build_passthrough()

        for kernel in self._graph.kernels:
            for edge in self._graph._out_edges(kernel):
                for out_name in edge.mapping:
                    t = kernel.outputs[out_name]
                    for real_t in self._passthrough.get(t, [t]):
                        self._out_refcount[real_t] += 1

        _seen_wid = set()
        for kernel in self._graph.kernels:
            for t in kernel.weights.values():
                mem = self._placement.get_tensor_memory(t)
                if mem is None:
                    continue
                if t.weight_id is not None:
                    key = (id(mem), t.weight_id)
                    if key in _seen_wid:
                        continue
                    _seen_wid.add(key)
                self._mem_usage[mem] += t.size_bytes
                self._alive[mem].add((t, "weight"))
            has_data_predecessor = bool(self._graph._in_edges(kernel))
            if not has_data_predecessor:
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
                    pending = self._stream_pending.setdefault(key, [])
                    if kernel not in pending:
                        pending.append(kernel)
                else:
                    if not self._start_kernel(kernel, now):
                        self._schedule_next_pending(key, now)
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
                bound = rk.bound()
                if isinstance(kernel, CommKernel) and len(rk.participants) > 1:
                    for part_dev in rk.participants:
                        self._trace.append(TraceEntry(
                            kernel, part_dev, stream, rk.start_us, now, bound))
                else:
                    self._trace.append(TraceEntry(
                        kernel, dev, stream, rk.start_us, now, bound))
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
        measurement_start_us = 0.0
        if self._measurement_start is not None:
            starts = [entry.start_us for entry in self._trace
                      if entry.kernel is self._measurement_start]
            if not starts:
                raise ValueError("Measurement-start kernel was not executed")
            measurement_start_us = min(starts)
        return SimulationResult(
            self._trace, total, dict(self._mem_peak), measurement_start_us)

    # ── Event queue ─────────────────────────────────────────────────

    def _push(self, t, typ, kernel, dev, stream):
        heapq.heappush(self._eq, (t, self._eid, typ, kernel, dev, stream))
        self._eid += 1

    # ── Kernel lifecycle ────────────────────────────────────────────

    def _is_multi_stream_comm(self, kernel: Kernel, parts) -> bool:
        """True when a collective comm blocks ALL participant streams."""
        return isinstance(kernel, CommKernel) and len(parts) > 1

    def _start_kernel(self, kernel: Kernel, now: float) -> bool:
        dev, stream, cap, parts = self._resolve(kernel)

        # Collective comm blocks ALL participant streams
        if self._is_multi_stream_comm(kernel, parts):
            for p_dev in parts:
                if (p_dev, stream) in self._stream_active:
                    if all(waiting is not kernel
                           for _, waiting in self._multi_stream_waiting):
                        self._multi_stream_waiting.append((now, kernel))
                    return False

        ct, mt, alpha, xfer, link_data = self._base_times(kernel, dev, parts)
        rk = RunningKernel(kernel, dev, stream, cap, ct, mt,
                           alpha, xfer, link_data, parts, now)
        key = (dev, stream)
        self._stream_active[key] = rk
        if self._is_multi_stream_comm(kernel, parts):
            for p_dev in parts:
                p_key = (p_dev, stream)
                if p_key != key:
                    self._stream_active[p_key] = rk
        self._on_dev.setdefault(dev, []).append(rk)
        for fk in rk.fabric_keys:
            self._on_fab.setdefault(fk, []).append(rk)
        self._allocate_outputs(kernel)
        self._advance_peers(rk, now)
        self._recompute_shares(rk)
        self._push(rk.eta(), "end", kernel, dev, stream)
        self._resched_peers(rk)
        return True

    def _schedule_next_pending(
        self, key: Tuple[Compute, int], now: float,
    ) -> None:
        """Retry queued work without letting a blocked collective stall it."""
        pending = self._stream_pending.get(key, [])
        while pending:
            next_k = pending.pop(0)
            if next_k not in self._completed:
                self._push(now, "start", next_k, key[0], key[1])
                break

    def _finish_kernel(self, rk: RunningKernel, now: float):
        key = (rk.device, rk.stream)
        del self._stream_active[key]

        # Clear all participant streams for multi-sender comm
        extra_keys = []
        if self._is_multi_stream_comm(rk.kernel, rk.participants):
            for p_dev in rk.participants:
                p_key = (p_dev, rk.stream)
                if p_key != key:
                    if self._stream_active.get(p_key) is rk:
                        del self._stream_active[p_key]
                    extra_keys.append(p_key)
                self._stream_end[p_key] = now

        self._on_dev[rk.device].remove(rk)
        for fk in rk.fabric_keys:
            self._on_fab[fk].remove(rk)
        self._advance_peers(rk, now)
        if self._on_dev.get(rk.device):
            self._recompute_shares(self._on_dev[rk.device][0])
        for fk in rk.fabric_keys:
            if self._on_fab.get(fk):
                self._recompute_shares(self._on_fab[fk][0])
        self._resched_peers(rk)
        # Schedule next pending kernel on primary stream
        self._schedule_next_pending(key, now)
        # Schedule next pending on extra participant streams
        for p_key in extra_keys:
            self._schedule_next_pending(p_key, now)
        # Retry multi-stream-waiting comms
        still_waiting = []
        for (earliest, k) in self._multi_stream_waiting:
            if k in self._completed:
                continue
            _, s, _, parts = self._resolve(k)
            if all((p, s) not in self._stream_active for p in parts):
                d, s2, _, _ = self._resolve(k)
                self._push(max(earliest, now), "start", k, d, s2)
            else:
                still_waiting.append((earliest, k))
        self._multi_stream_waiting = still_waiting

    # ── Memory tracking ────────────────────────────────────────────

    def _build_passthrough(self) -> Dict[Tensor, list]:
        """Map comm/structural identity outputs to upstream storage tensors."""
        pt: Dict[Tensor, list] = {}
        for kernel in self._graph.topological_sort():
            if kernel._requires_placement:
                continue
            all_sources = []
            for edge in self._graph._in_edges(kernel):
                for out_name, _ in edge.mapping.items():
                    upstream_t = edge.src.outputs[out_name]
                    all_sources.extend(pt.get(upstream_t, [upstream_t]))
            if all_sources:
                for t in kernel.outputs.values():
                    pt[t] = all_sources
        return pt

    def _allocate_outputs(self, kernel: Kernel) -> None:
        for t in kernel.outputs.values():
            if t in self._passthrough:
                continue
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
                for real_t in self._passthrough.get(src_t, [src_t]):
                    self._out_refcount[real_t] -= 1
                    if self._out_refcount[real_t] == 0:
                        mem = self._placement.get_tensor_memory(real_t)
                        if mem is not None:
                            self._mem_usage[mem] -= real_t.size_bytes
                            self._alive[mem].discard((real_t, "output"))
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
        affected: Set[RunningKernel] = set()
        for fk in rk.fabric_keys:
            for p in self._on_fab.get(fk, []):
                affected.add(p)
        for p in affected:
            if not p.link_data or p.network_transfer_time <= 0:
                p.net_share = 1.0
                continue
            worst_time = 0.0
            for key, (bytes_val, bw) in p.link_data.items():
                if bytes_val <= 0:
                    continue
                n_users = max(1, len(self._on_fab.get(key, [])))
                link_time = bytes_val / (bw * 1e3 / n_users)
                worst_time = max(worst_time, link_time)
            if worst_time > 0:
                p.net_share = p.network_transfer_time / worst_time
            elif isinstance(p.kernel, CommKernel):
                # Collectives model transfer time using an aggregate bandwidth,
                # so their link_data entries identify occupied fabric links but
                # intentionally carry no per-link byte counts.  Treat the whole
                # payload phase as occupying every registered link; otherwise
                # concurrent collectives (or remote reads/writes) see the links
                # in _on_fab but never contend with them.
                max_users = max(
                    len(self._on_fab.get(key, [])) for key in p.fabric_keys
                )
                p.net_share = 1.0 / max(1, max_users)
            else:
                p.net_share = 1.0

    def _advance_peers(self, rk: RunningKernel, now: float):
        for p in self._on_dev.get(rk.device, []):
            p.advance_to(now)
        for fk in rk.fabric_keys:
            for p in self._on_fab.get(fk, []):
                p.advance_to(now)

    def _resched_peers(self, rk: RunningKernel):
        seen: Set[RunningKernel] = set()
        for p in self._on_dev.get(rk.device, []):
            if p is not rk:
                seen.add(p)
        for fk in rk.fabric_keys:
            for p in self._on_fab.get(fk, []):
                if p is not rk:
                    seen.add(p)
        for p in seen:
            self._push(p.eta(), "end", p.kernel, p.device, p.stream)

    # ── Resolution helpers ──────────────────────────────────────────

    def _resolve(self, kernel: Kernel):
        if isinstance(kernel, CommKernel):
            devices = self._infer_comm_devices(kernel)
            stream = 0
            for predecessor in self._graph._dag.predecessors(kernel):
                if predecessor._requires_placement \
                        or predecessor in self._placement._mapping:
                    stream = self._placement.get_kernel_device(
                        predecessor).stream
                    break
            return devices[0], stream, 1.0, devices

        if kernel._requires_placement or kernel in self._placement._mapping:
            a = self._placement.get_kernel_device(kernel)
            return a.device, a.stream, a.resource_cap, []
        preds = list(self._graph._dag.predecessors(kernel))
        succs = list(self._graph._dag.successors(kernel))

        def is_placed(k):
            return k._requires_placement or k in self._placement._mapping

        pd = [self._placement.get_kernel_device(p).device
              for p in preds if is_placed(p)]
        sd = [self._placement.get_kernel_device(s).device
              for s in succs if is_placed(s)]
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
            if is_placed(p):
                stream = self._placement.get_kernel_device(p).stream
                break

        return primary, stream, 1.0, devs

    def _infer_comm_devices(self, kernel: CommKernel) -> List[Compute]:
        """Infer communication devices from the kernel's placed buffers."""
        return self._placement.infer_comm_devices(kernel)

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
        xfer = 0.0
        link_data: Dict[FabricKey, List] = {}

        if kernel._requires_placement:
            local_mem = self._hardware.find_local_memory(device)
            for t in list(kernel.inputs.values()) + list(kernel.weights.values()):
                mem = self._placement.get_tensor_memory(t) \
                    if self._placement.get_tensor_memory(t) \
                    else local_mem
                if mem is local_mem:
                    fab = self._hardware.find_fabric(device, mem)
                    bw = fab.dst_to_src_bandwidth_gbs * 1e3
                    if t.size_bytes > 0 and bw <= 0:
                        raise ValueError(
                            f"Zero read bandwidth on fabric '{fab.name}' "
                            f"for tensor with {t.size_bytes} bytes")
                    if bw > 0:
                        mt += t.size_bytes / bw
                else:
                    path = self._hardware.find_fabric_path_directed(device, mem)
                    for edge, direction in path:
                        rev_dir = 'rev' if direction == 'fwd' else 'fwd'
                        key = _fabric_key(edge, rev_dir)
                        bw_gbs = _direction_bw(edge, rev_dir)
                        if key in link_data:
                            link_data[key][0] += t.size_bytes
                        else:
                            link_data[key] = [t.size_bytes, bw_gbs]
            for t in kernel.outputs.values():
                mem = self._placement.get_tensor_memory(t) \
                    if self._placement.get_tensor_memory(t) \
                    else local_mem
                if mem is local_mem:
                    fab = self._hardware.find_fabric(device, mem)
                    bw = fab.src_to_dst_bandwidth_gbs * 1e3
                    if t.size_bytes > 0 and bw <= 0:
                        raise ValueError(
                            f"Zero write bandwidth on fabric '{fab.name}' "
                            f"for tensor with {t.size_bytes} bytes")
                    if bw > 0:
                        mt += t.size_bytes / bw
                else:
                    path = self._hardware.find_fabric_path_directed(device, mem)
                    for edge, direction in path:
                        key = _fabric_key(edge, direction)
                        bw_gbs = _direction_bw(edge, direction)
                        if key in link_data:
                            link_data[key][0] += t.size_bytes
                        else:
                            link_data[key] = [t.size_bytes, bw_gbs]

        alpha = 0.0
        if isinstance(kernel, CommKernel) and len(participants) >= 2 \
           and kernel.transferred_bytes > 0:
            for i, d1 in enumerate(participants):
                for d2 in participants[i + 1:]:
                    for edge, direction in \
                            self._hardware.find_fabric_path_directed(d1, d2):
                        if edge.is_full_duplex:
                            for d in ('fwd', 'rev'):
                                key = _fabric_key(edge, d)
                                if key not in link_data:
                                    link_data[key] = [0.0, _direction_bw(edge, d)]
                        else:
                            key = _fabric_key(edge, direction)
                            if key not in link_data:
                                link_data[key] = [0.0,
                                                  _direction_bw(edge, direction)]
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
        elif isinstance(kernel, CommKernel) and len(participants) == 1 \
                and kernel.transferred_bytes > 0:
            local_mem = self._hardware.find_local_memory(device)
            fab = self._hardware.find_fabric(device, local_mem)
            read_bw = fab.dst_to_src_bandwidth_gbs * 1e3
            write_bw = fab.src_to_dst_bandwidth_gbs * 1e3
            if read_bw > 0 and write_bw > 0:
                mt = max(kernel.total_bytes / read_bw,
                         kernel.total_bytes / write_bw)
            key = _fabric_key(fab, 'fwd')
            if key not in link_data:
                link_data[key] = [0.0, fab.src_to_dst_bandwidth_gbs]

        if not (isinstance(kernel, CommKernel) and len(participants) >= 2
                and kernel.transferred_bytes > 0):
            xfer = max((b / (bw * 1e3)
                        for b, bw in link_data.values() if b > 0),
                       default=0.0)

        return ct, mt, alpha, xfer, link_data

    @staticmethod
    def _infer_dtype(kernel: Kernel) -> str:
        if hasattr(kernel, "w_dtype"):
            return kernel.w_dtype
        if hasattr(kernel, "dtype_"):
            return kernel.dtype_
        if kernel.inputs:
            return next(iter(kernel.inputs.values())).dtype
        return "bf16"
