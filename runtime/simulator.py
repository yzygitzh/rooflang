"""Roofline simulator — DES with resource contention.

Overlapping kernels on the same device share compute/memory by weighted
cap; overlapping comms on the same fabric share link BW equally.
"""

from __future__ import annotations

import heapq
from enum import Enum
from typing import Dict, List, Set, Tuple

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.comm import AllToAll, CommKernel, Recv, Send
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement


class Bound(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"


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
    def __init__(self, trace: List[TraceEntry], total_time_us: float):
        self.trace = trace
        self.total_time_us = total_time_us


class RunningKernel:
    """Running kernel state in the DES."""

    __slots__ = ("kernel", "device", "stream", "resource_cap",
                 "compute_time", "memory_time", "network_time",
                 "fabric_edges", "start_us",
                 "cp", "mp", "np", "seg_start", "dev_share", "net_share")

    def __init__(self, kernel, device, stream, cap, ct, mt, nt, fabs, t0):
        self.kernel = kernel
        self.device = device
        self.stream = stream
        self.resource_cap = cap
        self.compute_time = ct
        self.memory_time = mt
        self.network_time = nt
        self.fabric_edges = fabs
        self.start_us = t0
        self.cp = self.mp = self.np = 0.0
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
        if self.network_time > 0:
            self.np += dt * self.net_share / self.network_time
        self.seg_start = now

    def eta(self) -> float:
        worst = 0.0
        if self.compute_time > 0 and self.cp < 1.0:
            worst = max(worst, (1.0 - self.cp) * self.compute_time / self.dev_share)
        if self.memory_time > 0 and self.mp < 1.0:
            worst = max(worst, (1.0 - self.mp) * self.memory_time / self.dev_share)
        if self.network_time > 0 and self.np < 1.0:
            worst = max(worst, (1.0 - self.np) * self.network_time / self.net_share)
        return self.seg_start + worst

    def bound(self) -> Bound:
        ct = self.compute_time / self.dev_share if self.compute_time > 0 else 0.0
        mt = self.memory_time / self.dev_share if self.memory_time > 0 else 0.0
        nt = self.network_time / self.net_share if self.network_time > 0 else 0.0
        best = max(ct, mt, nt)
        if best == nt and nt > 0:
            return Bound.NETWORK
        if best == ct and ct > 0:
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

        for kernel in self._graph.topological_sort():
            if not list(self._graph._dag.predecessors(kernel)):
                dev, stream, _, _ = self._resolve(kernel)
                self._push(self._stream_end.get((dev, stream), 0.0), "start",
                           RunningKernel(kernel, dev, stream, 0, 0, 0, 0, [], 0))

        while self._eq:
            now, _, typ, rk = heapq.heappop(self._eq)
            if rk.kernel in self._completed:
                continue
            if typ == "start":
                self._start_kernel(rk.kernel, now)
            else:
                rk.advance_to(now)
                if rk.eta() > now + 1e-9:
                    self._push(rk.eta(), "end", rk)
                    continue
                self._completed.add(rk.kernel)
                self._kernel_end[rk.kernel] = now
                self._stream_end[(rk.device, rk.stream)] = now
                self._trace.append(TraceEntry(
                    rk.kernel, rk.device, rk.stream,
                    rk.start_us, now, rk.bound()))
                self._finish_kernel(rk, now)
                for succ in self._graph._dag.successors(rk.kernel):
                    if succ in self._completed:
                        continue
                    preds = list(self._graph._dag.predecessors(succ))
                    if all(p in self._completed for p in preds):
                        t = max(self._kernel_end[p] for p in preds)
                        dev, strm, _, _ = self._resolve(succ)
                        sr = self._stream_end.get((dev, strm), 0.0)
                        self._push(max(t, sr), "start",
                                   RunningKernel(succ, dev, strm, 0, 0, 0, 0, [], 0))

        total = max(self._kernel_end.values()) if self._kernel_end else 0.0
        return SimulationResult(self._trace, total)

    # ── Event queue ─────────────────────────────────────────────────

    def _push(self, t, typ, rk):
        heapq.heappush(self._eq, (t, self._eid, typ, rk))
        self._eid += 1

    # ── Kernel lifecycle ────────────────────────────────────────────

    def _start_kernel(self, kernel: Kernel, now: float):
        dev, stream, cap, parts = self._resolve(kernel)
        ct, mt, nt, fabs = self._base_times(kernel, dev, parts)
        rk = RunningKernel(kernel, dev, stream, cap, ct, mt, nt, fabs, now)
        self._on_dev.setdefault(dev, []).append(rk)
        for fab in fabs:
            self._on_fab.setdefault(fab, []).append(rk)
        self._advance_peers(rk, now)
        self._recompute_shares(rk)
        self._push(rk.eta(), "end", rk)
        self._resched_peers(rk)

    def _finish_kernel(self, rk: RunningKernel, now: float):
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

    # ── Resource sharing ────────────────────────────────────────────

    def _recompute_shares(self, rk: RunningKernel):
        peers = self._on_dev.get(rk.device, [])
        tc = sum(p.resource_cap for p in peers)
        s = 1.0 / max(1.0, tc)
        for p in peers:
            p.dev_share = p.resource_cap * s
        for fab in rk.fabric_edges:
            fp = self._on_fab.get(fab, [])
            ns = 1.0 / max(1, len(fp))
            for p in fp:
                p.net_share = ns

    def _advance_peers(self, rk: RunningKernel, now: float):
        for p in self._on_dev.get(rk.device, []):
            p.advance_to(now)
        for fab in rk.fabric_edges:
            for p in self._on_fab.get(fab, []):
                p.advance_to(now)

    def _resched_peers(self, rk: RunningKernel):
        for p in self._on_dev.get(rk.device, []):
            if p is not rk:
                self._push(p.eta(), "end", p)
        for fab in rk.fabric_edges:
            for p in self._on_fab.get(fab, []):
                if p is not rk:
                    self._push(p.eta(), "end", p)

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
        primary = pd[0] if pd else sd[0]
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
        ct = kernel.flops / peak if peak > 0 else 0.0

        mt = 0.0
        for t in list(kernel.inputs.values()) + list(kernel.weights.values()) + \
                 list(kernel.outputs.values()):
            mem = self._placement.get_tensor_memory(t) \
                if self._placement.get_tensor_memory(t) \
                else self._hardware.find_local_memory(device)
            fab = self._hardware.find_fabric(device, mem)
            bw = fab.src_to_dst_bandwidth_gbs * 1e3
            if bw > 0:
                mt += t.size_bytes / bw

        nt = 0.0
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
            max_alpha = max(
                self._hardware.find_fabric(d1, d2).alpha_us
                for i, d1 in enumerate(participants)
                for d2 in participants[i + 1:])
            nt = max_alpha + kernel.transferred_bytes / (eff_bw * 1e3)

        return ct, mt, nt, fab_edges

    @staticmethod
    def _infer_dtype(kernel: Kernel) -> str:
        if hasattr(kernel, "w_dtype"):
            return kernel.w_dtype
        if hasattr(kernel, "dtype_"):
            return kernel.dtype_
        if kernel.inputs:
            return next(iter(kernel.inputs.values())).dtype
        return "bf16"
