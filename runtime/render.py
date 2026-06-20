"""Primitive row dataclass and renderers (Markdown table + JSON).

A PrimitiveRow wraps a Kernel with metadata the enumerator provides (name,
layer, group, phase) and derives roofline projections against a HardwareSpec.

The renderer takes a list of PrimitiveRows (one phase or full model) and
produces:
  - Markdown: per-primitive table + totals row + bottleneck summary.
  - JSON: structured dict suitable for downstream tooling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.hardware import HardwareSpec


@dataclass
class PrimitiveRow:
    """One row in the roofline output table.

    Constructed by the enumerator for each primitive it emits. Roofline
    projections (arith_intensity, roofline_tflops, compute_time_s) are
    derived from the kernel metrics + hardware spec via `project()`.
    """
    name: str
    layer: int
    phase: str          # "fwd" | "bwd" | "opt"
    kernel: Kernel
    dtype_compute: str  # dtype used for peak TFLOPS lookup

    arith_intensity: float = field(init=False, default=0.0)
    roofline_tflops: float = field(init=False, default=0.0)
    compute_time_s: float = field(init=False, default=0.0)
    mem_time_s: float = field(init=False, default=0.0)
    bound: str = field(init=False, default="")
    weight_mem: float = field(init=False, default=0.0)
    input_mem: float = field(init=False, default=0.0)
    output_mem: float = field(init=False, default=0.0)

    def project(self, hw: HardwareSpec) -> None:
        """Derive roofline projections and memory footprint from kernel + hw.

        Memory footprint (per-rank, assuming enumerator already sharded dims):
          weight_mem: persistent weight storage on this rank (lives for the
                      entire step; shared across fwd/bwd invocations).
          input_mem:  activation memory consumed (must be live when this
                      primitive executes; freed after consumption).
          output_mem: activation memory produced (must stay live until the
                      downstream consumer runs, or until backward if saved
                      for gradient computation).
        """
        k = self.kernel
        if k.transferred_bytes > 0:
            self.arith_intensity = k.flops / k.transferred_bytes
        else:
            self.arith_intensity = float("inf")

        peak = hw.peak_tflops.get(self.dtype_compute, 0.0)
        bw = hw.peak_bw_gbs

        if bw > 0:
            self.roofline_tflops = min(peak, self.arith_intensity * bw / 1e3)
        else:
            self.roofline_tflops = peak

        if peak > 0:
            self.compute_time_s = k.flops / (peak * 1e12)
        if bw > 0:
            self.mem_time_s = k.transferred_bytes / (bw * 1e9)

        if self.compute_time_s >= self.mem_time_s:
            self.bound = "compute"
        else:
            self.bound = "memory"

        self.weight_mem = k.weight_bytes
        self.input_mem = k.input_bytes
        self.output_mem = k.output_bytes

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "layer": self.layer,
            "phase": self.phase,
            "dtype_compute": self.dtype_compute,
            "arith_intensity": self.arith_intensity,
            "roofline_tflops": self.roofline_tflops,
            "compute_time_s": self.compute_time_s,
            "mem_time_s": self.mem_time_s,
            "bound": self.bound,
            "weight_mem": self.weight_mem,
            "input_mem": self.input_mem,
            "output_mem": self.output_mem,
        }
        d.update(self.kernel.to_dict())
        return d


@dataclass
class PhaseSummary:
    """Rolled-up totals for one phase (or the full model)."""
    phase: str
    total_flops: float = 0.0
    total_bytes: float = 0.0
    total_compute_time_s: float = 0.0
    total_mem_time_s: float = 0.0
    total_weight_mem: float = 0.0
    total_input_mem: float = 0.0
    total_output_mem: float = 0.0
    n_ops: int = 0

    @property
    def time_s(self) -> float:
        return max(self.total_compute_time_s, self.total_mem_time_s)

    @property
    def realized_tflops(self) -> float:
        t = self.time_s
        if t > 0:
            return self.total_flops / t / 1e12
        return 0.0

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "total_flops": self.total_flops,
            "total_bytes": self.total_bytes,
            "total_compute_time_s": self.total_compute_time_s,
            "total_mem_time_s": self.total_mem_time_s,
            "total_weight_mem": self.total_weight_mem,
            "total_input_mem": self.total_input_mem,
            "total_output_mem": self.total_output_mem,
            "time_s": self.time_s,
            "realized_tflops": self.realized_tflops,
            "n_ops": self.n_ops,
        }


def summarize(rows: List[PrimitiveRow], phase: str) -> PhaseSummary:
    """Roll up a list of PrimitiveRows into a PhaseSummary."""
    s = PhaseSummary(phase=phase)
    for r in rows:
        s.total_flops += r.kernel.flops
        s.total_bytes += r.kernel.transferred_bytes
        s.total_compute_time_s += r.compute_time_s
        s.total_mem_time_s += r.mem_time_s
        s.total_weight_mem += r.weight_mem
        s.total_input_mem += r.input_mem
        s.total_output_mem += r.output_mem
        s.n_ops += 1
    return s


# -- Markdown renderer --------------------------------------------------------

_MD_HEADER = (
    "| # | name | layer | dtype | AI | roof TFLOPS | "
    "compute(s) | mem(s) | bound |"
)
_MD_SEP = (
    "|---|------|-------|-------|---:|------------:|"
    "----------:|-------:|-------|"
)


def _fmt(x: float, digits: int = 4) -> str:
    if x == 0.0:
        return "0"
    if x >= 1e6:
        return f"{x:.2e}"
    return f"{x:.{digits}g}"


def render_md(rows: List[PrimitiveRow], title: str, hw: HardwareSpec) -> str:
    """Render a phase's PrimitiveRows as a Markdown table with totals."""
    lines = [f"## {title}", "", _MD_HEADER, _MD_SEP]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | {r.name} | {r.layer} | {r.dtype_compute} "
            f"| {_fmt(r.arith_intensity)} | {_fmt(r.roofline_tflops)} "
            f"| {_fmt(r.compute_time_s)} | {_fmt(r.mem_time_s)} "
            f"| {r.bound} |"
        )
    s = summarize(rows, title)
    lines.append("")
    lines.append(
        f"**Totals:** {_fmt(s.total_flops)} FLOP, "
        f"{_fmt(s.total_bytes)} B transferred, "
        f"{_fmt(s.time_s)} s, "
        f"{_fmt(s.realized_tflops)} TFLOPS realized"
    )
    return "\n".join(lines)


# -- JSON renderer ------------------------------------------------------------

def render_json(rows: List[PrimitiveRow], phase: str) -> dict:
    """Render a phase's PrimitiveRows as a JSON-serializable dict."""
    s = summarize(rows, phase)
    return {
        "phase": phase,
        "ops": [r.to_dict() for r in rows],
        "totals": s.to_dict(),
    }
