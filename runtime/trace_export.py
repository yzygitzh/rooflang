"""Export SimulationResult to Google Trace Event Format (JSON).

Output can be loaded in chrome://tracing or Perfetto UI.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from rooflang.runtime.simulator import SimulationResult


def export_trace(result: SimulationResult, path: str) -> None:
    """Write simulation trace as Google Trace Event Format JSON."""
    events: List[Dict[str, Any]] = []

    devices = {e.device for e in result.trace}
    for dev in devices:
        events.append({
            "name": "process_name", "ph": "M",
            "pid": dev.name, "tid": 0,
            "args": {"name": dev.name},
        })

    for entry in result.trace:
        events.append({
            "name": type(entry.kernel).__name__,
            "cat": entry.bound.value,
            "ph": "X",
            "ts": entry.start_us,
            "dur": entry.end_us - entry.start_us,
            "pid": entry.device.name,
            "tid": f"stream{entry.stream}",
            "args": {
                "flops": entry.kernel.flops,
                "input_bytes": entry.kernel.input_bytes,
                "output_bytes": entry.kernel.output_bytes,
                "bound": entry.bound.value,
            },
        })

    peak_memory = [{"name": mem.name, "bytes": bytes_val}
                   for mem, bytes_val in result.peak_memory.items()]

    output = {
        "traceEvents": events,
        "otherData": {
            "total_time_us": result.total_time_us,
            "peak_memory": peak_memory,
        },
    }

    with open(path, "w") as f:
        json.dump(output, f)
