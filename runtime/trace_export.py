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
        dur_us = entry.end_us - entry.start_us
        dur_s = dur_us / 1e6 if dur_us > 0 else 0.0
        kernel = entry.kernel
        peak_flops = max(entry.device.tflops.values(), default=0.0) * 1e12
        mfu = kernel.flops / (peak_flops * dur_s) if peak_flops > 0 and dur_s > 0 else 0.0
        input_bw = kernel.input_bytes / (dur_s * 1e9) if dur_s > 0 else 0.0
        weight_bw = kernel.weight_bytes / (dur_s * 1e9) if dur_s > 0 else 0.0
        output_bw = kernel.output_bytes / (dur_s * 1e9) if dur_s > 0 else 0.0
        inputs = {k: list(t.shape) for k, t in kernel.inputs.items()}
        weights = {k: list(t.shape) for k, t in kernel.weights.items()}
        outputs = {k: list(t.shape) for k, t in kernel.outputs.items()}
        events.append({
            "name": type(kernel).__name__,
            "cat": entry.bound.value,
            "ph": "X",
            "ts": entry.start_us,
            "dur": dur_us,
            "pid": entry.device.name,
            "tid": f"stream{entry.stream}",
            "args": {
                "flops": kernel.flops,
                "input_bytes": kernel.input_bytes,
                "weight_bytes": kernel.weight_bytes,
                "output_bytes": kernel.output_bytes,
                "input_bandwidth_gbs": input_bw,
                "weight_bandwidth_gbs": weight_bw,
                "output_bandwidth_gbs": output_bw,
                "mfu": mfu,
                "bound": entry.bound.value,
                "inputs": inputs,
                "weights": weights,
                "outputs": outputs,
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
