"""DeepSeek V4 Pro inference — Simulation Phase."""

from rooflang.runtime.simulator import Simulator
from rooflang.runtime.trace_export import export_trace


def simulate(
    g, p, hw, trace_path="dsv4_pro_prefill.json", measurement_start=None,
):
    """Run DES simulator and export trace."""
    result = Simulator(
        g, p, hw, measurement_start=measurement_start).run()
    export_trace(result, trace_path)
    return result
