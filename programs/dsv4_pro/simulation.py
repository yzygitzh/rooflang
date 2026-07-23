"""DeepSeek V4 Pro inference — Simulation Phase."""

from rooflang.runtime.simulator import Simulator
from rooflang.runtime.trace_export import export_trace


def simulate(g, p, hw, trace_path="dsv4_pro_prefill.json"):
    """Run DES simulator and export trace."""
    result = Simulator(g, p, hw).run()
    export_trace(result, trace_path)
    return result
