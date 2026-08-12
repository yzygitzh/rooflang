"""Unit tests for rooflang.runtime.trace_export."""

import json

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.placement import Placement
from rooflang.language.tensor import Tensor
from rooflang.runtime.simulator import Simulator
from rooflang.runtime.trace_export import export_trace


class SyntheticKernel(Kernel):
    def __init__(self, flops_val=0.0, **kwargs):
        super().__init__(**kwargs)
        self._flops_val = flops_val

    @property
    def flops(self):
        return self._flops_val


def _hw():
    gpu = Compute(name="gpu0", tflops={"bf16": 1.0})
    hbm = Memory(name="hbm", capacity_gb=80.0)
    hw = HardwareGraph()
    hw.add_node(gpu)
    hw.add_node(hbm)
    hw.add_edge(FabricEdge(name="link", src=gpu, dst=hbm,
                           src_to_dst_bandwidth_gbs=1000.0,
                           dst_to_src_bandwidth_gbs=1000.0,
                           is_full_duplex=False))
    return hw, gpu, hbm


class TestExportTrace:
    def test_export_creates_valid_json(self, tmp_path):
        hw, gpu, hbm = _hw()
        t = Tensor("bf16", (4,))
        k = SyntheticKernel(flops_val=1e6, inputs={"x": t})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = Simulator(g, p, hw).run()

        out = str(tmp_path / "trace.json")
        export_trace(result, out)

        with open(out) as f:
            data = json.load(f)
        assert "traceEvents" in data
        assert "otherData" in data
        assert len(data["traceEvents"]) >= 1

    def test_event_fields_match_trace(self, tmp_path):
        hw, gpu, hbm = _hw()
        t = Tensor("bf16", (4,))
        k = SyntheticKernel(flops_val=1e6, inputs={"x": t})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = Simulator(g, p, hw).run()
        entry = result.trace[0]

        out = str(tmp_path / "trace.json")
        export_trace(result, out)

        with open(out) as f:
            data = json.load(f)
        kernel_events = [e for e in data["traceEvents"] if e["ph"] == "X"]
        assert len(kernel_events) == 1
        ev = kernel_events[0]
        assert ev["name"] == "SyntheticKernel"
        assert ev["ts"] == pytest.approx(entry.start_us)
        assert ev["dur"] == pytest.approx(entry.end_us - entry.start_us)
        assert ev["pid"] == "gpu0"
        assert ev["tid"] == "stream0"
        assert ev["cat"] == entry.bound.value
        assert ev["args"]["weight_bytes"] == 0.0
        assert ev["args"]["mfu"] >= 0.0
        assert ev["args"]["input_bandwidth_gbs"] >= 0.0
        assert ev["args"]["weight_bandwidth_gbs"] >= 0.0
        assert ev["args"]["output_bandwidth_gbs"] >= 0.0

    def test_metadata_events_present(self, tmp_path):
        hw, gpu, hbm = _hw()
        t = Tensor("bf16", (4,))
        k = SyntheticKernel(flops_val=1e6, inputs={"x": t})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = Simulator(g, p, hw).run()

        out = str(tmp_path / "trace.json")
        export_trace(result, out)

        with open(out) as f:
            data = json.load(f)
        meta = [e for e in data["traceEvents"] if e["ph"] == "M"]
        assert len(meta) == 1
        assert meta[0]["args"]["name"] == "gpu0"

    def test_empty_trace(self, tmp_path):
        hw, gpu, hbm = _hw()
        g = ComputeGraph()
        p = Placement(hardware=hw)
        result = Simulator(g, p, hw).run()

        out = str(tmp_path / "trace.json")
        export_trace(result, out)

        with open(out) as f:
            data = json.load(f)
        assert data["traceEvents"] == []
        assert data["otherData"]["total_time_us"] == 0.0

    def test_other_data_contains_summary(self, tmp_path):
        hw, gpu, hbm = _hw()
        t = Tensor("bf16", (4,))
        k = SyntheticKernel(flops_val=1e6, inputs={"x": t})
        g = ComputeGraph()
        g.add_kernel(k)
        p = Placement(hardware=hw)
        p.set_kernel_device(k, gpu)
        result = Simulator(g, p, hw).run()

        out = str(tmp_path / "trace.json")
        export_trace(result, out)

        with open(out) as f:
            data = json.load(f)
        other = data["otherData"]
        assert other["total_time_us"] == pytest.approx(result.total_time_us)
        assert other["measurement_start_us"] == pytest.approx(
            result.measurement_start_us)
        assert other["measured_time_us"] == pytest.approx(
            result.measured_time_us)
        assert isinstance(other["peak_memory"], list)
        assert other["peak_memory"][0]["name"] == "hbm"
        assert other["peak_memory"][0]["bytes"] == result.peak_memory[hbm]
