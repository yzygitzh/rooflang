"""Unit tests for rooflang.language.placement (Placement + DeviceAssignment)."""

import pytest

from rooflang.language.placement import Placement, DeviceAssignment
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.graph import ComputeGraph
from rooflang.language.tensor import Tensor


# ── DeviceAssignment tests ───────────────────────────────────────────


class TestDeviceAssignment:
    def test_basic(self):
        gpu = Compute(name="gpu0", tflops={"bf16": 2250.0})
        da = DeviceAssignment(device=gpu, stream=1, resource_cap=0.5)
        assert da.device is gpu
        assert da.stream == 1
        assert da.resource_cap == 0.5

    def test_defaults(self):
        gpu = Compute(name="gpu0")
        da = DeviceAssignment(device=gpu)
        assert da.stream == 0
        assert da.resource_cap == 1.0


# ── Placement.set tests ──────────────────────────────────────────────


class TestPlacementSet:
    def test_basic(self):
        p = Placement()
        gpu = Compute(name="gpu0")
        k = Kernel()
        p.set(k, gpu, stream=2, resource_cap=0.7)
        da = p.get(k)
        assert da.device is gpu
        assert da.stream == 2
        assert da.resource_cap == 0.7

    def test_default_stream_and_cap(self):
        p = Placement()
        gpu = Compute(name="gpu0")
        k = Kernel()
        p.set(k, gpu)
        da = p.get(k)
        assert da.stream == 0
        assert da.resource_cap == 1.0

    def test_overwrite(self):
        p = Placement()
        gpu0 = Compute(name="gpu0")
        gpu1 = Compute(name="gpu1")
        k = Kernel()
        p.set(k, gpu0, stream=0)
        p.set(k, gpu1, stream=1)
        assert p.get(k).device is gpu1
        assert p.get(k).stream == 1

    def test_resource_cap_zero_raises(self):
        p = Placement()
        with pytest.raises(ValueError, match="resource_cap must be in"):
            p.set(Kernel(), Compute(name="g"), resource_cap=0.0)

    def test_resource_cap_negative_raises(self):
        p = Placement()
        with pytest.raises(ValueError, match="resource_cap must be in"):
            p.set(Kernel(), Compute(name="g"), resource_cap=-0.1)

    def test_resource_cap_above_one_raises(self):
        p = Placement()
        with pytest.raises(ValueError, match="resource_cap must be in"):
            p.set(Kernel(), Compute(name="g"), resource_cap=1.01)

    def test_resource_cap_one_ok(self):
        p = Placement()
        k = Kernel()
        p.set(k, Compute(name="g"), resource_cap=1.0)
        assert p.get(k).resource_cap == 1.0


# ── Placement.get tests ──────────────────────────────────────────────


class TestPlacementGet:
    def test_unplaced_raises(self):
        p = Placement()
        with pytest.raises(KeyError, match="Kernel not placed"):
            p.get(Kernel())


# ── Placement.placed_kernels tests ───────────────────────────────────


class TestPlacedKernels:
    def test_empty(self):
        assert Placement().placed_kernels == frozenset()

    def test_after_set(self):
        p = Placement()
        k1, k2 = Kernel(), Kernel()
        gpu = Compute(name="gpu0")
        p.set(k1, gpu)
        p.set(k2, gpu, stream=1)
        assert p.placed_kernels == frozenset({k1, k2})


# ── Placement.validate tests ─────────────────────────────────────────


class TestPlacementValidate:
    def test_all_placed_passes(self):
        p = Placement()
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        k1 = Kernel(inputs={"x": Tensor("bf16", (4,))},
                    outputs={"y": Tensor("bf16", (4,))})
        k2 = Kernel(inputs={"a": Tensor("bf16", (4,))},
                    outputs={"b": Tensor("bf16", (4,))})
        g.add_kernel(k1)
        g.add_kernel(k2)
        p.set(k1, gpu)
        p.set(k2, gpu, stream=1)
        p.validate(g)

    def test_unplaced_raises(self):
        p = Placement()
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        k1 = Kernel()
        k2 = Kernel()
        g.add_kernel(k1)
        g.add_kernel(k2)
        p.set(k1, gpu)
        with pytest.raises(ValueError, match="Unplaced kernels"):
            p.validate(g)

    def test_extraneous_raises(self):
        p = Placement()
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        k1 = Kernel()
        k2 = Kernel()
        g.add_kernel(k1)
        p.set(k1, gpu)
        p.set(k2, gpu)
        with pytest.raises(ValueError, match="Extraneous placements"):
            p.validate(g)

    def test_empty_graph_passes(self):
        Placement().validate(ComputeGraph())
