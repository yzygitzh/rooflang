"""Unit tests for HardwareGraph topology (find_fabric, find_local_memory)."""

import pytest

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


def _two_gpu_graph():
    """Two GPUs connected via NVSwitch, each with HBM."""
    g0 = Compute(name="gpu0", tflops={"bf16": 2250.0})
    g1 = Compute(name="gpu1", tflops={"bf16": 2250.0})
    nv = Compute(name="nvswitch")
    hbm0 = Memory(name="hbm0", capacity_gb=288.0)
    hbm1 = Memory(name="hbm1", capacity_gb=288.0)

    fab_g0_nv = FabricEdge(name="nvlink", src=g0, dst=nv,
                           src_to_dst_bandwidth_gbs=900.0,
                           dst_to_src_bandwidth_gbs=900.0,
                           is_full_duplex=True, alpha_us=0.5)
    fab_g1_nv = FabricEdge(name="nvlink", src=g1, dst=nv,
                           src_to_dst_bandwidth_gbs=900.0,
                           dst_to_src_bandwidth_gbs=900.0,
                           is_full_duplex=True, alpha_us=0.5)
    fab_g0_hbm = FabricEdge(name="hbm", src=g0, dst=hbm0,
                            src_to_dst_bandwidth_gbs=7750.0,
                            dst_to_src_bandwidth_gbs=7750.0,
                            is_full_duplex=False, alpha_us=0.5)
    fab_g1_hbm = FabricEdge(name="hbm", src=g1, dst=hbm1,
                            src_to_dst_bandwidth_gbs=7750.0,
                            dst_to_src_bandwidth_gbs=7750.0,
                            is_full_duplex=False, alpha_us=0.5)

    hw = HardwareGraph()
    for comp in [g0, g1, nv]:
        hw.add_node(comp)
    for mem in [hbm0, hbm1]:
        hw.add_node(mem)
    for edge in [fab_g0_nv, fab_g1_nv, fab_g0_hbm, fab_g1_hbm]:
        hw.add_edge(edge)

    return hw, g0, g1, nv, hbm0, hbm1, fab_g0_nv, fab_g0_hbm


class TestFindFabricDirect:
    def test_direct_connection(self):
        hw, g0, _, nv, _, _, fab_g0_nv, _ = _two_gpu_graph()
        result = hw.find_fabric(g0, nv)
        assert result is fab_g0_nv

    def test_direct_gpu_to_hbm(self):
        hw, g0, _, _, hbm0, _, _, fab_g0_hbm = _two_gpu_graph()
        result = hw.find_fabric(g0, hbm0)
        assert result is fab_g0_hbm


class TestFindFabricMultiHop:
    def test_gpu_to_gpu_via_nvswitch(self):
        hw, g0, g1, _, _, _, _, _ = _two_gpu_graph()
        fab = hw.find_fabric(g0, g1)
        assert fab.src_to_dst_bandwidth_gbs == 900.0
        assert fab.dst_to_src_bandwidth_gbs == 900.0
        assert fab.alpha_us == pytest.approx(1.0)
        assert fab.is_full_duplex is True

    def test_bottleneck_bandwidth(self):
        a = Compute(name="a")
        b = Compute(name="b")
        c = Compute(name="c")
        hw = HardwareGraph()
        for comp in [a, b, c]:
            hw.add_node(comp)
        hw.add_edge(FabricEdge(name="fast", src=a, dst=b,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=True, alpha_us=1.0))
        hw.add_edge(FabricEdge(name="slow", src=b, dst=c,
                               src_to_dst_bandwidth_gbs=100.0,
                               dst_to_src_bandwidth_gbs=100.0,
                               is_full_duplex=True, alpha_us=2.0))
        fab = hw.find_fabric(a, c)
        assert fab.src_to_dst_bandwidth_gbs == 100.0
        assert fab.alpha_us == pytest.approx(3.0)

    def test_half_duplex_propagates(self):
        a = Compute(name="a")
        b = Compute(name="b")
        c = Compute(name="c")
        hw = HardwareGraph()
        for comp in [a, b, c]:
            hw.add_node(comp)
        hw.add_edge(FabricEdge(name="f1", src=a, dst=b,
                               src_to_dst_bandwidth_gbs=500.0,
                               dst_to_src_bandwidth_gbs=500.0,
                               is_full_duplex=False, alpha_us=0.0))
        hw.add_edge(FabricEdge(name="f2", src=b, dst=c,
                               src_to_dst_bandwidth_gbs=500.0,
                               dst_to_src_bandwidth_gbs=500.0,
                               is_full_duplex=True, alpha_us=0.0))
        fab = hw.find_fabric(a, c)
        assert fab.is_full_duplex is False


class TestFindFabricErrors:
    def test_no_path(self):
        a = Compute(name="a")
        b = Compute(name="b")
        hw = HardwareGraph()
        hw.add_node(a)
        hw.add_node(b)
        with pytest.raises(ValueError, match="No path"):
            hw.find_fabric(a, b)

    def test_same_node(self):
        a = Compute(name="a")
        hw = HardwareGraph()
        hw.add_node(a)
        with pytest.raises(ValueError, match="same node"):
            hw.find_fabric(a, a)


class TestFindLocalMemory:
    def test_finds_highest_bw_memory(self):
        hw, g0, _, _, hbm0, _, _, _ = _two_gpu_graph()
        assert hw.find_local_memory(g0) is hbm0

    def test_no_memory_raises(self):
        g = Compute(name="gpu")
        hw = HardwareGraph()
        hw.add_node(g)
        with pytest.raises(ValueError, match="No memory"):
            hw.find_local_memory(g)

    def test_picks_higher_bw_when_multiple(self):
        g = Compute(name="gpu")
        slow = Memory(name="dram", capacity_gb=1024.0)
        fast = Memory(name="hbm", capacity_gb=288.0)
        hw = HardwareGraph()
        for comp in [g, slow, fast]:
            hw.add_node(comp)
        hw.add_edge(FabricEdge(name="pcie", src=g, dst=slow,
                               src_to_dst_bandwidth_gbs=64.0,
                               dst_to_src_bandwidth_gbs=64.0,
                               is_full_duplex=True))
        hw.add_edge(FabricEdge(name="hbm", src=g, dst=fast,
                               src_to_dst_bandwidth_gbs=7750.0,
                               dst_to_src_bandwidth_gbs=7750.0,
                               is_full_duplex=False))
        assert hw.find_local_memory(g) is fast
