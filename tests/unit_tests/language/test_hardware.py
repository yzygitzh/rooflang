"""Unit tests for rooflang.language.hardware (component nodes)."""

import pytest

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


# ── Component tests ──────────────────────────────────────────────────


class TestComputeInit:
    def test_basic(self):
        c = Compute(name="gpu0", tflops={"bf16": 2250.0})
        assert c.name == "gpu0"
        assert c.tflops["bf16"] == 2250.0

    def test_default_tflops_empty(self):
        c = Compute(name="cpu")
        assert c.tflops == {}


class TestMemoryInit:
    def test_basic(self):
        m = Memory(name="hbm", capacity_gb=192.0)
        assert m.capacity_gb == 192.0

    def test_default_capacity_zero(self):
        m = Memory(name="dram")
        assert m.capacity_gb == 0.0


class TestFabricEdgeTransferTimeUs:
    def test_full_duplex_time(self):
        f = FabricEdge(
            name="nvlink",
            src=Compute(name="g0"),
            dst=Compute(name="g1"),
            src_to_dst_bandwidth_gbs=900.0,
            dst_to_src_bandwidth_gbs=900.0,
            is_full_duplex=True,
            alpha_us=1.0,
        )
        t = f.transfer_time_us(src_to_dst_bytes=900e3, dst_to_src_bytes=450e3)
        t_fwd = 900e3 / (900.0 * 1e3)
        t_rev = 450e3 / (900.0 * 1e3)
        assert t == pytest.approx(1.0 + max(t_fwd, t_rev))

    def test_half_duplex_time(self):
        f = FabricEdge(
            name="pcie",
            src=Compute(name="g0"),
            dst=Memory(name="dram"),
            src_to_dst_bandwidth_gbs=64.0,
            dst_to_src_bandwidth_gbs=64.0,
            is_full_duplex=False,
            alpha_us=0.5,
        )
        t = f.transfer_time_us(src_to_dst_bytes=64e3, dst_to_src_bytes=64e3)
        t_fwd = 64e3 / (64.0 * 1e3)
        t_rev = 64e3 / (64.0 * 1e3)
        assert t == pytest.approx(0.5 + t_fwd + t_rev)

    def test_zero_bytes_no_time(self):
        f = FabricEdge(
            name="link",
            src=Compute(name="a"),
            dst=Compute(name="b"),
            src_to_dst_bandwidth_gbs=100.0,
            dst_to_src_bandwidth_gbs=100.0,
            is_full_duplex=True,
        )
        assert f.transfer_time_us(0.0, 0.0) == 0.0


class TestHardwareGraphConstruction:
    def test_construction(self):
        g = Compute(name="gpu0")
        m = Memory(name="hbm0", capacity_gb=80.0)
        hw = HardwareGraph()
        hw.add_node(g)
        hw.add_node(m)
        hw.add_edge(FabricEdge(
            name="bus", src=g, dst=m,
            src_to_dst_bandwidth_gbs=3000.0,
            dst_to_src_bandwidth_gbs=3000.0,
            is_full_duplex=True,
        ))
        assert g in hw.nodes
        assert m in hw.nodes

    def test_duplicate_name_rejected(self):
        hw = HardwareGraph()
        hw.add_node(Compute(name="gpu0"))
        with pytest.raises(ValueError, match="Duplicate hardware component name"):
            hw.add_node(Memory(name="gpu0", capacity_gb=80.0))


# ── Helpers ─────────────────────────────────────────────────────────


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

    return hw, g0, g1, nv, hbm0, hbm1, fab_g0_nv, fab_g1_nv, fab_g0_hbm


# ── find_fabric_path tests ──────────────────────────────────────────


class TestFindFabricPath:
    def test_direct_returns_single_edge(self):
        hw, g0, _, nv, _, _, fab_g0_nv, _, _ = _two_gpu_graph()
        path = hw.find_fabric_path(g0, nv)
        assert path == [fab_g0_nv]

    def test_multi_hop_returns_all_edges(self):
        hw, g0, g1, nv, _, _, fab_g0_nv, _, _ = _two_gpu_graph()
        path = hw.find_fabric_path(g0, g1)
        assert len(path) == 2
        assert fab_g0_nv in path

    def test_same_node_returns_empty(self):
        hw, g0, _, _, _, _, _, _, _ = _two_gpu_graph()
        assert hw.find_fabric_path(g0, g0) == []

    def test_no_path_raises(self):
        a = Compute(name="a")
        b = Compute(name="b")
        hw = HardwareGraph()
        hw.add_node(a)
        hw.add_node(b)
        with pytest.raises(ValueError, match="No path"):
            hw.find_fabric_path(a, b)


# ── find_aggregate_bandwidth tests ──────────────────────────────────


class TestFindAggregateBandwidth:
    def test_single_device_returns_inf(self):
        hw, g0, _, _, _, _, _, _, _ = _two_gpu_graph()
        assert hw.find_aggregate_bandwidth([g0]) == float("inf")

    def test_two_gpus_via_nvswitch(self):
        hw, g0, g1, _, _, _, _, _, _ = _two_gpu_graph()
        assert hw.find_aggregate_bandwidth([g0, g1]) == 900.0

    def test_asymmetric_picks_min(self):
        a = Compute(name="a")
        b = Compute(name="b")
        c = Compute(name="c")
        hw = HardwareGraph()
        for comp in [a, b, c]:
            hw.add_node(comp)
        hw.add_edge(FabricEdge(name="fast", src=a, dst=b,
                               src_to_dst_bandwidth_gbs=1000.0,
                               dst_to_src_bandwidth_gbs=1000.0,
                               is_full_duplex=True))
        hw.add_edge(FabricEdge(name="slow", src=b, dst=c,
                               src_to_dst_bandwidth_gbs=100.0,
                               dst_to_src_bandwidth_gbs=100.0,
                               is_full_duplex=True))
        assert hw.find_aggregate_bandwidth([a, b, c]) == 100.0


# ── find_aggregate_latency tests ────────────────────────────────────


class TestFindAggregateLatency:
    def test_single_device_returns_zero(self):
        hw, g0, _, _, _, _, _, _, _ = _two_gpu_graph()
        assert hw.find_aggregate_latency([g0]) == 0.0

    def test_two_gpus_via_nvswitch(self):
        hw, g0, g1, _, _, _, _, _, _ = _two_gpu_graph()
        assert hw.find_aggregate_latency([g0, g1]) == pytest.approx(1.0)

    def test_picks_max_latency_pair(self):
        a = Compute(name="a")
        b = Compute(name="b")
        c = Compute(name="c")
        hw = HardwareGraph()
        for comp in [a, b, c]:
            hw.add_node(comp)
        hw.add_edge(FabricEdge(name="f1", src=a, dst=b,
                               src_to_dst_bandwidth_gbs=500.0,
                               dst_to_src_bandwidth_gbs=500.0,
                               is_full_duplex=True, alpha_us=1.0))
        hw.add_edge(FabricEdge(name="f2", src=b, dst=c,
                               src_to_dst_bandwidth_gbs=500.0,
                               dst_to_src_bandwidth_gbs=500.0,
                               is_full_duplex=True, alpha_us=2.0))
        assert hw.find_aggregate_latency([a, b, c]) == pytest.approx(3.0)


# ── HardwareGraph edge errors ────────────────────────────────────────


class TestHardwareGraphAddEdge:
    def test_src_not_in_graph_raises(self):
        hw = HardwareGraph()
        g0 = Compute(name="g0")
        g1 = Compute(name="g1")
        hw.add_node(g1)
        with pytest.raises(ValueError, match="Node not in graph"):
            hw.add_edge(FabricEdge(name="f", src=g0, dst=g1,
                                   src_to_dst_bandwidth_gbs=1.0,
                                   dst_to_src_bandwidth_gbs=1.0,
                                   is_full_duplex=False))

    def test_dst_not_in_graph_raises(self):
        hw = HardwareGraph()
        g0 = Compute(name="g0")
        g1 = Compute(name="g1")
        hw.add_node(g0)
        with pytest.raises(ValueError, match="Node not in graph"):
            hw.add_edge(FabricEdge(name="f", src=g0, dst=g1,
                                   src_to_dst_bandwidth_gbs=1.0,
                                   dst_to_src_bandwidth_gbs=1.0,
                                   is_full_duplex=False))

    def test_multi_fabric_picks_best(self):
        hw = HardwareGraph()
        g0 = Compute(name="g0")
        hbm = Memory(name="hbm", capacity_gb=80.0)
        hw.add_node(g0)
        hw.add_node(hbm)
        hw.add_edge(FabricEdge(name="slow", src=g0, dst=hbm,
                               src_to_dst_bandwidth_gbs=1.0,
                               dst_to_src_bandwidth_gbs=1.0,
                               is_full_duplex=False))
        hw.add_edge(FabricEdge(name="fast", src=g0, dst=hbm,
                               src_to_dst_bandwidth_gbs=10.0,
                               dst_to_src_bandwidth_gbs=10.0,
                               is_full_duplex=False))
        fab = hw.find_fabric(g0, hbm)
        assert fab.name == "fast"


class TestHardwareGraphFindFabric:
    def test_node_not_in_graph_raises(self):
        hw = HardwareGraph()
        g0 = Compute(name="g0")
        g1 = Compute(name="g1")
        hw.add_node(g0)
        with pytest.raises(ValueError, match="No path"):
            hw.find_fabric(g0, g1)
