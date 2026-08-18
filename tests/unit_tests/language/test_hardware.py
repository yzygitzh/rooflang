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

    def test_kind(self):
        c = Compute(name="accelerator", kind="gpu")
        assert c.kind == "gpu"

    def test_default_kind_none(self):
        c = Compute(name="custom")
        assert c.kind is None


class TestMemoryInit:
    def test_basic(self):
        m = Memory(name="hbm", capacity_gb=192.0)
        assert m.capacity_gb == 192.0

    def test_default_capacity_zero(self):
        m = Memory(name="dram")
        assert m.capacity_gb == 0.0

    def test_kind(self):
        m = Memory(name="memory", kind="dram")
        assert m.kind == "dram"


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


def _two_route_graph(route_a, route_b):
    """Two equal-hop routes, each described by two (fwd, rev, alpha) links."""
    src = Compute(name="src")
    dst = Compute(name="dst")
    mid_a = Compute(name="mid-a")
    mid_b = Compute(name="mid-b")
    hw = HardwareGraph()
    for component in [src, dst, mid_a, mid_b]:
        hw.add_node(component)

    routes = []
    for name, mid, specs in [
        ("route-a", mid_a, route_a),
        ("route-b", mid_b, route_b),
    ]:
        edges = []
        for index, (edge_src, edge_dst, spec) in enumerate([
            (src, mid, specs[0]),
            (mid, dst, specs[1]),
        ]):
            fwd_bw, rev_bw, alpha = spec
            edge = FabricEdge(
                name=f"{name}-{index}", src=edge_src, dst=edge_dst,
                src_to_dst_bandwidth_gbs=fwd_bw,
                dst_to_src_bandwidth_gbs=rev_bw,
                is_full_duplex=True, alpha_us=alpha,
            )
            hw.add_edge(edge)
            edges.append(edge)
        routes.append(edges)
    return hw, src, dst, routes[0], routes[1]


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

    def test_routing_weight_can_prefer_more_hops(self):
        hw, src, dst, _, fast = _two_route_graph(
            [(100.0, 100.0, 0.5), (100.0, 100.0, 0.5)],
            [(200.0, 200.0, 0.5), (200.0, 200.0, 0.5)],
        )
        direct = FabricEdge(
            name="direct", src=src, dst=dst,
            src_to_dst_bandwidth_gbs=1.0,
            dst_to_src_bandwidth_gbs=1.0,
            is_full_duplex=True, alpha_us=10.0,
        )
        hw.add_edge(direct)

        assert hw.find_fabric_path(src, dst) == fast

    def test_minimizes_sum_of_inverse_bandwidths(self):
        hw, src, dst, lower_weight, wider_bottleneck = _two_route_graph(
            [(100.0, 100.0, 0.5), (10.0, 10.0, 0.5)],
            [(15.0, 15.0, 0.5), (15.0, 15.0, 0.5)],
        )

        assert hw.find_fabric_path(src, dst) == lower_weight
        assert hw.find_fabric(src, dst).src_to_dst_bandwidth_gbs == 10.0
        assert lower_weight != wider_bottleneck

    def test_directional_bandwidth_can_select_different_routes(self):
        hw, src, dst, forward_route, reverse_route = _two_route_graph(
            [(100.0, 10.0, 0.5), (100.0, 10.0, 0.5)],
            [(50.0, 50.0, 0.5), (50.0, 50.0, 0.5)],
        )

        assert hw.find_fabric_path(src, dst) == forward_route
        assert hw.find_fabric_path(dst, src) == list(reversed(reverse_route))

    def test_parallel_fabric_uses_highest_bandwidth(self):
        src = Compute(name="src")
        mid = Compute(name="mid")
        dst = Compute(name="dst")
        hw = HardwareGraph()
        for component in [src, mid, dst]:
            hw.add_node(component)
        high_bw = FabricEdge(
            name="high-bw", src=src, dst=mid,
            src_to_dst_bandwidth_gbs=100.0,
            dst_to_src_bandwidth_gbs=100.0,
            is_full_duplex=True, alpha_us=10.0,
        )
        low_alpha = FabricEdge(
            name="low-alpha", src=src, dst=mid,
            src_to_dst_bandwidth_gbs=80.0,
            dst_to_src_bandwidth_gbs=80.0,
            is_full_duplex=True, alpha_us=1.0,
        )
        bottleneck = FabricEdge(
            name="bottleneck", src=mid, dst=dst,
            src_to_dst_bandwidth_gbs=50.0,
            dst_to_src_bandwidth_gbs=50.0,
            is_full_duplex=True, alpha_us=1.0,
        )
        for edge in [high_bw, low_alpha, bottleneck]:
            hw.add_edge(edge)

        assert hw.find_fabric_path(src, dst) == [high_bw, bottleneck]
        assert hw.find_fabric(src, dst).src_to_dst_bandwidth_gbs == 50.0
        assert hw.find_fabric(src, dst).alpha_us == 11.0

    def test_paths_are_cached_and_invalidated_by_topology_changes(self):
        hw, g0, g1, _, _, _, _, _, _ = _two_gpu_graph()
        hw._find_route.cache_clear()
        hw._find_fabric_path.cache_clear()
        hw._find_fabric_path_directed.cache_clear()

        assert len(hw.find_fabric_path(g0, g1)) == 2
        assert len(hw.find_fabric_path(g0, g1)) == 2
        assert hw._find_fabric_path.cache_info().hits == 1
        assert len(hw.find_fabric_path_directed(g0, g1)) == 2
        assert len(hw.find_fabric_path_directed(g0, g1)) == 2
        assert hw._find_fabric_path_directed.cache_info().hits == 1
        assert hw._find_route.cache_info().misses == 1
        assert hw._find_route.cache_info().hits == 1

        hw.add_edge(FabricEdge(
            name="direct", src=g0, dst=g1,
            src_to_dst_bandwidth_gbs=10_000.0,
            dst_to_src_bandwidth_gbs=10_000.0,
            is_full_duplex=True,
        ))

        assert len(hw.find_fabric_path(g0, g1)) == 1
        assert len(hw.find_fabric_path_directed(g0, g1)) == 1
        assert hw._find_route.cache_info().misses == 1


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
