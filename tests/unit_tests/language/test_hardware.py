"""Unit tests for rooflang.language.hardware (component + spec)."""

import pytest

from rooflang.language.hardware.component import (
    Compute, Memory, Fabric, Cluster,
)
from rooflang.language.hardware.spec import (
    HardwareSpec, LinkSpec, InterNodeSpec,
    hardware_spec, collective_bytes, collective_time, HW_B300,
)


# ── Component tests ──────────────────────────────────────────────────


class TestCompute:
    def test_basic(self):
        c = Compute(name="gpu0", tflops={"bf16": 2250.0})
        assert c.name == "gpu0"
        assert c.tflops["bf16"] == 2250.0

    def test_default_tflops_empty(self):
        c = Compute(name="cpu")
        assert c.tflops == {}


class TestMemory:
    def test_basic(self):
        m = Memory(name="hbm", capacity_gb=192.0)
        assert m.capacity_gb == 192.0

    def test_default_capacity_zero(self):
        m = Memory(name="dram")
        assert m.capacity_gb == 0.0


class TestFabric:
    def test_full_duplex_time(self):
        f = Fabric(
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
        f = Fabric(
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
        f = Fabric(
            name="link",
            src=Compute(name="a"),
            dst=Compute(name="b"),
            src_to_dst_bandwidth_gbs=100.0,
            dst_to_src_bandwidth_gbs=100.0,
            is_full_duplex=True,
        )
        assert f.transfer_time_us(0.0, 0.0) == 0.0


class TestCluster:
    def test_construction(self):
        g = Compute(name="gpu0")
        m = Memory(name="hbm0", capacity_gb=80.0)
        f = Fabric(
            name="bus", src=g, dst=m,
            src_to_dst_bandwidth_gbs=3000.0,
            dst_to_src_bandwidth_gbs=3000.0,
            is_full_duplex=True,
        )
        c = Cluster(computes=[g], memories=[m], fabrics=[f])
        assert len(c.computes) == 1
        assert len(c.fabrics) == 1


# ── Spec tests ───────────────────────────────────────────────────────


class TestHardwareSpec:
    def test_b300_preset(self):
        hw = hardware_spec("b300")
        assert hw is HW_B300
        assert hw.peak_tflops["bf16"] == 2250.0
        assert hw.peak_bw_gbs == 7750.0

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="unknown hardware preset"):
            hardware_spec("nonexistent")


class TestCollectiveBytes:
    def test_all_reduce(self):
        result = collective_bytes("all_reduce", 1024.0, 4)
        assert result == pytest.approx(2.0 * (3 / 4) * 1024.0)

    def test_all_gather(self):
        result = collective_bytes("all_gather", 1024.0, 8)
        assert result == pytest.approx((7 / 8) * 1024.0)

    def test_reduce_scatter(self):
        result = collective_bytes("reduce_scatter", 1024.0, 4)
        assert result == pytest.approx((3 / 4) * 1024.0)

    def test_all_to_all(self):
        result = collective_bytes("all_to_all", 1024.0, 4)
        assert result == pytest.approx((3 / 4) * 1024.0)

    def test_broadcast(self):
        result = collective_bytes("broadcast", 1024.0, 4)
        assert result == 1024.0

    def test_world_1_returns_zero(self):
        assert collective_bytes("all_reduce", 1024.0, 1) == 0.0

    def test_unknown_op_raises(self):
        with pytest.raises(ValueError, match="unknown collective op"):
            collective_bytes("scatter", 1024.0, 4)


class TestCollectiveTime:
    def test_world_1_returns_zero(self):
        hw = hardware_spec("b300")
        assert collective_time("all_reduce", 1024.0, 1, "intra", hw) == 0.0

    def test_intra_uses_nvlink_bw(self):
        hw = hardware_spec("b300")
        t = collective_time("all_reduce", 1e9, 8, "intra", hw)
        alpha = hw.intra_node.alpha_us * 1e-6
        wire = collective_bytes("all_reduce", 1e9, 8)
        expected = alpha + wire / (hw.intra_node.bw_gbs * 1e9)
        assert t == pytest.approx(expected)

    def test_inter_a2a_uses_p2p_bw(self):
        hw = hardware_spec("b300")
        t = collective_time("all_to_all", 1e9, 8, "inter", hw)
        alpha = hw.inter_node.alpha_us * 1e-6
        wire = collective_bytes("all_to_all", 1e9, 8)
        bw = min(hw.intra_node.bw_gbs, hw.inter_node.p2p_bw_gbs) * 1e9
        expected = alpha + wire / bw
        assert t == pytest.approx(expected)

    def test_inter_collective_uses_collective_bw(self):
        hw = hardware_spec("b300")
        t = collective_time("all_reduce", 1e9, 8, "inter", hw)
        alpha = hw.inter_node.alpha_us * 1e-6
        wire = collective_bytes("all_reduce", 1e9, 8)
        bw = min(hw.intra_node.bw_gbs, hw.inter_node.collective_bw_gbs) * 1e9
        expected = alpha + wire / bw
        assert t == pytest.approx(expected)
