"""Unit tests for hardware preset classes."""

from rooflang.language.hardware.component import Compute, Memory
from rooflang.programs.presets.b300 import B300ClusterA, B300SuperChipA


class TestB300AggregateOverride:
    def test_intra_node(self):
        hw = B300ClusterA(n_nodes=1)
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert hw.find_aggregate_bandwidth(gpus) == 900.0

    def test_inter_node(self):
        hw = B300ClusterA(n_nodes=2)
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert hw.find_aggregate_bandwidth(gpus) == 800.0

    def test_single_device_returns_inf(self):
        hw = B300ClusterA(n_nodes=1)
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert hw.find_aggregate_bandwidth(gpus[:1]) == float("inf")


class TestB300SuperChipA:
    def test_single_gpu(self):
        hw = B300SuperChipA()
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert len(gpus) == 1

    def test_hbm_capacity(self):
        hw = B300SuperChipA()
        hbm = [n for n in hw.nodes
                if isinstance(n, Memory) and "hbm3e" in n.name]
        assert len(hbm) == 1
        assert hbm[0].capacity_gb == 21504.0

    def test_topology_has_nodes(self):
        hw = B300SuperChipA()
        assert len(hw.nodes) > 0

    def test_find_local_memory(self):
        hw = B300SuperChipA()
        gpu = [n for n in hw.nodes
               if isinstance(n, Compute) and "nvidia-b300" in n.name][0]
        mem = hw.find_local_memory(gpu)
        assert isinstance(mem, Memory)
        assert "hbm3e" in mem.name
