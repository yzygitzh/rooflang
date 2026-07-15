"""Unit tests for hardware preset classes."""

from rooflang.language.hardware.component import Compute
from rooflang.programs.presets.b300 import B300ClusterA


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
