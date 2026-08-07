"""Tests for DeepSeek V4 Pro placement strategies."""

import pytest

from rooflang.programs.dsv4_pro import optimization
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_b300_cluster_a_cp8_ep8_1node,
)


def test_cp8_ep8_1node_rejects_decode():
    with pytest.raises(ValueError, match="supports prefill only"):
        optimize_model_b300_cluster_a_cp8_ep8_1node(
            g=None, layers=[], hw=None, decode_steps=[object()])


def test_dp8_ep8_requires_equal_parallel_sizes(monkeypatch):
    monkeypatch.setattr(optimization, "DP", 4)
    monkeypatch.setattr(optimization, "EP", 8)
    with pytest.raises(ValueError, match="requires DP == EP"):
        optimization.optimize_model_b300_cluster_a_dp8_ep8_1node(
            g=None, layers=[], hw=None)


def test_cp8_ep8_requires_equal_parallel_sizes(monkeypatch):
    monkeypatch.setattr(optimization, "CP", 4)
    monkeypatch.setattr(optimization, "EP", 8)
    with pytest.raises(ValueError, match="requires CP == EP"):
        optimization.optimize_model_b300_cluster_a_cp8_ep8_1node(
            g=None, layers=[], hw=None)
