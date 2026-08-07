"""Tests for DeepSeek V4 Pro placement strategies."""

import pytest

from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_b300_cluster_a_cp8_ep8_1node,
)


def test_cp8_ep8_1node_rejects_decode():
    with pytest.raises(ValueError, match="supports prefill only"):
        optimize_model_b300_cluster_a_cp8_ep8_1node(
            g=None, layers=[], hw=None, decode_steps=[object()])
