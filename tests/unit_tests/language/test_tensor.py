"""Unit tests for rooflang.language.tensor (Tensor dataclass)."""

import pytest

from rooflang.language.tensor import Tensor


class TestTensorProperties:
    def test_n_elements_1d(self):
        t = Tensor("bf16", (1024,))
        assert t.n_elements == 1024

    def test_n_elements_2d(self):
        t = Tensor("bf16", (32, 64))
        assert t.n_elements == 32 * 64

    def test_n_elements_3d(self):
        t = Tensor("fp32", (2, 3, 4))
        assert t.n_elements == 24

    def test_n_elements_scalar_shape(self):
        t = Tensor("bf16", (1,))
        assert t.n_elements == 1

    def test_size_bytes_bf16(self):
        t = Tensor("bf16", (4, 4))
        assert t.size_bytes == 16 * 2.0

    def test_size_bytes_fp32(self):
        t = Tensor("fp32", (4, 4))
        assert t.size_bytes == 16 * 4.0

    def test_size_bytes_fp4(self):
        t = Tensor("fp4", (64,))
        assert t.size_bytes == 64 * 0.5


class TestTensorDefaults:
    def test_location_default_none(self):
        t = Tensor("bf16", (4,))
        assert t.location is None

    def test_location_set(self):
        t = Tensor("bf16", (4,), location="hbm")
        assert t.location == "hbm"


class TestTensorEquality:
    def test_equal(self):
        a = Tensor("bf16", (4, 4))
        b = Tensor("bf16", (4, 4))
        assert a == b

    def test_different_dtype(self):
        a = Tensor("bf16", (4, 4))
        b = Tensor("fp32", (4, 4))
        assert a != b

    def test_different_shape(self):
        a = Tensor("bf16", (4, 4))
        b = Tensor("bf16", (8, 8))
        assert a != b

    def test_different_location(self):
        a = Tensor("bf16", (4, 4), location=None)
        b = Tensor("bf16", (4, 4), location="hbm")
        assert a != b
