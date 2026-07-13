"""Unit tests for rooflang.language.tensor (Tensor dataclass)."""

from rooflang.language.tensor import Tensor


class TestNElements:
    def test_1d(self):
        assert Tensor("bf16", (1024,)).n_elements == 1024

    def test_2d(self):
        assert Tensor("bf16", (32, 64)).n_elements == 32 * 64

    def test_3d(self):
        assert Tensor("fp32", (2, 3, 4)).n_elements == 24

    def test_scalar(self):
        assert Tensor("bf16", (1,)).n_elements == 1


class TestSizeBytes:
    def test_bf16(self):
        assert Tensor("bf16", (4, 4)).size_bytes == 16 * 2.0

    def test_fp32(self):
        assert Tensor("fp32", (4, 4)).size_bytes == 16 * 4.0

    def test_fp4(self):
        assert Tensor("fp4", (64,)).size_bytes == 64 * 0.5


class TestTensorInit:
    def test_location_default_none(self):
        assert Tensor("bf16", (4,)).location is None

    def test_location_set(self):
        assert Tensor("bf16", (4,), location="hbm").location == "hbm"

    def test_equal(self):
        assert Tensor("bf16", (4, 4)) == Tensor("bf16", (4, 4))

    def test_different_dtype(self):
        assert Tensor("bf16", (4, 4)) != Tensor("fp32", (4, 4))

    def test_different_shape(self):
        assert Tensor("bf16", (4, 4)) != Tensor("bf16", (8, 8))

    def test_different_location(self):
        assert Tensor("bf16", (4,), location=None) != Tensor("bf16", (4,), location="hbm")
