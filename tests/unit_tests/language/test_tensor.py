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
    def test_identity_based_equality(self):
        t = Tensor("bf16", (4, 4))
        assert t == t

    def test_different_instances_not_equal(self):
        assert Tensor("bf16", (4, 4)) != Tensor("bf16", (4, 4))

    def test_hashable_by_identity(self):
        t1 = Tensor("bf16", (4, 4))
        t2 = Tensor("bf16", (4, 4))
        d = {t1: "a", t2: "b"}
        assert len(d) == 2
