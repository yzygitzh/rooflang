"""Unit tests for rooflang.language.utils (dtype_bytes, gemm_scale_bytes)."""

import pytest

from rooflang.language.utils import dtype_bytes, gemm_scale_bytes


class TestDtypeBytes:
    @pytest.mark.parametrize("dtype,expected", [
        ("fp4", 0.5),
        ("fp8", 1.0),
        ("ue8m0", 1.0),
        ("bf16", 2.0),
        ("fp16", 2.0),
        ("fp32", 4.0),
    ])
    def test_known_dtypes(self, dtype, expected):
        assert dtype_bytes(dtype) == expected

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="unknown dtype"):
            dtype_bytes("int8")


class TestGemmScaleBytes:
    def test_fp8_single_block(self):
        result = gemm_scale_bytes(128, 128, "fp8")
        assert result == 1.0

    def test_fp8_multiple_blocks(self):
        result = gemm_scale_bytes(256, 256, "fp8")
        assert result == 4.0

    def test_fp8_non_aligned(self):
        result = gemm_scale_bytes(129, 129, "fp8")
        n_blocks = ((129 + 127) // 128) * ((129 + 127) // 128)
        assert result == n_blocks * 1.0

    def test_fp4_scales(self):
        result = gemm_scale_bytes(64, 64, "fp4")
        n_scales = 64 * ((64 + 31) // 32)
        assert result == n_scales * 1.0

    def test_fp4_non_aligned_k(self):
        result = gemm_scale_bytes(32, 33, "fp4")
        n_scales = 32 * ((33 + 31) // 32)
        assert result == n_scales * 1.0

    def test_bf16_returns_zero(self):
        assert gemm_scale_bytes(1024, 1024, "bf16") == 0.0

    def test_fp32_returns_zero(self):
        assert gemm_scale_bytes(512, 512, "fp32") == 0.0
