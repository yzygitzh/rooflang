"""Shared utility functions for primitive Kernel constructors."""


def dtype_bytes(dtype: str) -> float:
    """Bytes-per-element by compute or storage dtype."""
    table = {"fp4": 0.5, "fp8": 1.0, "ue8m0": 1.0,
             "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}
    if dtype not in table:
        raise ValueError(f"unknown dtype: {dtype}")
    return table[dtype]


def gemm_scale_bytes(out_features: int, in_features: int,
                     w_dtype: str) -> float:
    """HBM bytes for per-block scale tensor accompanying a quantized weight.

    fp8: 1 ue8m0 scale per 128×128 weight block.
    fp4: 1 ue8m0 scale per 32 elements along K (per-row).
    Other dtypes: 0.
    """
    if w_dtype == "fp8":
        n_blocks = ((out_features + 127) // 128) * ((in_features + 127) // 128)
        return n_blocks * dtype_bytes("ue8m0")
    if w_dtype == "fp4":
        n_scales = out_features * ((in_features + 31) // 32)
        return n_scales * dtype_bytes("ue8m0")
    return 0.0
