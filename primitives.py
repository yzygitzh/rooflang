"""Closed-form forward FLOPs and HBM-byte formulas for the model-roofline
op enumerators.

Each function returns (flops, bytes). The "bytes" definition is per-kernel:
GEMM treats every input as freshly read from HBM (no reuse); attention
assumes flash-style SMEM reuse of K/V tiles (no S² in HBM). Each docstring
states its own reuse model. Backward and optimizer primitives live in
separate modules.
"""
from typing import Tuple


def dtype_bytes(dtype: str) -> float:
    """Bytes-per-element by compute or storage dtype.

    fp4 is 0.5 (two values packed per byte). fp8 e4m3 / e5m2 are 1. ue8m0
    (block scale) is also 1. bf16 / fp16 are 2. fp32 is 4.
    """
    table = {"fp4": 0.5, "fp8": 1.0, "ue8m0": 1.0,
             "bf16": 2.0, "fp16": 2.0, "fp32": 4.0}
    if dtype not in table:
        raise ValueError(f"unknown dtype: {dtype}")
    return table[dtype]


def gemm_flops_bytes(M: int, N: int, K: int,
                     w_dtype: str, a_dtype: str, out_dtype: str = "bf16",
                     ) -> Tuple[float, float]:
    """Dense GEMM C(M,N) = A(M,K) · B(K,N).

    flops:
        Each output element C[i,j] is a dot product of length K, costing
        K multiplies + (K-1) adds ≈ 2K flops (fused MAC counted as 2).
        Over M·N output elements, total = 2·M·N·K.

    bytes (HBM, no reuse):
        Read A: M·K elements at sizeof(a_dtype) each.
        Read B: K·N elements at sizeof(w_dtype) each.
        Write C: M·N elements at sizeof(out_dtype) each.
        Total = M·K·a + K·N·w + M·N·o.
    """
    flops = 2.0 * M * N * K
    bytes_ = (M * K * dtype_bytes(a_dtype)
              + K * N * dtype_bytes(w_dtype)
              + M * N * dtype_bytes(out_dtype))
    return flops, bytes_


def gemm_scale_bytes(out_features: int, in_features: int,
                     w_dtype: str) -> float:
    """HBM bytes for the per-block scale tensor accompanying a quantized weight.

    Quantized GEMMs store the weight in low-precision (fp8 / fp4) and an
    accompanying scale tensor in ue8m0 (1 byte / element). The block layout
    sets how often a scale is required:

      fp8 (e4m3): 1 ue8m0 scale per 128×128 weight block.
          → scales has shape (ceil(out/128), ceil(in/128)) at 1 B each.
      fp4 (e2m1): 1 ue8m0 scale per 32 weight elements along K (per-row).
          → scales has shape (out, ceil(in/32)) at 1 B each.

    Other dtypes have no per-block scale, so 0 B.
    """
    if w_dtype == "fp8":
        return ((out_features + 127) // 128) * ((in_features + 127) // 128) * 1.0
    if w_dtype == "fp4":
        return out_features * ((in_features + 31) // 32) * 1.0
    return 0.0


def rmsnorm_flops_bytes(M: int, D: int, dtype: str = "bf16",
                        ) -> Tuple[float, float]:
    """RMSNorm over M rows × D dims:  y = x · rsqrt(mean(x²) + eps) · gamma.

    flops:
        Per row of D elements:
          - square each x[i]:            D mults
          - sum-reduction over D:        D adds (D-1 ≈ D)
          - mean / +eps / rsqrt:         O(1), dropped
          - multiply x[i] by inv_rms:    D mults
          - multiply by gamma[i]:        D mults
        Total per row = D + D + D + D = 4D, so 4·M·D over the batch.

    bytes:
        Fused single-pass kernel (Apex / Triton / aten::rms_norm style):
        x is loaded once into SMEM/registers and reused for both reduction
        and normalize. Unfused eager-mode would read x twice (~3·M·D bytes);
        not modeled here.
        Read x (M·D) + read gamma (D, broadcast) + write y (M·D)
        = (2·M·D + D) · sizeof(dtype).
    """
    flops = 4.0 * M * D
    bytes_ = (2 * M * D + D) * dtype_bytes(dtype)
    return flops, bytes_


def layernorm_flops_bytes(M: int, D: int, dtype: str = "bf16",
                          ) -> Tuple[float, float]:
    """LayerNorm over M rows × D dims:
       mean = mean(x); var = mean((x-mean)²);
       y = (x-mean)·rsqrt(var+eps)·gamma + beta.

    flops:
        Per row of D elements:
          - sum-reduction for mean:      D adds
          - subtract mean:               D subs
          - square (x-mean):             D mults
          - sum-reduction for var:       D adds
          - /D, +eps, rsqrt:             O(1), dropped
          - multiply by inv_std:         D mults
          - multiply by gamma:           D mults
          - add beta:                    D adds
        Total per row = D·7 = 7D, so 7·M·D over the batch.

    bytes:
        Fused single-pass assumption (same caveat as rmsnorm — eager-mode
        unfused would read x twice for the two reductions).
        Read x (M·D) + read gamma+beta (2·D, broadcast) + write y (M·D)
        = (2·M·D + 2·D) · sizeof(dtype).
    """
    flops = 7.0 * M * D
    bytes_ = (2 * M * D + 2 * D) * dtype_bytes(dtype)
    return flops, bytes_


def rope_flops_bytes(M: int, D: int, dtype: str = "bf16",
                     ) -> Tuple[float, float]:
    """RoPE rotation over M tokens × D dims (Q or K). The D dims are
    paired into D/2 (cos, sin)-rotated 2-D groups:

        y[2i]   = x[2i]   · cos(θ_i) - x[2i+1] · sin(θ_i)
        y[2i+1] = x[2i]   · sin(θ_i) + x[2i+1] · cos(θ_i)

    flops:
        Per pair: 4 mults + 1 sub + 1 add = 6 flops.
        D/2 pairs per token → 3·D flops per token.
        Total = 3·M·D.

    bytes:
        Read x and write y: 2·M·D · sizeof(dtype). The cos/sin tables are
        D-sized and assumed cached (negligible per-call HBM traffic).
    """
    flops = 3.0 * M * D
    bytes_ = 2 * M * D * dtype_bytes(dtype)
    return flops, bytes_


def attn_flops_bytes(B: int, H: int, H_kv: int,
                     S_q: int, S_kv: int, Hd: int,
                     dtype: str = "bf16", causal: bool = False,
                     ) -> Tuple[float, float]:
    """Flash-style multi-head attention forward (no S² matrix in HBM).

    flops:
        QK^T per (b,h,q): K dot product of length Hd over S_kv keys
            → 2·S_kv·Hd flops, summed over S_q queries → 2·S_q·S_kv·Hd.
        softmax·V per (b,h,q): weighted sum of S_kv values of length Hd
            → 2·S_kv·Hd flops, ×S_q → 2·S_q·S_kv·Hd.
        Softmax itself is O(S_q·S_kv) per head, dropped as lower-order.
        Sum over B·H heads: 4·B·H·S_q·S_kv·Hd.
        Causal with S_q == S_kv halves the work (upper-triangular skipped).

    bytes (flash-tiled — K/V tiles reused in SMEM across queries; no S×S
           attention matrix written back to HBM):
        Read Q + write O: 2·B·H·S_q·Hd · sizeof(dtype).
        Read K + V      : 2·B·H_kv·S_kv·Hd · sizeof(dtype). GQA-aware via H_kv.
        Total = (2·B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype).
    """
    flops = 4.0 * B * H * S_q * S_kv * Hd
    if causal and S_q == S_kv:
        flops *= 0.5
    bytes_ = (2 * B * H * S_q * Hd
              + 2 * B * H_kv * S_kv * Hd) * dtype_bytes(dtype)
    return flops, bytes_


def sparse_attn_flops_bytes(B: int, H: int, H_kv: int,
                            S_q: int, k_sel: int, Hd: int,
                            dtype: str = "bf16",
                            ) -> Tuple[float, float]:
    """Sparse attention: each query attends to k_sel selected K/V tokens
    (e.g. window + index_topk in the deepseek-v4-pro design). H_kv < H
    covers GQA / MQA / MLA — set H_kv = 1 for MQA or MLA-style shared KV.

    flops:
        Per (b, h, q): 2·k_sel·Hd for QK^T + 2·k_sel·Hd for softmax·V
        = 4·k_sel·Hd. Over B·H·S_q: 4·B·H·S_q·k_sel·Hd. (Q is up-projected
        per Q-head before the dot, so the H factor is on Q-heads.)

    bytes:
        Read Q + write O: 2·B·H·S_q·Hd · sizeof(dtype).
        Gathered K + V  : 2·B·H_kv·S_q·k_sel·Hd · sizeof(dtype) — sized by
            H_kv (one KV set per KV-head; reused across Q-heads in group).
        Total = (2·B·H·S_q·Hd + 2·B·H_kv·S_q·k_sel·Hd) · sizeof(dtype).
    """
    flops = 4.0 * B * H * S_q * k_sel * Hd
    bytes_ = (2 * B * H * S_q * Hd
              + 2 * B * H_kv * S_q * k_sel * Hd) * dtype_bytes(dtype)
    return flops, bytes_
