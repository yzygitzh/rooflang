"""Closed-form forward FLOPs and HBM-byte formulas for the model-roofline
op enumerators.

Each function returns a dict with the same five keys, so callers can
sum / aggregate them uniformly:

    {
      "flops":             ...,    # floating-point operations
      "transferred_bytes": ...,    # canonical HBM read+write — bandwidth roofline input
      "input_bytes":       ...,    # incoming activations
      "weight_bytes":      ...,    # weights + their scale tensors (γ / β fold in here)
      "output_bytes":      ...,    # outgoing activations
    }

Default invariant (enforced by _kernel_result):
    transferred_bytes == input_bytes + weight_bytes + output_bytes

A primitive that needs to model traffic outside the 3-category split —
workspace, K/V re-load across q-blocks, atomic-RMW, hierarchical-cache
modelling — passes `transferred_bytes_override=<value>` into the helper;
that value becomes `transferred_bytes` in the dict and the invariant no
longer holds. Consumers always read `transferred_bytes` as authoritative.

The per-category fields exist for downstream memory-capacity analysis
(does it fit in HBM, peak activation memory, KV-cache headroom, …) which
the bandwidth-side transferred_bytes number alone cannot answer.

The "bytes" definition is per-kernel: GEMM treats every input as freshly
read from HBM (no reuse); attention assumes flash-style SMEM reuse of K/V
tiles (no S² in HBM). Each docstring states its own reuse model. Backward
and optimizer primitives live in separate modules.
"""
from typing import Dict, Optional


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


def _kernel_result(flops: float, input_bytes: float,
                   weight_bytes: float, output_bytes: float,
                   transferred_bytes_override: Optional[float] = None,
                   ) -> Dict[str, float]:
    """Wrap a primitive's per-category bytes into the standard result dict.
    Shared by primitives.forward / .backward / .optimizer.

    transferred_bytes defaults to input_bytes + weight_bytes + output_bytes.
    Pass transferred_bytes_override when a kernel needs to bypass that sum
    (workspace, K/V re-load across q-blocks, hierarchical-cache modelling).
    """
    transferred = (transferred_bytes_override
                   if transferred_bytes_override is not None
                   else input_bytes + weight_bytes + output_bytes)
    return {
        "flops":             flops,
        "transferred_bytes": transferred,
        "input_bytes":       input_bytes,
        "weight_bytes":      weight_bytes,
        "output_bytes":      output_bytes,
    }


def gemm_flops_bytes(M: int, N: int, K: int,
                     w_dtype: str, a_dtype: str, out_dtype: str = "bf16",
                     ) -> Dict[str, float]:
    """Dense GEMM C(M,N) = A(M,K) · B(K,N).

    flops:
        Each output element C[i,j] is a dot product of length K, costing
        K multiplies + (K-1) adds ≈ 2K flops (fused MAC counted as 2).
        Over M·N output elements, total = 2·M·N·K.

    bytes (HBM, no reuse):
        input_bytes  = M·K · sizeof(a_dtype)              (read A)
        weight_bytes = K·N · sizeof(w_dtype) + scale      (read B + per-block scales)
        output_bytes = M·N · sizeof(out_dtype)            (write C)
    """
    input_bytes  = M * K * dtype_bytes(a_dtype)
    weight_bytes = K * N * dtype_bytes(w_dtype) + gemm_scale_bytes(N, K, w_dtype)
    output_bytes = M * N * dtype_bytes(out_dtype)
    return _kernel_result(2.0 * M * N * K, input_bytes, weight_bytes, output_bytes)


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
                        ) -> Dict[str, float]:
    """RMSNorm over M rows × D dims:  y = x · rsqrt(mean(x²) + eps) · gamma.

    flops:
        Per row of D elements:
          - square each x[i]:            D mults
          - sum-reduction over D:        D adds (D-1 ≈ D)
          - mean / +eps / rsqrt:         O(1), dropped
          - multiply x[i] by inv_rms:    D mults
          - multiply by gamma[i]:        D mults
        Total per row = D + D + D + D = 4D, so 4·M·D over the batch.

    bytes (fused single-pass; eager-mode unfused would read x twice):
        input_bytes  = M·D · sizeof(dtype)      (read x)
        weight_bytes = D · sizeof(dtype)        (read gamma, broadcast once)
        output_bytes = M·D · sizeof(dtype)      (write y)
    """
    b = dtype_bytes(dtype)
    return _kernel_result(4.0 * M * D, M * D * b, D * b, M * D * b)


def layernorm_flops_bytes(M: int, D: int, dtype: str = "bf16",
                          ) -> Dict[str, float]:
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

    bytes (fused single-pass; same caveat as rmsnorm):
        input_bytes  = M·D · sizeof(dtype)      (read x)
        weight_bytes = 2·D · sizeof(dtype)      (read gamma + beta, broadcast)
        output_bytes = M·D · sizeof(dtype)      (write y)
    """
    b = dtype_bytes(dtype)
    return _kernel_result(7.0 * M * D, M * D * b, 2 * D * b, M * D * b)


def rope_flops_bytes(M: int, D: int, dtype: str = "bf16",
                     ) -> Dict[str, float]:
    """RoPE rotation over M tokens × D dims (Q or K). The D dims are
    paired into D/2 (cos, sin)-rotated 2-D groups:

        y[2i]   = x[2i]   · cos(θ_i) - x[2i+1] · sin(θ_i)
        y[2i+1] = x[2i]   · sin(θ_i) + x[2i+1] · cos(θ_i)

    flops:
        Per pair: 4 mults + 1 sub + 1 add = 6 flops.
        D/2 pairs per token → 3·D flops per token.
        Total = 3·M·D.

    bytes (cos/sin tables are D-sized and assumed cached):
        input_bytes  = M·D · sizeof(dtype)      (read x)
        weight_bytes = 0                        (cos/sin negligible)
        output_bytes = M·D · sizeof(dtype)      (write y)
    """
    b = dtype_bytes(dtype)
    return _kernel_result(3.0 * M * D, M * D * b, 0.0, M * D * b)


def attn_flops_bytes(B: int, H: int, H_kv: int,
                     S_q: int, S_kv: int, Hd: int,
                     dtype: str = "bf16", causal: bool = False,
                     ) -> Dict[str, float]:
    """Flash-style multi-head attention forward (no S² matrix in HBM).

    flops:
        QK^T per (b,h,q): K dot product of length Hd over S_kv keys
            → 2·S_kv·Hd flops, summed over S_q queries → 2·S_q·S_kv·Hd.
        softmax·V per (b,h,q): weighted sum of S_kv values of length Hd
            → 2·S_kv·Hd flops, ×S_q → 2·S_q·S_kv·Hd.
        Softmax itself is O(S_q·S_kv) per head, dropped as lower-order.
        Sum over B·H heads: 4·B·H·S_q·S_kv·Hd.
        Causal with S_q == S_kv halves the work (upper-triangular skipped).

    bytes (flash-tiled — K/V tiles reused in SMEM; no S×S in HBM):
        input_bytes  = (B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
                       (Q + K + V reads)
        weight_bytes = 0                        (no weights — qkv-proj is a separate gemm)
        output_bytes = B·H·S_q·Hd · sizeof(dtype)
                       (O write)
    """
    flops = 4.0 * B * H * S_q * S_kv * Hd
    if causal and S_q == S_kv:
        flops *= 0.5
    b = dtype_bytes(dtype)
    input_bytes  = (B * H * S_q * Hd + 2 * B * H_kv * S_kv * Hd) * b
    output_bytes = B * H * S_q * Hd * b
    return _kernel_result(flops, input_bytes, 0.0, output_bytes)


def sparse_attn_flops_bytes(B: int, H: int, H_kv: int,
                            S_q: int, k_sel: int, Hd: int,
                            dtype: str = "bf16",
                            ) -> Dict[str, float]:
    """Sparse attention: each query attends to k_sel selected K/V tokens
    (e.g. window + index_topk in the deepseek-v4-pro design). H_kv < H
    covers GQA / MQA / MLA — set H_kv = 1 for MQA or MLA-style shared KV.

    flops:
        Per (b, h, q): 2·k_sel·Hd for QK^T + 2·k_sel·Hd for softmax·V
        = 4·k_sel·Hd. Over B·H·S_q: 4·B·H·S_q·k_sel·Hd. (Q is up-projected
        per Q-head before the dot, so the H factor is on Q-heads.)

    bytes:
        input_bytes  = (B·H·S_q·Hd + 2·B·H_kv·S_q·k_sel·Hd) · sizeof(dtype)
                       (Q read + gathered K + V at k_sel positions)
        weight_bytes = 0
        output_bytes = B·H·S_q·Hd · sizeof(dtype)        (O write)
    """
    b = dtype_bytes(dtype)
    input_bytes  = (B * H * S_q * Hd + 2 * B * H_kv * S_q * k_sel * Hd) * b
    output_bytes = B * H * S_q * Hd * b
    return _kernel_result(4.0 * B * H * S_q * k_sel * Hd,
                          input_bytes, 0.0, output_bytes)
