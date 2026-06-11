"""Closed-form backward FLOPs and HBM-byte formulas, paired with forward.py.

Each function returns (flops, bytes). The "bytes" definition is per-kernel
(same convention as forward): GEMM backward treats inputs as freshly read
from HBM (no reuse); attention backward assumes flash-style SMEM tile reuse
plus one extra QK^T recomputation pass to recover the softmax matrix from
the saved log-sum-exp.

Parameter-gradient writes (dW, dγ, dβ) take a `grad_dtype` argument that
defaults to "fp32" — matching standard mixed-precision recipes (PyTorch
AMP, Megatron-LM, DeepSpeed). Recipes that accumulate in bf16 / fp8 /
fp4 should pass that explicitly. Activation gradients (dX, dQ/dK/dV) stay
at the activation dtype, since they flow into the upstream op as inputs.
"""
from typing import Tuple
from .forward import dtype_bytes


def gemm_dx_flops_bytes(M: int, N: int, K: int,
                        w_dtype: str, a_dtype: str, out_dtype: str = "bf16",
                        ) -> Tuple[float, float]:
    """Gradient w.r.t. the input X of forward Y(M,N) = X(M,K)·W^T(K,N).

    dX(M,K) = dY(M,N) · W(N,K).

    flops:
        Each dX[i,k] is a dot product of length N (sum over output dims),
        costing 2N flops. Over M·K elements → 2·M·N·K.

    bytes (HBM, no reuse):
        Read dY: M·N · sizeof(out_dtype).
        Read W : N·K · sizeof(w_dtype).
        Write dX: M·K · sizeof(a_dtype) (passed back to upstream op).
        Total = M·N·o + N·K·w + M·K·a.
    """
    flops = 2.0 * M * N * K
    bytes_ = (M * N * dtype_bytes(out_dtype)
              + N * K * dtype_bytes(w_dtype)
              + M * K * dtype_bytes(a_dtype))
    return flops, bytes_


def gemm_dw_flops_bytes(M: int, N: int, K: int,
                        w_dtype: str, a_dtype: str, out_dtype: str = "bf16",
                        grad_dtype: str = "fp32",
                        ) -> Tuple[float, float]:
    """Gradient w.r.t. the weight W of forward Y(M,N) = X(M,K)·W^T(K,N).

    dW(N,K) = dY^T(N,M) · X(M,K).

    flops:
        Each dW[n,k] is a dot product of length M (sum over batch),
        costing 2M flops. Over N·K elements → 2·M·N·K.

    bytes (HBM, no reuse):
        Read dY: M·N · sizeof(out_dtype).
        Read X : M·K · sizeof(a_dtype).
        Write dW: N·K · sizeof(grad_dtype). Default fp32 matches standard
            mixed-precision recipes (PyTorch AMP, Megatron-LM, DeepSpeed);
            pass "bf16" / "fp8" if the recipe uses lower-precision grad
            accumulation.
        Total = M·N·o + M·K·a + N·K·g.
    """
    flops = 2.0 * M * N * K
    bytes_ = (M * N * dtype_bytes(out_dtype)
              + M * K * dtype_bytes(a_dtype)
              + N * K * dtype_bytes(grad_dtype))
    return flops, bytes_


def rmsnorm_backward_flops_bytes(M: int, D: int, dtype: str = "bf16",
                                 grad_dtype: str = "fp32",
                                 ) -> Tuple[float, float]:
    """RMSNorm backward. Forward:  y = γ · x · r,  r = rsqrt(mean(x²)+eps).

    flops (per row of D):
      - d_x_hat = γ · dy:                          D mults
      - dot(d_x_hat, x):                           D mults + D adds  (= 2D)
      - r·d_x_hat (first term of dx):              D mults
      - scalar = -r³·dot(...)/D:                   O(1), dropped
      - x·scalar (second term of dx):              D mults
      - sum the two terms into dx:                 D adds
      - dγ += dy · (x·r)  (accumulated over batch): D mults + D mults + D adds
                                                   (3D per row contribution)
      Total per row ≈ 9D, so 9·M·D over the batch (~2.25× forward 4MD).

    bytes (fused single-pass; same caveat as forward):
        Read x, dy, γ:  (2·M·D + D) · sizeof(dtype).
        Write dx:       M·D · sizeof(dtype).
        Write dγ:       D · sizeof(grad_dtype) (recipe-dependent; fp32 by
            default for atomic / reduce accumulation safety).
        Total = (3·M·D + D) · sizeof(dtype) + D · sizeof(grad_dtype).
    """
    flops = 9.0 * M * D
    bytes_ = (3 * M * D + D) * dtype_bytes(dtype) + D * dtype_bytes(grad_dtype)
    return flops, bytes_


def layernorm_backward_flops_bytes(M: int, D: int, dtype: str = "bf16",
                                   grad_dtype: str = "fp32",
                                   ) -> Tuple[float, float]:
    """LayerNorm backward — like rmsnorm backward, plus the mean-subtract
    gradient and dβ accumulation.

    flops (per row of D):
      Inner derivation (~9D, parallel to rmsnorm) plus:
      - dβ += dy:                                  D adds
      - extra mean-gradient term in dx:            D adds (re-center)
      Total ≈ 11D per row, so 11·M·D (~2.2× forward 5MD; conventional
      Megatron-LM accounting uses ~10D).

    bytes (fused single-pass):
        Read x, dy, γ: (2·M·D + D) · sizeof(dtype).
        Write dx:      M·D · sizeof(dtype).
        Write dγ, dβ:  2·D · sizeof(grad_dtype) (recipe-dependent; fp32 default).
        Total = (3·M·D + D) · sizeof(dtype) + 2·D · sizeof(grad_dtype).
    """
    flops = 11.0 * M * D
    bytes_ = (3 * M * D + D) * dtype_bytes(dtype) + 2 * D * dtype_bytes(grad_dtype)
    return flops, bytes_


def rope_backward_flops_bytes(M: int, D: int, dtype: str = "bf16",
                              ) -> Tuple[float, float]:
    """RoPE backward = forward rotation with negated angle (transpose of
    the rotation matrix). Same per-pair cost as forward.

    flops:  3·M·D (D/2 pairs × 6 flops, identical breakdown to forward).
    bytes:  2·M·D · sizeof(dtype) (read dy, write dx).
    """
    flops = 3.0 * M * D
    bytes_ = 2 * M * D * dtype_bytes(dtype)
    return flops, bytes_


def attn_backward_flops_bytes(B: int, H: int, H_kv: int,
                              S_q: int, S_kv: int, Hd: int,
                              dtype: str = "bf16", causal: bool = False,
                              ) -> Tuple[float, float]:
    """Flash-Attention v2 backward (FA only saves log-sum-exp, recomputes S).

    flops (four backward matmuls + one recompute pass = 5 total, each
           2·B·H·S_q·S_kv·Hd):
        recompute S = Q·K^T   (FA didn't save the full S×S matrix)
        dV = P^T·dO
        dP = dO·V^T
        dQ = dS·K
        dK = dS^T·Q
        Softmax-backward elementwise (P, dS): O(B·H·S_q·S_kv), dropped.
        Total = 10·B·H·S_q·S_kv·Hd = 2.5× forward (which was 4·B·H·S_q·S_kv·Hd).
        Causal halving applies when S_q == S_kv.

    bytes (flash-tiled, similar SMEM K/V tile reuse to forward):
        Read Q, K, V, dO at full precision + LSE (per-row fp32 scalar).
        Write dQ, dK, dV in fp32 (gradient accumulators).
        Per FA-v2 analysis this works out to ~2× forward bytes; modeled
        as exactly 2× here.
    """
    flops = 10.0 * B * H * S_q * S_kv * Hd
    if causal and S_q == S_kv:
        flops *= 0.5
    fwd_bytes = (2 * B * H * S_q * Hd
                 + 2 * B * H_kv * S_kv * Hd) * dtype_bytes(dtype)
    return flops, 2.0 * fwd_bytes


def sparse_attn_backward_flops_bytes(B: int, H: int, H_kv: int,
                                     S_q: int, k_sel: int, Hd: int,
                                     dtype: str = "bf16",
                                     ) -> Tuple[float, float]:
    """Sparse-attention backward — same FA-v2 structure with k_sel keys
    in place of S_kv, plus gather/scatter on K, V, dK, dV at the selected
    indices (the gather cost itself is bandwidth, accounted in bytes).

    flops:  10·B·H·S_q·k_sel·Hd  (2.5× forward sparse-attn).
    bytes:  ~2× forward sparse-attn bytes.
    """
    flops = 10.0 * B * H * S_q * k_sel * Hd
    fwd_bytes = (2 * B * H * S_q * Hd
                 + 2 * B * H_kv * S_q * k_sel * Hd) * dtype_bytes(dtype)
    return flops, 2.0 * fwd_bytes
