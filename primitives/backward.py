"""Closed-form backward FLOPs and HBM-byte formulas, paired with forward.py.

Each function returns the same five-key dict as forward.py, built via the
shared `_kernel_result` helper:
  {flops, transferred_bytes, input_bytes, weight_bytes, output_bytes}.

Per-kernel bytes convention:
  - GEMM backward: every input freshly read from HBM (no reuse).
  - Attention backward: per-tensor read-once / write-once accounting,
    matching the same flash-style SMEM K/V tile reuse model the forward
    primitive uses, plus one extra QK^T recomputation pass to recover
    the softmax matrix from the saved log-sum-exp. This is a strict lower
    bound — FA-v2's two-pass structure (outer-K to compute dK/dV, outer-Q
    to compute dQ) typically incurs additional HBM traffic from K/V tile
    re-loads across q-blocks that this closed form does not model. The
    overhead depends on tile sizes (B_q, B_k) and is hardware-specific.

Parameter-gradient writes (dW, dγ, dβ) take a `grad_dtype` argument that
defaults to "fp32" — matching standard mixed-precision recipes (PyTorch
AMP, Megatron-LM, DeepSpeed). Recipes that accumulate in bf16 / fp8 / fp4
should pass that explicitly. Activation gradients (dX, dQ/dK/dV) stay at
the activation dtype, since they flow into the upstream op as inputs.
"""
from typing import Dict
from .forward import dtype_bytes, gemm_scale_bytes, _kernel_result


def gemm_dx_flops_bytes(M: int, N: int, K: int,
                        w_dtype: str, a_dtype: str, out_dtype: str = "bf16",
                        ) -> Dict[str, float]:
    """Gradient w.r.t. the input X of forward Y(M,N) = X(M,K)·W^T(K,N).

    dX(M,K) = dY(M,N) · W(N,K).

    flops:
        Each dX[i,k] is a dot product of length N (sum over output dims),
        costing 2N flops. Over M·K elements → 2·M·N·K.

    bytes (HBM, no reuse):
        input_bytes  = M·N · sizeof(out_dtype)              (read dY)
        weight_bytes = N·K · sizeof(w_dtype) + scale        (read W + per-block scales)
        output_bytes = M·K · sizeof(a_dtype)                (write dX, passed to upstream)
    """
    input_bytes  = M * N * dtype_bytes(out_dtype)
    weight_bytes = N * K * dtype_bytes(w_dtype) + gemm_scale_bytes(N, K, w_dtype)
    output_bytes = M * K * dtype_bytes(a_dtype)
    return _kernel_result(2.0 * M * N * K, input_bytes, weight_bytes, output_bytes)


def gemm_dw_flops_bytes(M: int, N: int, K: int,
                        w_dtype: str, a_dtype: str, out_dtype: str = "bf16",
                        grad_dtype: str = "fp32",
                        ) -> Dict[str, float]:
    """Gradient w.r.t. the weight W of forward Y(M,N) = X(M,K)·W^T(K,N).

    dW(N,K) = dY^T(N,M) · X(M,K).

    flops:
        Each dW[n,k] is a dot product of length M (sum over batch),
        costing 2M flops. Over N·K elements → 2·M·N·K.

    bytes (HBM, no reuse):
        input_bytes  = M·N · sizeof(out_dtype) + M·K · sizeof(a_dtype)
                       (read dY + read X — both saved-activation reads)
        weight_bytes = 0                            (no weight is read)
        output_bytes = N·K · sizeof(grad_dtype)     (write dW, fp32 by default)
    """
    flops = 2.0 * M * N * K
    input_bytes  = M * N * dtype_bytes(out_dtype) + M * K * dtype_bytes(a_dtype)
    output_bytes = N * K * dtype_bytes(grad_dtype)
    return _kernel_result(flops, input_bytes, 0.0, output_bytes)


def rmsnorm_backward_flops_bytes(M: int, D: int, dtype: str = "bf16",
                                 grad_dtype: str = "fp32",
                                 ) -> Dict[str, float]:
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
        input_bytes  = 2·M·D · sizeof(dtype)       (read x + read dy)
        weight_bytes = D · sizeof(dtype)           (read γ, broadcast)
        output_bytes = M·D · sizeof(dtype) + D · sizeof(grad_dtype)
                       (write dx + write dγ; dγ in grad_dtype, fp32 default)
    """
    db = dtype_bytes(dtype)
    gb = dtype_bytes(grad_dtype)
    return _kernel_result(9.0 * M * D,
                          2 * M * D * db, D * db, M * D * db + D * gb)


def layernorm_backward_flops_bytes(M: int, D: int, dtype: str = "bf16",
                                   grad_dtype: str = "fp32",
                                   ) -> Dict[str, float]:
    """LayerNorm backward — like rmsnorm backward, plus the mean-subtract
    gradient and dβ accumulation.

    flops (per row of D):
      Inner derivation (~9D, parallel to rmsnorm) plus:
      - dβ += dy:                                  D adds
      - extra mean-gradient term in dx:            D adds (re-center)
      Total ≈ 11D per row, so 11·M·D (~2.2× forward 5MD; conventional
      Megatron-LM accounting uses ~10D).

    bytes (fused single-pass):
        input_bytes  = 2·M·D · sizeof(dtype)       (read x + read dy)
        weight_bytes = D · sizeof(dtype)           (read γ; β not read in bwd)
        output_bytes = M·D · sizeof(dtype) + 2·D · sizeof(grad_dtype)
                       (write dx + dγ + dβ)
    """
    db = dtype_bytes(dtype)
    gb = dtype_bytes(grad_dtype)
    return _kernel_result(11.0 * M * D,
                          2 * M * D * db, D * db, M * D * db + 2 * D * gb)


def rope_backward_flops_bytes(M: int, D: int, dtype: str = "bf16",
                              ) -> Dict[str, float]:
    """RoPE backward = forward rotation with negated angle (transpose of
    the rotation matrix). Same per-pair cost as forward.

    flops:  3·M·D (D/2 pairs × 6 flops, identical breakdown to forward).
    bytes:
        input_bytes  = M·D · sizeof(dtype)         (read dy)
        weight_bytes = 0                           (cos/sin tables cached)
        output_bytes = M·D · sizeof(dtype)         (write dx)
    """
    b = dtype_bytes(dtype)
    return _kernel_result(3.0 * M * D, M * D * b, 0.0, M * D * b)


def attn_backward_flops_bytes(B: int, H: int, H_kv: int,
                              S_q: int, S_kv: int, Hd: int,
                              dtype: str = "bf16", causal: bool = False,
                              ) -> Dict[str, float]:
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

    bytes (per-tensor read-once / write-once; strict lower bound — see
           module docstring for why real FA-v2 traffic exceeds this):
        input_bytes  = (2·B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
                       (Q + K + V + dO reads)
        weight_bytes = 0
        output_bytes = (B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
                       (dQ + dK + dV writes, sized by H_q vs H_kv)
        transferred_bytes = input + output
                          = (3·B·H·S_q·Hd + 4·B·H_kv·S_kv·Hd) · sizeof(dtype).
    """
    flops = 10.0 * B * H * S_q * S_kv * Hd
    if causal and S_q == S_kv:
        flops *= 0.5
    b = dtype_bytes(dtype)
    input_bytes  = (2 * B * H * S_q * Hd + 2 * B * H_kv * S_kv * Hd) * b
    output_bytes = (B * H * S_q * Hd + 2 * B * H_kv * S_kv * Hd) * b
    return _kernel_result(flops, input_bytes, 0.0, output_bytes)


def sparse_attn_backward_flops_bytes(B: int, H: int, H_kv: int,
                                     S_q: int, k_sel: int, Hd: int,
                                     dtype: str = "bf16",
                                     ) -> Dict[str, float]:
    """Sparse-attention backward — same FA-v2 structure with k_sel keys
    in place of S_kv, plus gather/scatter on K, V, dK, dV at the selected
    indices (the gather cost itself is bandwidth, accounted in bytes).

    flops:  10·B·H·S_q·k_sel·Hd  (2.5× forward sparse-attn).
    bytes (per-tensor read-once / write-once; same caveat as attn bwd):
        input_bytes  = (2·B·H·S_q·Hd + 2·B·H_kv·S_q·k_sel·Hd) · sizeof(dtype)
                       (Q + dO + gathered K + gathered V)
        weight_bytes = 0
        output_bytes = (B·H·S_q·Hd + 2·B·H_kv·S_q·k_sel·Hd) · sizeof(dtype)
                       (dQ + dK + dV writes)
    """
    flops = 10.0 * B * H * S_q * k_sel * Hd
    b = dtype_bytes(dtype)
    input_bytes  = (2 * B * H * S_q * Hd + 2 * B * H_kv * S_q * k_sel * Hd) * b
    output_bytes = (B * H * S_q * Hd + 2 * B * H_kv * S_q * k_sel * Hd) * b
    return _kernel_result(flops, input_bytes, 0.0, output_bytes)
