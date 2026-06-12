"""Backward-pass Kernel subclasses, paired with forward.py.

Per-kernel bytes convention:
  - GEMM backward: every input freshly read from HBM (no reuse).
  - Attention backward: per-tensor read-once / write-once accounting,
    matching the same flash-style SMEM K/V tile reuse model the forward
    primitive uses, plus one extra QK^T recomputation pass to recover
    the softmax matrix from the saved log-sum-exp. This is a strict lower
    bound — FA-v2's two-pass structure (outer-K to compute dK/dV, outer-Q
    to compute dQ) typically incurs additional HBM traffic from K/V tile
    re-loads across q-blocks that this closed form does not model.

Parameter-gradient writes (dW, dγ, dβ) take a `grad_dtype` argument that
defaults to "fp32" — matching standard mixed-precision recipes. Recipes
that accumulate in bf16 / fp8 / fp4 should pass that explicitly.
Activation gradients (dX, dQ/dK/dV) stay at the activation dtype.
"""

from .kernel import Kernel
from .forward import dtype_bytes, gemm_scale_bytes


class GemmDX(Kernel):
    """dX(M,K) = dY(M,N) · W(N,K).

    flops = 2·M·N·K.
    bytes (HBM, no reuse):
        input_bytes  = M·N · sizeof(out_dtype)         (read dY)
        weight_bytes = N·K · sizeof(w_dtype) + scale   (read W)
        output_bytes = M·K · sizeof(a_dtype)           (write dX)
    """

    def __init__(self, M: int, N: int, K: int,
                 w_dtype: str, a_dtype: str, out_dtype: str = "bf16"):
        self.M, self.N, self.K = M, N, K
        self.w_dtype, self.a_dtype, self.out_dtype = w_dtype, a_dtype, out_dtype
        ib = M * N * dtype_bytes(out_dtype)
        wb = N * K * dtype_bytes(w_dtype) + gemm_scale_bytes(N, K, w_dtype)
        ob = M * K * dtype_bytes(a_dtype)
        super().__init__(
            flops=2.0 * M * N * K,
            transferred_bytes=ib + wb + ob,
            input_bytes=ib, weight_bytes=wb, output_bytes=ob,
        )


class GemmDW(Kernel):
    """dW(N,K) = dY^T(N,M) · X(M,K).

    flops = 2·M·N·K.
    bytes (HBM, no reuse):
        input_bytes  = M·N · sizeof(out_dtype) + M·K · sizeof(a_dtype)
                       (read dY + read X)
        weight_bytes = 0
        output_bytes = N·K · sizeof(grad_dtype) (write dW, fp32 default)
    """

    def __init__(self, M: int, N: int, K: int,
                 w_dtype: str, a_dtype: str, out_dtype: str = "bf16",
                 grad_dtype: str = "fp32"):
        self.M, self.N, self.K = M, N, K
        self.w_dtype, self.a_dtype, self.out_dtype = w_dtype, a_dtype, out_dtype
        self.grad_dtype = grad_dtype
        ib = M * N * dtype_bytes(out_dtype) + M * K * dtype_bytes(a_dtype)
        ob = N * K * dtype_bytes(grad_dtype)
        super().__init__(
            flops=2.0 * M * N * K,
            transferred_bytes=ib + ob,
            input_bytes=ib, weight_bytes=0.0, output_bytes=ob,
        )


class RMSNormBackward(Kernel):
    """RMSNorm backward.

    flops = 9·M·D per batch (~2.25× forward 4MD).
    bytes (fused single-pass):
        input_bytes  = 2·M·D · sizeof(dtype)           (read x + dy)
        weight_bytes = D · sizeof(dtype)               (read γ)
        output_bytes = M·D · sizeof(dtype) + D · sizeof(grad_dtype)
                       (write dx + dγ)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16",
                 grad_dtype: str = "fp32"):
        self.M, self.D, self.dtype_, self.grad_dtype = M, D, dtype, grad_dtype
        db = dtype_bytes(dtype)
        gb = dtype_bytes(grad_dtype)
        ib = 2 * M * D * db
        wb = D * db
        ob = M * D * db + D * gb
        super().__init__(
            flops=9.0 * M * D,
            transferred_bytes=ib + wb + ob,
            input_bytes=ib, weight_bytes=wb, output_bytes=ob,
        )


class LayerNormBackward(Kernel):
    """LayerNorm backward.

    flops = 11·M·D per batch.
    bytes (fused single-pass):
        input_bytes  = 2·M·D · sizeof(dtype)           (read x + dy)
        weight_bytes = D · sizeof(dtype)               (read γ)
        output_bytes = M·D · sizeof(dtype) + 2·D · sizeof(grad_dtype)
                       (write dx + dγ + dβ)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16",
                 grad_dtype: str = "fp32"):
        self.M, self.D, self.dtype_, self.grad_dtype = M, D, dtype, grad_dtype
        db = dtype_bytes(dtype)
        gb = dtype_bytes(grad_dtype)
        ib = 2 * M * D * db
        wb = D * db
        ob = M * D * db + 2 * D * gb
        super().__init__(
            flops=11.0 * M * D,
            transferred_bytes=ib + wb + ob,
            input_bytes=ib, weight_bytes=wb, output_bytes=ob,
        )


class RoPEBackward(Kernel):
    """RoPE backward = forward rotation with negated angle.

    flops = 3·M·D.
    bytes: input_bytes = M·D (dy), output_bytes = M·D (dx).
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16"):
        self.M, self.D, self.dtype_ = M, D, dtype
        b = dtype_bytes(dtype)
        ib = M * D * b
        ob = M * D * b
        super().__init__(
            flops=3.0 * M * D,
            transferred_bytes=ib + ob,
            input_bytes=ib, weight_bytes=0.0, output_bytes=ob,
        )


class AttnBackward(Kernel):
    """Flash-Attention v2 backward (recompute S from saved LSE).

    flops = 10·B·H·S_q·S_kv·Hd (×0.5 if causal and S_q==S_kv).
    bytes (per-tensor read-once / write-once; strict lower bound):
        input_bytes  = (2·B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
        output_bytes = (B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, S_kv: int, Hd: int,
                 dtype: str = "bf16", causal: bool = False):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.S_kv, self.Hd = S_q, S_kv, Hd
        self.dtype_, self.causal = dtype, causal
        flops = 10.0 * B * H * S_q * S_kv * Hd
        if causal and S_q == S_kv:
            flops *= 0.5
        b = dtype_bytes(dtype)
        ib = (2 * B * H * S_q * Hd + 2 * B * H_kv * S_kv * Hd) * b
        ob = (B * H * S_q * Hd + 2 * B * H_kv * S_kv * Hd) * b
        super().__init__(
            flops=flops,
            transferred_bytes=ib + ob,
            input_bytes=ib, weight_bytes=0.0, output_bytes=ob,
        )


class SparseAttnBackward(Kernel):
    """Sparse-attention backward (FA-v2 structure with k_sel keys).

    flops = 10·B·H·S_q·k_sel·Hd.
    bytes (per-tensor read-once / write-once):
        input_bytes  = (2·B·H·S_q·Hd + 2·B·H_kv·S_q·k_sel·Hd) · sizeof(dtype)
        output_bytes = (B·H·S_q·Hd + 2·B·H_kv·S_q·k_sel·Hd) · sizeof(dtype)
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, k_sel: int, Hd: int,
                 dtype: str = "bf16"):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.k_sel, self.Hd = S_q, k_sel, Hd
        self.dtype_ = dtype
        b = dtype_bytes(dtype)
        ib = (2 * B * H * S_q * Hd + 2 * B * H_kv * S_q * k_sel * Hd) * b
        ob = (B * H * S_q * Hd + 2 * B * H_kv * S_q * k_sel * Hd) * b
        super().__init__(
            flops=10.0 * B * H * S_q * k_sel * Hd,
            transferred_bytes=ib + ob,
            input_bytes=ib, weight_bytes=0.0, output_bytes=ob,
        )
