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

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.utils import dtype_bytes, gemm_scale_bytes


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
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    @property
    def input_bytes(self) -> float:
        return self.M * self.N * dtype_bytes(self.out_dtype)

    @property
    def weight_bytes(self) -> float:
        return (self.N * self.K * dtype_bytes(self.w_dtype)
                + gemm_scale_bytes(self.N, self.K, self.w_dtype))

    @property
    def output_bytes(self) -> float:
        return self.M * self.K * dtype_bytes(self.a_dtype)


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
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    @property
    def input_bytes(self) -> float:
        return (self.M * self.N * dtype_bytes(self.out_dtype)
                + self.M * self.K * dtype_bytes(self.a_dtype))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.N * self.K * dtype_bytes(self.grad_dtype)


class RMSNorm(Kernel):
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
        super().__init__()

    @property
    def flops(self) -> float:
        return 9.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        return 2 * self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return self.D * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return (self.M * self.D * dtype_bytes(self.dtype_)
                + self.D * dtype_bytes(self.grad_dtype))


class LayerNorm(Kernel):
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
        super().__init__()

    @property
    def flops(self) -> float:
        return 11.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        return 2 * self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return self.D * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return (self.M * self.D * dtype_bytes(self.dtype_)
                + 2 * self.D * dtype_bytes(self.grad_dtype))


class RoPE(Kernel):
    """RoPE backward = forward rotation with negated angle.

    flops = 3·M·D.
    bytes: input_bytes = M·D (dy), output_bytes = M·D (dx).
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16"):
        self.M, self.D, self.dtype_ = M, D, dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 3.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)


class Attn(Kernel):
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
        super().__init__()

    @property
    def flops(self) -> float:
        f = 10.0 * self.B * self.H * self.S_q * self.S_kv * self.Hd
        if self.causal and self.S_q == self.S_kv:
            f *= 0.5
        return f

    @property
    def input_bytes(self) -> float:
        b = dtype_bytes(self.dtype_)
        return (2 * self.B * self.H * self.S_q * self.Hd
                + 2 * self.B * self.H_kv * self.S_kv * self.Hd) * b

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        b = dtype_bytes(self.dtype_)
        return (self.B * self.H * self.S_q * self.Hd
                + 2 * self.B * self.H_kv * self.S_kv * self.Hd) * b


class SparseAttn(Kernel):
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
        super().__init__()

    @property
    def flops(self) -> float:
        return 10.0 * self.B * self.H * self.S_q * self.k_sel * self.Hd

    @property
    def input_bytes(self) -> float:
        b = dtype_bytes(self.dtype_)
        return (2 * self.B * self.H * self.S_q * self.Hd
                + 2 * self.B * self.H_kv * self.S_q * self.k_sel * self.Hd) * b

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        b = dtype_bytes(self.dtype_)
        return (self.B * self.H * self.S_q * self.Hd
                + 2 * self.B * self.H_kv * self.S_q * self.k_sel * self.Hd) * b
