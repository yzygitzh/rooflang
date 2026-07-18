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
from rooflang.language.utils import dtype_bytes, gemm_scale_bytes


class ReadInput(Kernel):
    """ReadInput backward: device-to-host transfer.

    Symmetric to forward ReadInput (host-to-device). Models writing
    data from GPU HBM back to CPU DRAM.

    flops = 0 (pure memcpy).
    bytes:
        input_bytes  = n_elements · sizeof(dtype)  (from GPU HBM)
        output_bytes = n_elements · sizeof(dtype)  (to CPU DRAM)
    """

    def __init__(self, n_elements: int, dtype: str = "int32"):
        self.n_elements = n_elements
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.n_elements * dtype_bytes(self.dtype_)


class Embedding(Kernel):
    """Embedding backward: scatter-add gradients to M rows of the table.

    flops = M·D (one add per gathered element).
    bytes:
        input_bytes  = M·D·sizeof(a_dtype) + M·sizeof(idx_dtype)
                       (upstream gradient + saved token indices)
        weight_bytes = 0
        output_bytes = M·D·sizeof(grad_dtype)  (scattered gradient rows)
    """

    def __init__(self, M: int, V: int, D: int,
                 a_dtype: str = "bf16", idx_dtype: str = "int32",
                 grad_dtype: str = "fp32"):
        self.M, self.V, self.D = M, V, D
        self.a_dtype = a_dtype
        self.idx_dtype = idx_dtype
        self.grad_dtype = grad_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return float(self.M * self.D)

    @property
    def input_bytes(self) -> float:
        return (self.M * self.D * dtype_bytes(self.a_dtype)
                + self.M * dtype_bytes(self.idx_dtype))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.grad_dtype)


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


class TokenDispatch(Kernel):
    """TokenDispatch backward: gather scattered gradients + softmax backward.

    flops = M·topk·D (gather-reduce for d_x) + 4·M·N_experts (softmax bwd).
    bytes:
        input_bytes  = M·topk·D·sizeof(a_dtype) + M·topk·4 + M·N_experts·4
                       (d_scattered + d_routing_weights + saved routing_probs)
        output_bytes = M·D·sizeof(a_dtype) + M·N_experts·4
                       (d_x + d_routing_logits)
    """

    def __init__(self, M: int, D: int, N_experts: int, topk: int,
                 a_dtype: str = "bf16"):
        self.M, self.D = M, D
        self.N_experts = N_experts
        self.topk = topk
        self.a_dtype = a_dtype
        self.M_e = M * topk // N_experts
        super().__init__()

    @property
    def flops(self) -> float:
        return self.M * self.topk * self.D + 4.0 * self.M * self.N_experts

    @property
    def input_bytes(self) -> float:
        return (self.M * self.topk * self.D * dtype_bytes(self.a_dtype)
                + self.M * self.topk * dtype_bytes("fp32")
                + self.M * self.N_experts * dtype_bytes("fp32"))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return (self.M * self.D * dtype_bytes(self.a_dtype)
                + self.M * self.N_experts * dtype_bytes("fp32"))


class TokenCombine(Kernel):
    """TokenCombine backward: scale d_y by routing weights + compute d_routing.

    flops = 3·M·topk·D (scale for d_expert_outs + dot for d_routing_weights).
    bytes:
        input_bytes  = M·D·sizeof(a_dtype) + M·topk·D·sizeof(a_dtype) + M·topk·4
                       (d_y + saved expert_outputs + saved routing_weights)
        output_bytes = M·topk·D·sizeof(a_dtype) + M·topk·4
                       (d_expert_outputs + d_routing_weights)
    """

    def __init__(self, M: int, D: int, N_experts: int, topk: int,
                 a_dtype: str = "bf16"):
        self.M, self.D = M, D
        self.N_experts = N_experts
        self.topk = topk
        self.a_dtype = a_dtype
        self.M_e = M * topk // N_experts
        super().__init__()

    @property
    def flops(self) -> float:
        return 3.0 * self.M * self.topk * self.D

    @property
    def input_bytes(self) -> float:
        return (self.M * self.D * dtype_bytes(self.a_dtype)
                + self.M * self.topk * self.D * dtype_bytes(self.a_dtype)
                + self.M * self.topk * dtype_bytes("fp32"))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return (self.M * self.topk * self.D * dtype_bytes(self.a_dtype)
                + self.M * self.topk * dtype_bytes("fp32"))


class StridedGemmDX(Kernel):
    """Backward dX for StridedGemm: dX = dY · W^T.

    flops = 2·M·N·K.
    bytes:
        input_bytes  = out_elems · sizeof(out_dtype)        (read dY)
        weight_bytes = K·N · sizeof(w_dtype) + scale        (read W)
        output_bytes = in_elems · sizeof(a_dtype)           (write dX)
    """

    def __init__(self, M: int, N: int, K: int,
                 w_dtype: str, a_dtype: str = "bf16", out_dtype: str = "bf16",
                 *, in_elems: int = 0, out_elems: int = 0):
        self.M, self.N, self.K = M, N, K
        self.w_dtype, self.a_dtype, self.out_dtype = w_dtype, a_dtype, out_dtype
        self._in_elems = in_elems if in_elems else M * K
        self._out_elems = out_elems if out_elems else M * N
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    @property
    def input_bytes(self) -> float:
        return self._out_elems * dtype_bytes(self.out_dtype)

    @property
    def weight_bytes(self) -> float:
        return (self.K * self.N * dtype_bytes(self.w_dtype)
                + gemm_scale_bytes(self.N, self.K, self.w_dtype))

    @property
    def output_bytes(self) -> float:
        return self._in_elems * dtype_bytes(self.a_dtype)


class StridedGemmDW(Kernel):
    """Backward dW for StridedGemm: dW = X^T · dY.

    flops = 2·M·N·K.
    bytes:
        input_bytes  = out_elems · sizeof(out_dtype) + in_elems · sizeof(a_dtype)
                       (read dY + X)
        weight_bytes = 0
        output_bytes = K·N · sizeof(grad_dtype)             (write dW)
    """

    def __init__(self, M: int, N: int, K: int,
                 w_dtype: str, a_dtype: str = "bf16", out_dtype: str = "bf16",
                 grad_dtype: str = "fp32",
                 *, in_elems: int = 0, out_elems: int = 0):
        self.M, self.N, self.K = M, N, K
        self.w_dtype, self.a_dtype, self.out_dtype = w_dtype, a_dtype, out_dtype
        self.grad_dtype = grad_dtype
        self._in_elems = in_elems if in_elems else M * K
        self._out_elems = out_elems if out_elems else M * N
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    @property
    def input_bytes(self) -> float:
        return (self._out_elems * dtype_bytes(self.out_dtype)
                + self._in_elems * dtype_bytes(self.a_dtype))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.K * self.N * dtype_bytes(self.grad_dtype)
