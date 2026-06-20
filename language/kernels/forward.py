"""Forward-pass Kernel subclasses for the model-roofline op enumerators.

Each class takes shape + dtype arguments, computes flops / byte-category
fields in __init__, and inherits composition (FusedKernel, OverlappedKernel)
from the base Kernel class in kernel.py.

The "bytes" definition is per-kernel: GEMM treats every input as freshly
read from HBM (no reuse); attention assumes flash-style SMEM reuse of K/V
tiles (no S² in HBM). Each docstring states its own reuse model.
"""

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.utils import dtype_bytes, gemm_scale_bytes


class Gemm(Kernel):
    """Dense GEMM C(M,N) = A(M,K) · B(K,N).

    flops = 2·M·N·K (fused MAC counted as 2 per output element).
    bytes (HBM, no reuse):
        input_bytes  = M·K · sizeof(a_dtype)
        weight_bytes = K·N · sizeof(w_dtype) + scale bytes
        output_bytes = M·N · sizeof(out_dtype)
    """

    def __init__(self, M: int, N: int, K: int,
                 w_dtype: str, a_dtype: str, out_dtype: str = "bf16"):
        self.M, self.N, self.K = M, N, K
        self.w_dtype, self.a_dtype, self.out_dtype = w_dtype, a_dtype, out_dtype
        ib = M * K * dtype_bytes(a_dtype)
        wb = K * N * dtype_bytes(w_dtype) + gemm_scale_bytes(N, K, w_dtype)
        ob = M * N * dtype_bytes(out_dtype)
        super().__init__(
            flops=2.0 * M * N * K,
            transferred_bytes=ib + wb + ob,
            input_bytes=ib, weight_bytes=wb, output_bytes=ob,
        )


class RMSNorm(Kernel):
    """RMSNorm:  y = x · rsqrt(mean(x²) + eps) · gamma.

    flops = 4·M·D (square + reduce + mul-by-rsqrt + mul-by-gamma).
    bytes (fused single-pass):
        input_bytes  = M·D · sizeof(dtype)
        weight_bytes = D · sizeof(dtype)   (gamma, broadcast)
        output_bytes = M·D · sizeof(dtype)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16"):
        self.M, self.D, self.dtype_ = M, D, dtype
        b = dtype_bytes(dtype)
        ib = M * D * b
        wb = D * b
        ob = M * D * b
        super().__init__(
            flops=4.0 * M * D,
            transferred_bytes=ib + wb + ob,
            input_bytes=ib, weight_bytes=wb, output_bytes=ob,
        )


class LayerNorm(Kernel):
    """LayerNorm:  y = (x-mean)·rsqrt(var+eps)·gamma + beta.

    flops = 7·M·D (mean + sub + sq + var + mul-invstd + mul-gamma + add-beta).
    bytes (fused single-pass):
        input_bytes  = M·D · sizeof(dtype)
        weight_bytes = 2·D · sizeof(dtype)  (gamma + beta)
        output_bytes = M·D · sizeof(dtype)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16"):
        self.M, self.D, self.dtype_ = M, D, dtype
        b = dtype_bytes(dtype)
        ib = M * D * b
        wb = 2 * D * b
        ob = M * D * b
        super().__init__(
            flops=7.0 * M * D,
            transferred_bytes=ib + wb + ob,
            input_bytes=ib, weight_bytes=wb, output_bytes=ob,
        )


class RoPE(Kernel):
    """RoPE rotation: y[2i] = x[2i]·cos - x[2i+1]·sin, etc.

    flops = 3·M·D  (D/2 pairs × 6 flops per pair).
    bytes (cos/sin cached):
        input_bytes  = M·D · sizeof(dtype)
        weight_bytes = 0
        output_bytes = M·D · sizeof(dtype)
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


class Attn(Kernel):
    """Flash-style multi-head attention forward (no S² in HBM).

    flops = 4·B·H·S_q·S_kv·Hd  (×0.5 if causal and S_q==S_kv).
    bytes (flash-tiled — K/V reused in SMEM):
        input_bytes  = (B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
        weight_bytes = 0
        output_bytes = B·H·S_q·Hd · sizeof(dtype)
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, S_kv: int, Hd: int,
                 dtype: str = "bf16", causal: bool = False):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.S_kv, self.Hd = S_q, S_kv, Hd
        self.dtype_, self.causal = dtype, causal
        flops = 4.0 * B * H * S_q * S_kv * Hd
        if causal and S_q == S_kv:
            flops *= 0.5
        b = dtype_bytes(dtype)
        ib = (B * H * S_q * Hd + 2 * B * H_kv * S_kv * Hd) * b
        ob = B * H * S_q * Hd * b
        super().__init__(
            flops=flops,
            transferred_bytes=ib + ob,
            input_bytes=ib, weight_bytes=0.0, output_bytes=ob,
        )


class SparseAttn(Kernel):
    """Sparse attention: each query attends to k_sel selected K/V tokens.

    flops = 4·B·H·S_q·k_sel·Hd.
    bytes:
        input_bytes  = (B·H·S_q·Hd + 2·B·H_kv·S_q·k_sel·Hd) · sizeof(dtype)
        weight_bytes = 0
        output_bytes = B·H·S_q·Hd · sizeof(dtype)
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, k_sel: int, Hd: int,
                 dtype: str = "bf16"):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.k_sel, self.Hd = S_q, k_sel, Hd
        self.dtype_ = dtype
        b = dtype_bytes(dtype)
        ib = (B * H * S_q * Hd + 2 * B * H_kv * S_q * k_sel * Hd) * b
        ob = B * H * S_q * Hd * b
        super().__init__(
            flops=4.0 * B * H * S_q * k_sel * Hd,
            transferred_bytes=ib + ob,
            input_bytes=ib, weight_bytes=0.0, output_bytes=ob,
        )
