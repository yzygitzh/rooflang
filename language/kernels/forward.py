"""Forward-pass Kernel subclasses for the model-roofline op enumerators.

Each class takes shape + dtype arguments and exposes roofline metrics
(flops, input_bytes, weight_bytes, output_bytes) as @property methods.

The "bytes" definition is per-kernel: GEMM treats every input as freshly
read from HBM (no reuse); attention assumes flash-style SMEM reuse of K/V
tiles (no S² in HBM). Each docstring states its own reuse model.
"""

from fractions import Fraction

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.utils import dtype_bytes, gemm_scale_bytes


class Nop(Kernel):
    """Zero-cost dependency kernel with arbitrary tensor ports."""

    _requires_placement = False

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return 0.0

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 0.0

    @property
    def transferred_bytes(self) -> float:
        return 0.0


class ReadInput(Kernel):
    """Host-to-device transfer: read token indices from CPU memory to GPU.

    flops = 0 (pure memcpy).
    bytes:
        input_bytes  = n_elements · sizeof(dtype)   (from CPU DRAM)
        output_bytes = n_elements · sizeof(dtype)   (to GPU HBM)
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


class Slice(Kernel):
    """Materialize a contiguous slice into a compact output allocation.

    The graph builder supplies the full source tensor as the input and the
    selected region as the output.  The generic Kernel byte accounting reads
    the input tensor and writes the compact output tensor.  Unlike a view,
    the output does not alias or pin the source allocation.
    """

    def __init__(self):
        super().__init__()


class Embedding(Kernel):
    """Token embedding lookup: gather M rows from a V×D table.

    flops = 0 (pure gather, no arithmetic).
    bytes:
        input_bytes  = M · sizeof(idx_dtype)        (token indices)
        weight_bytes = V · D · sizeof(w_dtype)      (full embedding table in HBM)
        output_bytes = M · D · sizeof(out_dtype)
    """

    def __init__(self, M: int, V: int, D: int,
                 w_dtype: str = "bf16", idx_dtype: str = "int32",
                 out_dtype: str = "bf16"):
        self.M, self.V, self.D = M, V, D
        self.w_dtype = w_dtype
        self.idx_dtype = idx_dtype
        self.out_dtype = out_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 0.0

    @property
    def input_bytes(self) -> float:
        return self.M * dtype_bytes(self.idx_dtype)

    @property
    def weight_bytes(self) -> float:
        return self.V * self.D * dtype_bytes(self.w_dtype)

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.out_dtype)


class ElementwiseOp(Kernel):
    """Element-wise binary op: y = op(a, b).

    Supported ops:
      "add": y = a + b
      "mul": y = a * b

    flops = M·D (one op per element).
    bytes:
        input_bytes  = 2·M·D · sizeof(dtype)  (two input tensors)
        weight_bytes = 0
        output_bytes = M·D · sizeof(dtype)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16",
                 op: str = "add"):
        self.M, self.D, self.dtype_ = M, D, dtype
        self.op = op
        super().__init__()

    @property
    def flops(self) -> float:
        return float(self.M * self.D)

    @property
    def input_bytes(self) -> float:
        return 2.0 * self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)


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
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.N * self.K

    @property
    def input_bytes(self) -> float:
        return self.M * self.K * dtype_bytes(self.a_dtype)

    @property
    def weight_bytes(self) -> float:
        return (self.K * self.N * dtype_bytes(self.w_dtype)
                + gemm_scale_bytes(self.N, self.K, self.w_dtype))

    @property
    def output_bytes(self) -> float:
        return self.M * self.N * dtype_bytes(self.out_dtype)


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
        super().__init__()

    @property
    def flops(self) -> float:
        return 4.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return self.D * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)


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
        super().__init__()

    @property
    def flops(self) -> float:
        return 7.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 2 * self.D * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.dtype_)


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
    """Flash-style multi-head attention forward (no S² in HBM).

    flops = 4·B·H·S_q·S_kv·Hd (×0.5 for a triangular causal matrix).
    bytes (flash-tiled — K/V reused in SMEM):
        input_bytes  = (B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
        weight_bytes = 0
        output_bytes = B·H·S_q·Hd · sizeof(dtype)
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, S_kv: int, Hd: int,
                 dtype: str = "bf16", causal: bool = False,
                 triangular: bool | None = None):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.S_kv, self.Hd = S_q, S_kv, Hd
        self.dtype_, self.causal = dtype, causal
        self.triangular = (causal and S_q == S_kv
                           if triangular is None else triangular)
        super().__init__()

    @property
    def flops(self) -> float:
        f = 4.0 * self.B * self.H * self.S_q * self.S_kv * self.Hd
        if self.triangular:
            f *= 0.5
        return f

    @property
    def input_bytes(self) -> float:
        b = dtype_bytes(self.dtype_)
        return (self.B * self.H * self.S_q * self.Hd
                + 2 * self.B * self.H_kv * self.S_kv * self.Hd) * b

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.B * self.H * self.S_q * self.Hd * dtype_bytes(self.dtype_)


class StridedGemm(Kernel):
    """Gemm-like kernel where I/O shapes differ from standard M*K / M*N.

    Used for grouped linears (input != M*K), fused gated projections
    (output != M*N), or strided writes (output rows != M).
    Flops still follow 2*M*N*K.

    bytes:
        input_bytes  = in_elems · sizeof(a_dtype)
        weight_bytes = K·N · sizeof(w_dtype) + scale
        output_bytes = out_elems · sizeof(out_dtype)
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
        return self._in_elems * dtype_bytes(self.a_dtype)

    @property
    def weight_bytes(self) -> float:
        return (self.K * self.N * dtype_bytes(self.w_dtype)
                + gemm_scale_bytes(self.N, self.K, self.w_dtype))

    @property
    def output_bytes(self) -> float:
        return self._out_elems * dtype_bytes(self.out_dtype)


class SparseAttn(Kernel):
    """Sparse attention with an optional fused sparse indexer.

    Attention FLOPS are 4·B·H·S_q·k_sel·Hd.  During causal prefill, only
    the context-dependent ``causal_k_sel`` portion receives the 0.5 triangular
    factor; fixed window/Top-K work remains unchanged.  When indexer_s_kv is
    nonzero, the dominant FP4 index scoring and reduction work is added to the
    same roofline unit and receives the causal factor as a whole.  The FP4
    index cache remains an explicit input tensor, but Top-K indices are
    internal transient data.

    The full main KV tensor remains resident, while ideal cross-query reuse
    reads at most min(S_kv, S_q·k_sel) positions.  input_read_fraction exposes
    that sparse access to the simulator on the individual ``kv`` port.

    kv_factor=2 for standard MHA (separate K/V caches);
    kv_factor=1 for MLA (shared latent read once from HBM).

    ``dtype`` selects the device TFLOPS used for compute time.  Q, main KV,
    and output storage may independently use ``q_dtype``, ``kv_dtype``, and
    ``out_dtype``; each defaults to ``dtype`` for backward compatibility.
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, k_sel: int, S_kv: int, Hd: int,
                 dtype: str = "bf16", kv_factor: int = 2,
                 indexer_s_kv: int = 0, indexer_h: int = 0,
                 indexer_hd: int = 0, indexer_dtype: str = "fp4",
                 *, q_dtype: str | None = None,
                 kv_dtype: str | None = None,
                 out_dtype: str | None = None,
                 causal: bool = False, causal_k_sel: int = 0):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.k_sel, self.S_kv, self.Hd = S_q, k_sel, S_kv, Hd
        self.dtype_ = dtype
        self.kv_factor = kv_factor
        self.q_dtype = q_dtype or dtype
        self.kv_dtype = kv_dtype or dtype
        self.out_dtype = out_dtype or dtype
        self.indexer_s_kv = indexer_s_kv
        self.indexer_h = indexer_h
        self.indexer_hd = indexer_hd
        self.indexer_dtype = indexer_dtype
        self.causal = causal
        self.causal_k_sel = causal_k_sel
        super().__init__()

    @property
    def flops(self) -> float:
        effective_k_sel = self.k_sel
        indexer_factor = 1.0
        if self.causal:
            effective_k_sel -= 0.5 * self.causal_k_sel
            indexer_factor = 0.5
        attention = (4.0 * self.B * self.H * self.S_q
                     * effective_k_sel * self.Hd)
        indexer_score = (2.0 * self.B * self.S_q * self.indexer_h
                         * self.indexer_s_kv * self.indexer_hd
                         * indexer_factor)
        indexer_reduce = (3.0 * self.B * self.S_q * self.indexer_h
                          * self.indexer_s_kv * indexer_factor)
        return attention + indexer_score + indexer_reduce

    def input_read_fraction(self, port: str) -> float:
        if port == "kv":
            return min(
                Fraction(1, 1),
                Fraction(self.S_q * self.k_sel, self.S_kv),
            )
        return 1.0

    @property
    def input_tensor_bytes(self) -> float:
        return (
            self.B * self.H * self.S_q * self.Hd
            * dtype_bytes(self.q_dtype)
            + self.kv_factor * self.B * self.H_kv
            * self.S_kv * self.Hd * dtype_bytes(self.kv_dtype)
            + self.B * self.indexer_s_kv * self.indexer_hd
            * dtype_bytes(self.indexer_dtype)
        )

    @property
    def input_bytes(self) -> float:
        main_kv_reads = min(self.S_kv, self.S_q * self.k_sel)
        return (
            self.B * self.H * self.S_q * self.Hd
            * dtype_bytes(self.q_dtype)
            + self.kv_factor * self.B * self.H_kv
            * main_kv_reads * self.Hd * dtype_bytes(self.kv_dtype)
            + self.B * self.indexer_s_kv * self.indexer_hd
            * dtype_bytes(self.indexer_dtype)
        )

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return (self.B * self.H * self.S_q * self.Hd
                * dtype_bytes(self.out_dtype))


class TokenDispatch(Kernel):
    """MoE token dispatch: softmax routing + topk selection + token scatter.

    flops = 5·M·N_experts (softmax: max + sub + exp + sum + div per row).
    bytes:
        input_bytes  = M·D·sizeof(a_dtype) + M·N_experts·4
        output_bytes = M·topk·D·sizeof(a_dtype)
    """

    def __init__(self, M: int, D: int, N_experts: int, topk: int,
                 a_dtype: str = "bf16"):
        self.M, self.D = M, D
        self.N_experts = N_experts
        self.topk = topk
        self.a_dtype = a_dtype
        self.M_e = Fraction(M * topk, N_experts)
        super().__init__()

    @property
    def flops(self) -> float:
        return 5.0 * self.M * self.N_experts

    @property
    def input_bytes(self) -> float:
        return (self.M * self.D * dtype_bytes(self.a_dtype)
                + self.M * self.N_experts * dtype_bytes("fp32"))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.topk * self.D * dtype_bytes(self.a_dtype)


class TokenCombine(Kernel):
    """MoE token combine: weighted sum of expert outputs.

    flops = 2·M·topk·D (multiply by routing weight + accumulate).
    bytes:
        input_bytes  = M·topk·D·sizeof(a_dtype)
        output_bytes = M·D·sizeof(a_dtype)
    """

    def __init__(self, M: int, D: int, N_experts: int, topk: int,
                 a_dtype: str = "bf16"):
        self.M, self.D = M, D
        self.N_experts = N_experts
        self.topk = topk
        self.a_dtype = a_dtype
        self.M_e = Fraction(M * topk, N_experts)
        super().__init__()

    @property
    def flops(self) -> float:
        return 2.0 * self.M * self.topk * self.D

    @property
    def input_bytes(self) -> float:
        return self.M * self.topk * self.D * dtype_bytes(self.a_dtype)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.D * dtype_bytes(self.a_dtype)


class Sampling(Kernel):
    """Argmax / top-p sampling: select token IDs from logits.

    flops = M · V (comparisons for argmax / top-p selection).
    bytes:
        input_bytes  = M · V · sizeof(dtype)       (read logits)
        output_bytes = M · sizeof(out_dtype)        (write token IDs)
    """

    def __init__(self, M: int, V: int,
                 dtype: str = "bf16", out_dtype: str = "int32"):
        self.M, self.V = M, V
        self.dtype_, self.out_dtype = dtype, out_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return float(self.M * self.V)

    @property
    def input_bytes(self) -> float:
        return self.M * self.V * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * dtype_bytes(self.out_dtype)
