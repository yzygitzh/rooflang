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
      "sigmoid_mul": y = a * sigmoid(b)

    flops = M·D for add/mul, 5·M·D for a sigmoid followed by multiply.
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
        ops_per_element = 5 if self.op == "sigmoid_mul" else 1
        return float(ops_per_element * self.M * self.D)

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


class PartialRMSNorm(Kernel):
    """RMSNorm a prefix while forwarding the remaining features unchanged."""

    def __init__(self, M: int, input_dim: int, norm_dim: int,
                 dtype: str = "bf16"):
        if not 0 < norm_dim <= input_dim:
            raise ValueError(
                f"norm_dim must be in [1, input_dim], got {norm_dim}")
        self.M, self.input_dim = M, input_dim
        self.norm_dim, self.dtype_ = norm_dim, dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 4.0 * self.M * self.norm_dim

    @property
    def input_bytes(self) -> float:
        return self.M * self.input_dim * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return self.norm_dim * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return self.M * self.input_dim * dtype_bytes(self.dtype_)


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


class AttnRes(Kernel):
    """Kimi AttnRes weighted residual aggregation.

    ``R`` is the number of saved block residuals.  The current prefix sum is
    appended internally, so every token scores and combines ``R + 1`` hidden
    vectors.  Kimi's implementation casts the aggregation path to FP32.
    """

    def __init__(self, B: int, S: int, D: int, R: int,
                 dtype: str = "bf16", compute_dtype: str = "fp32"):
        self.B, self.S, self.D, self.R = B, S, D, R
        self.dtype_ = compute_dtype
        self.storage_dtype = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        candidates = self.R + 1
        tokens = self.B * self.S
        # RMS normalization + score dot + weighted reduction, plus softmax.
        return float(
            7 * tokens * candidates * self.D
            + 5 * tokens * candidates
            + self.D
        )

    @property
    def input_bytes(self) -> float:
        return (self.B * self.S * (self.R + 1) * self.D
                * dtype_bytes(self.storage_dtype))

    @property
    def weight_bytes(self) -> float:
        return 2 * self.D * dtype_bytes(self.storage_dtype)

    @property
    def output_bytes(self) -> float:
        return (self.B * self.S * self.D
                * dtype_bytes(self.storage_dtype))


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


class KimiK3MlaAttn(Kernel):
    """Kimi-K3 dense MLA with an absorbed latent KV core.

    The logical query/output tensors retain their per-head dimensions while
    the persistent KV input stores one shared latent plus the RoPE component.
    The KV-B projection is absorbed into the attention kernel and remains an
    explicit weight for memory-footprint accounting.
    """

    def __init__(
        self,
        B: int,
        H: int,
        S_q: int,
        S_kv: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_cache_dim: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        dtype: str = "bf16",
        kv_transform_dtype: str = "bf16",
        *,
        q_dtype: str | None = None,
        kv_dtype: str = "fp8",
        out_dtype: str | None = None,
        causal: bool = False,
        selected_pairs: int | Fraction | None = None,
        kv_transform_tokens: int | Fraction | None = None,
    ):
        self.B, self.H = B, H
        self.S_q, self.S_kv = S_q, S_kv
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_cache_dim = kv_cache_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.dtype_ = dtype
        self.kv_transform_dtype = kv_transform_dtype
        self.q_dtype = q_dtype or dtype
        self.kv_dtype = kv_dtype
        self.out_dtype = out_dtype or dtype
        self.causal = causal
        self.selected_pairs = (
            self._default_selected_pairs() if selected_pairs is None
            else Fraction(selected_pairs))
        self.kv_transform_tokens = (
            Fraction(self.S_q) if kv_transform_tokens is None
            else Fraction(kv_transform_tokens))
        super().__init__()

    def _default_selected_pairs(self) -> Fraction:
        if self.causal and self.S_q == self.S_kv:
            return Fraction(self.S_q * (self.S_q + 1), 2)
        return Fraction(self.S_q * self.S_kv)

    @property
    def attention_flops(self) -> float:
        return float(
            2 * self.B * self.H * self.selected_pairs
            * (self.kv_cache_dim + self.kv_lora_rank))

    @property
    def kv_transform_flops(self) -> float:
        kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
        return float(
            2 * self.B * self.kv_transform_tokens
            * self.kv_lora_rank * kv_out)

    @property
    def flops(self) -> float:
        return self.attention_flops + self.kv_transform_flops

    @property
    def flops_by_dtype(self) -> dict[str, float]:
        result = {self.dtype_: self.attention_flops}
        result[self.kv_transform_dtype] = (
            result.get(self.kv_transform_dtype, 0.0)
            + self.kv_transform_flops)
        return {dtype: flops for dtype, flops in result.items() if flops > 0}

    @property
    def input_bytes(self) -> float:
        return (
            self.B * self.S_q * self.H * self.qk_head_dim
            * dtype_bytes(self.q_dtype)
            + self.B * self.S_kv * self.kv_cache_dim
            * dtype_bytes(self.kv_dtype)
        )

    @property
    def weight_bytes(self) -> float:
        kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
        return (
            self.kv_lora_rank * kv_out
            * dtype_bytes(self.kv_transform_dtype)
            + gemm_scale_bytes(
                kv_out, self.kv_lora_rank, self.kv_transform_dtype)
        )

    @property
    def output_bytes(self) -> float:
        return (self.B * self.S_q * self.H * self.v_head_dim
                * dtype_bytes(self.out_dtype))


class KimiK3DeltaAttn(Kernel):
    """Kimi Delta Attention forward kernel.

    Prefill uses the published chunkwise FLOP formula
    ``6*T*d^2 + 3*T*C*d + T*C^2`` per head.  Single-token decode uses the
    recurrent state update directly.  Q/K/V short convolutions, Q/K L2
    normalization, and the fused gate activations are included here.
    """

    def __init__(
        self,
        B: int,
        H: int,
        S: int,
        K: int,
        V: int,
        mode: str,
        chunk_size: int = 64,
        conv_size: int = 4,
        dtype: str = "bf16",
        state_dtype: str = "bf16",
    ):
        if mode not in {"chunk", "recurrent"}:
            raise ValueError(f"unsupported KDA mode: {mode}")
        self.B, self.H, self.S = B, H, S
        self.K, self.V = K, V
        self.mode = mode
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.dtype_ = dtype
        self.state_dtype = state_dtype
        self.cp_rank = None
        super().__init__()

    def input_read_fraction(self, port: str) -> float:
        # The first sequence rank participates in the ring halo exchange but
        # intentionally ignores the tail received from the last rank.
        if port == "conv_halo" and self.cp_rank == 0:
            return 0.0
        return 1.0

    @property
    def attention_flops(self) -> float:
        if self.mode == "chunk":
            per_head = (
                6 * self.S * self.K * self.V
                + 3 * self.S * self.chunk_size * self.K
                + self.S * self.chunk_size ** 2
            )
        else:
            per_head = (
                7 * self.S * self.K * self.V
                + 2 * self.S * self.V
            )
        return float(self.B * self.H * per_head)

    @property
    def preprocessing_flops(self) -> float:
        elements = self.B * self.H * self.S
        conv = 3 * elements * self.K * (2 * self.conv_size + 4)
        qk_norm = 2 * elements * (3 * self.K + 1)
        gate = elements * (5 * self.K + 3)
        return float(conv + qk_norm + gate)

    @property
    def flops(self) -> float:
        return self.attention_flops + self.preprocessing_flops


class KimiK3DeltaAttnCpSummary(Kernel):
    """Build one rank-local ``(M, S_ext)`` KDA transition summary.

    This is the additional CP pre-process beyond the ordinary local chunk
    kernel.  The dense M-chain is accumulated in FP32; other tensor-matrix
    work follows the BF16 KDA compute path.
    """

    def __init__(
        self,
        B: int,
        H: int,
        S: int,
        K: int,
        V: int,
        rank: int = 0,
        world: int = 1,
        chunk_size: int = 64,
        conv_size: int = 4,
        dtype: str = "bf16",
        state_dtype: str = "bf16",
    ):
        self.B, self.H, self.S = B, H, S
        self.K, self.V = K, V
        self.rank, self.world = rank, world
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.dtype_ = dtype
        self.state_dtype = state_dtype
        super().__init__()

    @property
    def n_chunks(self) -> int:
        return (self.S + self.chunk_size - 1) // self.chunk_size

    @property
    def bf16_flops(self) -> float:
        if self.rank == self.world - 1:
            return 0.0
        per_head = (
            4 * self.S * self.K * self.V
            + 2 * self.n_chunks * self.K * self.V
            + self.S * self.V
            + 2 * self.S * self.K * self.K
            + self.n_chunks * self.K * self.K
        )
        return float(self.B * self.H * per_head)

    @property
    def fp32_flops(self) -> float:
        if self.rank == self.world - 1:
            return 0.0
        return float(
            2 * self.B * self.H * self.n_chunks * self.K ** 3)

    @property
    def flops(self) -> float:
        return self.bf16_flops + self.fp32_flops

    @property
    def flops_by_dtype(self) -> dict[str, float]:
        return {self.dtype_: self.bf16_flops, "fp32": self.fp32_flops}

    @property
    def input_bytes(self) -> float:
        # The summary is fused with the local WY representation.
        return 0.0


class KimiK3DeltaAttnCpMerge(Kernel):
    """Merge preceding rank summaries into one rank's KDA initial state."""

    def __init__(
        self,
        B: int,
        H: int,
        K: int,
        V: int,
        rank: int,
        world: int,
        state_dtype: str = "bf16",
        summary_dtype: str = "fp32",
    ):
        self.B, self.H = B, H
        self.K, self.V = K, V
        self.rank, self.world = rank, world
        self.state_dtype = state_dtype
        self.summary_dtype = summary_dtype
        self.dtype_ = "fp32"
        super().__init__()

    @property
    def flops(self) -> float:
        per_summary = 2 * self.K * self.K * self.V + self.K * self.V
        return float(self.B * self.H * self.rank * per_summary)

    def input_read_fraction(self, port: str) -> float:
        if port != "summaries":
            return 1.0
        return Fraction(self.rank, self.world)

    @property
    def input_bytes(self) -> float:
        summary = self.B * self.H * self.rank * self.K * (self.K + self.V)
        return summary * dtype_bytes(self.summary_dtype)


class KimiK3DeltaAttnStateStore(Kernel):
    """Materialize persistent recurrent and short-convolution KDA state."""

    def __init__(
        self,
        B: int,
        H: int,
        S: int,
        K: int,
        V: int,
        conv_size: int = 4,
        state_dtype: str = "bf16",
    ):
        self.B, self.H, self.S = B, H, S
        self.K, self.V = K, V
        self.conv_size = conv_size
        self.state_dtype = state_dtype
        super().__init__()

    def input_read_fraction(self, port: str) -> float:
        # Input is only a graph dependency on the fused KDA result.  The
        # recurrent/conv state is materialized from registers, so it does not
        # introduce another HBM read of that result.
        return 0.0

    @property
    def input_bytes(self) -> float:
        # Input is a dependency on the KDA result, not an additional read.
        return 0.0


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


class DpskV4SparseAttn(Kernel):
    """DeepSeek V4 sparse attention with an optional fused sparse indexer.

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
    Fused indexer FLOPs use ``indexer_compute_dtype``, which defaults to the
    index-cache storage ``indexer_dtype``.
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, k_sel: int, S_kv: int, Hd: int,
                 dtype: str = "bf16", kv_factor: int = 2,
                 indexer_s_kv: int = 0, indexer_h: int = 0,
                 indexer_hd: int = 0, indexer_dtype: str = "fp4",
                 indexer_compute_dtype: str | None = None,
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
        self.indexer_compute_dtype = indexer_compute_dtype or indexer_dtype
        self.causal = causal
        self.causal_k_sel = causal_k_sel
        super().__init__()

    @property
    def attention_flops(self) -> float:
        effective_k_sel = self.k_sel
        if self.causal:
            effective_k_sel -= 0.5 * self.causal_k_sel
        return (4.0 * self.B * self.H * self.S_q
                * effective_k_sel * self.Hd)

    @property
    def indexer_flops(self) -> float:
        factor = 0.5 if self.causal else 1.0
        indexer_score = (2.0 * self.B * self.S_q * self.indexer_h
                         * self.indexer_s_kv * self.indexer_hd)
        indexer_reduce = (3.0 * self.B * self.S_q * self.indexer_h
                          * self.indexer_s_kv)
        return (indexer_score + indexer_reduce) * factor

    @property
    def flops(self) -> float:
        return self.attention_flops + self.indexer_flops

    @property
    def flops_by_dtype(self) -> dict[str, float]:
        result = {self.dtype_: self.attention_flops}
        result[self.indexer_compute_dtype] = (
            result.get(self.indexer_compute_dtype, 0.0)
            + self.indexer_flops)
        return {dtype: flops for dtype, flops in result.items() if flops > 0}

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


class Glm52SparseAttn(Kernel):
    """GLM-5.2 DSA with full/shared indexers and compressed FP8 KV.

    Main attention uses an absorbed MLA core whose QK/V dimensions are
    ``kv_cache_dim``/``kv_lora_rank``; logical Q/output tensors still use
    ``qk_head_dim``/``v_head_dim``.  Full-indexer layers fuse the history
    score, ReLU, head-weight reduction, causal masking, and logical Top-K
    selection.  The index projections and index-cache construction remain
    separate kernels.
    """

    def __init__(
        self,
        B: int,
        H: int,
        S_q: int,
        k_sel: int,
        S_kv: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_cache_dim: int,
        dtype: str = "bf16",
        kv_lora_rank: int = 0,
        qk_nope_head_dim: int = 0,
        kv_transform_dtype: str = "fp8",
        indexer_mode: str = "shared",
        indexer_s_kv: int = 0,
        indexer_h: int = 0,
        indexer_hd: int = 0,
        indexer_dtype: str = "fp8",
        indexer_compute_dtype: str = "fp8",
        indexer_reduce_dtype: str = "fp32",
        *,
        q_dtype: str | None = None,
        kv_dtype: str = "fp8",
        out_dtype: str | None = None,
        index_q_dtype: str = "bf16",
        index_weight_dtype: str = "fp32",
        causal: bool = False,
        selected_pairs: int | Fraction | None = None,
        indexer_pairs: int | Fraction | None = None,
        kv_transform_tokens: int | Fraction | None = None,
    ):
        if indexer_mode not in {"full", "shared"}:
            raise ValueError(
                f"indexer_mode must be 'full' or 'shared', got {indexer_mode}")
        if indexer_mode == "full" and min(
                indexer_s_kv, indexer_h, indexer_hd) <= 0:
            raise ValueError("full indexer dimensions must be positive")
        if indexer_mode == "shared" and any(
                (indexer_s_kv, indexer_h, indexer_hd)):
            raise ValueError("shared indexer must not own indexer dimensions")

        self.B, self.H = B, H
        self.S_q, self.k_sel, self.S_kv = S_q, k_sel, S_kv
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_cache_dim = kv_cache_dim
        self.dtype_ = dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.kv_transform_dtype = kv_transform_dtype
        self.indexer_mode = indexer_mode
        self.indexer_s_kv = indexer_s_kv
        self.indexer_h = indexer_h
        self.indexer_hd = indexer_hd
        self.indexer_dtype = indexer_dtype
        self.indexer_compute_dtype = indexer_compute_dtype
        self.indexer_reduce_dtype = indexer_reduce_dtype
        self.q_dtype = q_dtype or dtype
        self.kv_dtype = kv_dtype
        self.out_dtype = out_dtype or dtype
        self.index_q_dtype = index_q_dtype
        self.index_weight_dtype = index_weight_dtype
        self.causal = causal
        self.selected_pairs = (
            self._default_selected_pairs() if selected_pairs is None
            else Fraction(selected_pairs))
        self.indexer_pairs = (
            self._default_indexer_pairs() if indexer_pairs is None
            else Fraction(indexer_pairs))
        self.kv_transform_tokens = (
            Fraction(self.S_q) if kv_transform_tokens is None
            else Fraction(kv_transform_tokens))
        super().__init__()

    def _default_selected_pairs(self) -> Fraction:
        selected = min(self.S_kv, self.k_sel)
        if not self.causal:
            return Fraction(self.S_q * selected)
        if self.S_q == self.S_kv:
            ramp = min(self.S_q, selected)
            return Fraction(
                ramp * (ramp + 1) // 2 + (self.S_q - ramp) * ramp)
        return Fraction(self.S_q * selected, 2)

    def _default_indexer_pairs(self) -> Fraction:
        if self.indexer_mode == "shared":
            return Fraction(0)
        if not self.causal:
            return Fraction(self.S_q * self.indexer_s_kv)
        if self.S_q == self.indexer_s_kv:
            return Fraction(self.S_q * (self.S_q + 1), 2)
        return Fraction(self.S_q * self.indexer_s_kv, 2)

    @property
    def attention_flops(self) -> float:
        return float(
            2 * self.B * self.H * self.selected_pairs
            * (self.kv_cache_dim + self.kv_lora_rank))

    @property
    def kv_transform_flops(self) -> float:
        if self.kv_lora_rank == 0:
            return 0.0
        kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
        return float(
            2 * self.B * self.kv_transform_tokens
            * self.kv_lora_rank * kv_out)

    @property
    def indexer_score_flops(self) -> float:
        return float(
            2 * self.B * self.indexer_h * self.indexer_pairs
            * self.indexer_hd)

    @property
    def indexer_reduce_flops(self) -> float:
        return float(3 * self.B * self.indexer_h * self.indexer_pairs)

    @property
    def indexer_flops(self) -> float:
        return self.indexer_score_flops + self.indexer_reduce_flops

    @property
    def flops(self) -> float:
        return (self.attention_flops + self.kv_transform_flops
                + self.indexer_flops)

    @property
    def flops_by_dtype(self) -> dict[str, float]:
        result = {self.dtype_: self.attention_flops}
        result[self.kv_transform_dtype] = (
            result.get(self.kv_transform_dtype, 0.0)
            + self.kv_transform_flops)
        result[self.indexer_compute_dtype] = (
            result.get(self.indexer_compute_dtype, 0.0)
            + self.indexer_score_flops)
        result[self.indexer_reduce_dtype] = (
            result.get(self.indexer_reduce_dtype, 0.0)
            + self.indexer_reduce_flops)
        return {dtype: flops for dtype, flops in result.items() if flops > 0}

    def input_read_fraction(self, port: str) -> float:
        if port == "kv":
            return min(
                Fraction(1),
                Fraction(self.S_q * min(self.k_sel, self.S_kv), self.S_kv),
            )
        return 1.0

    @property
    def input_tensor_bytes(self) -> float:
        main = (
            self.B * self.S_q * self.H * self.qk_head_dim
            * dtype_bytes(self.q_dtype)
            + self.B * self.S_kv * self.kv_cache_dim
            * dtype_bytes(self.kv_dtype)
        )
        if self.indexer_mode == "shared":
            return main
        indexer = (
            self.B * self.S_q * self.indexer_h * self.indexer_hd
            * dtype_bytes(self.index_q_dtype)
            + self.B * self.indexer_s_kv * self.indexer_hd
            * dtype_bytes(self.indexer_dtype)
            + self.B * self.S_q * self.indexer_h
            * dtype_bytes(self.index_weight_dtype)
        )
        return main + indexer

    @property
    def input_bytes(self) -> float:
        main_kv_reads = min(self.S_kv, self.S_q * self.k_sel)
        main = (
            self.B * self.S_q * self.H * self.qk_head_dim
            * dtype_bytes(self.q_dtype)
            + self.B * main_kv_reads * self.kv_cache_dim
            * dtype_bytes(self.kv_dtype)
        )
        if self.indexer_mode == "shared":
            return main
        return main + (
            self.B * self.S_q * self.indexer_h * self.indexer_hd
            * dtype_bytes(self.index_q_dtype)
            + self.B * self.indexer_s_kv * self.indexer_hd
            * dtype_bytes(self.indexer_dtype)
            + self.B * self.S_q * self.indexer_h
            * dtype_bytes(self.index_weight_dtype)
        )

    @property
    def weight_bytes(self) -> float:
        if self.kv_lora_rank == 0:
            return 0.0
        kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
        return (
            self.kv_lora_rank * kv_out
            * dtype_bytes(self.kv_transform_dtype)
            + gemm_scale_bytes(
                kv_out, self.kv_lora_rank, self.kv_transform_dtype)
        )

    @property
    def output_bytes(self) -> float:
        return (self.B * self.S_q * self.H * self.v_head_dim
                * dtype_bytes(self.out_dtype))


class TokenDispatch(Kernel):
    """MoE token dispatch: routing score + topk selection + token scatter.

    Softmax costs 5·M·N_experts; sigmoid costs 3·M·N_experts.
    bytes:
        input_bytes  = M·D·sizeof(a_dtype) + M·N_experts·4
        output_bytes = M·topk·D·sizeof(a_dtype)
    """

    def __init__(self, M: int, D: int, N_experts: int, topk: int,
                 a_dtype: str = "bf16", scoring_func: str = "softmax"):
        if scoring_func not in {"softmax", "sigmoid"}:
            raise ValueError(
                f"unsupported routing scoring function: {scoring_func}")
        self.M, self.D = M, D
        self.N_experts = N_experts
        self.topk = topk
        self.a_dtype = a_dtype
        self.scoring_func = scoring_func
        self.M_e = Fraction(M * topk, N_experts)
        super().__init__()

    @property
    def flops(self) -> float:
        score_flops = 5.0 if self.scoring_func == "softmax" else 3.0
        return score_flops * self.M * self.N_experts

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
