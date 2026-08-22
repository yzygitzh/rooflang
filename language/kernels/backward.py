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


class PartialRMSNorm(Kernel):
    """Backward for RMSNorm on a prefix with an identity suffix.

    Only the normalized prefix performs arithmetic and contributes a gamma
    gradient.  The full activation and upstream gradient are read because the
    untouched suffix gradient is forwarded to the full-size ``dx`` tensor.
    """

    def __init__(self, M: int, input_dim: int, norm_dim: int,
                 dtype: str = "bf16", grad_dtype: str = "fp32"):
        if not 0 < norm_dim <= input_dim:
            raise ValueError(
                f"norm_dim must be in [1, input_dim], got {norm_dim}")
        self.M, self.input_dim = M, input_dim
        self.norm_dim, self.dtype_ = norm_dim, dtype
        self.grad_dtype = grad_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 9.0 * self.M * self.norm_dim

    @property
    def input_bytes(self) -> float:
        return 2 * self.M * self.input_dim * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return self.norm_dim * dtype_bytes(self.dtype_)

    @property
    def output_bytes(self) -> float:
        return (self.M * self.input_dim * dtype_bytes(self.dtype_)
                + self.norm_dim * dtype_bytes(self.grad_dtype))


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


class AttnRes(Kernel):
    """Backward for Kimi AttnRes weighted residual aggregation.

    The fused backward recomputes normalization scores and probabilities,
    differentiates the weighted reduction, softmax, score projection, and
    per-candidate RMS normalization, then accumulates gradients for the two
    length-``D`` parameter vectors.  All arithmetic follows the FP32 cast in
    the reference implementation while activations retain their storage
    dtype in HBM.
    """

    def __init__(self, B: int, S: int, D: int, R: int,
                 dtype: str = "bf16", compute_dtype: str = "fp32",
                 grad_dtype: str = "fp32"):
        self.B, self.S, self.D, self.R = B, S, D, R
        self.dtype_ = compute_dtype
        self.storage_dtype = dtype
        self.grad_dtype = grad_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        candidates = self.R + 1
        tokens = self.B * self.S
        return float(
            18 * tokens * candidates * self.D
            + 9 * tokens * candidates
            + 3 * self.D
        )

    @property
    def input_bytes(self) -> float:
        tokens = self.B * self.S
        values = tokens * (self.R + 1) * self.D
        upstream = tokens * self.D
        return (values + upstream) * dtype_bytes(self.storage_dtype)

    @property
    def weight_bytes(self) -> float:
        return 2 * self.D * dtype_bytes(self.storage_dtype)

    @property
    def output_bytes(self) -> float:
        activation_grads = self.B * self.S * (self.R + 1) * self.D
        return (activation_grads * dtype_bytes(self.storage_dtype)
                + 2 * self.D * dtype_bytes(self.grad_dtype))


class Attn(Kernel):
    """Flash-Attention v2 backward (recompute S from saved LSE).

    flops = 10·B·H·S_q·S_kv·Hd (×0.5 for a triangular causal
    matrix).
    bytes (per-tensor read-once / write-once; strict lower bound):
        input_bytes  = (2·B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
        output_bytes = (B·H·S_q·Hd + 2·B·H_kv·S_kv·Hd) · sizeof(dtype)
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
        f = 10.0 * self.B * self.H * self.S_q * self.S_kv * self.Hd
        if self.triangular:
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


class KimiK3MlaAttn(Kernel):
    """Backward for Kimi-K3 dense MLA with an absorbed latent KV core.

    The main attention recomputes QK scores and differentiates both QK and
    PV contractions.  The absorbed KV-B transform has separate activation
    and weight-gradient contractions, matching ``Glm52SparseAttn`` without
    its sparse indexer path.
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
        grad_dtype: str = "fp32",
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
        self.grad_dtype = grad_dtype
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
        per_pair = 6 * self.kv_cache_dim + 4 * self.kv_lora_rank
        return float(self.B * self.H * self.selected_pairs * per_pair)

    @property
    def kv_transform_flops(self) -> float:
        kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
        return float(
            4 * self.B * self.kv_transform_tokens
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
            + self.B * self.S_q * self.H * self.v_head_dim
            * dtype_bytes(self.out_dtype)
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
        kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
        return (
            self.B * self.S_q * self.H * self.qk_head_dim
            * dtype_bytes(self.q_dtype)
            + self.B * self.S_kv * self.kv_cache_dim
            * dtype_bytes(self.kv_dtype)
            + self.kv_lora_rank * kv_out
            * dtype_bytes(self.grad_dtype)
        )


class KimiK3DeltaAttn(Kernel):
    """Backward for Kimi Delta Attention.

    FLA's chunk backward recomputes the local WY representation and recurrent
    states before evaluating the two matrix-gradient paths.  The dominant
    KDA core is therefore modeled as three times its forward arithmetic.  The
    fused short-convolution, Q/K normalization, and gate preprocessing has
    data- and parameter-gradient paths and is modeled as twice forward.

    ``recurrent`` is retained as a roofline estimate for completeness even
    though the Kimi reference implementation only enables ``chunk`` mode in
    training.
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
        grad_dtype: str = "fp32",
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
        self.grad_dtype = grad_dtype
        super().__init__()

    @property
    def attention_flops(self) -> float:
        if self.mode == "chunk":
            forward_per_head = (
                6 * self.S * self.K * self.V
                + 3 * self.S * self.chunk_size * self.K
                + self.S * self.chunk_size ** 2
            )
        else:
            forward_per_head = (
                7 * self.S * self.K * self.V
                + 2 * self.S * self.V
            )
        return float(3 * self.B * self.H * forward_per_head)

    @property
    def preprocessing_flops(self) -> float:
        elements = self.B * self.H * self.S
        conv = 3 * elements * self.K * (2 * self.conv_size + 4)
        qk_norm = 2 * elements * (3 * self.K + 1)
        gate = elements * (5 * self.K + 3)
        return float(2 * (conv + qk_norm + gate))

    @property
    def flops(self) -> float:
        return self.attention_flops + self.preprocessing_flops

    @property
    def input_bytes(self) -> float:
        tokens = self.B * self.H * self.S
        activations = tokens * (2 * self.K + 3 * self.V)
        total = (activations * dtype_bytes(self.dtype_)
                 + tokens * dtype_bytes("fp32"))
        if self.mode == "recurrent":
            state = self.B * self.H * self.K * self.V
            conv_state = (self.B * self.H * (2 * self.K + self.V)
                          * self.conv_size)
            total += ((state + conv_state)
                      * dtype_bytes(self.state_dtype))
        return total

    @property
    def weight_bytes(self) -> float:
        conv_params = self.H * (2 * self.K + self.V) * (self.conv_size + 1)
        gate_params = self.H * (self.V + 1)
        return (conv_params * dtype_bytes(self.dtype_)
                + gate_params * dtype_bytes("fp32"))

    @property
    def output_bytes(self) -> float:
        tokens = self.B * self.H * self.S
        activation_grads = tokens * (2 * self.K + 2 * self.V)
        parameter_grads = (
            self.H * (2 * self.K + self.V) * (self.conv_size + 1)
            + self.H * (self.V + 1)
        )
        total = (activation_grads * dtype_bytes(self.dtype_)
                 + tokens * dtype_bytes("fp32")
                 + parameter_grads * dtype_bytes(self.grad_dtype))
        if self.mode == "recurrent":
            state = self.B * self.H * self.K * self.V
            conv_state = (self.B * self.H * (2 * self.K + self.V)
                          * self.conv_size)
            total += ((state + conv_state)
                      * dtype_bytes(self.state_dtype))
        return total


class KimiK3DeltaAttnCpSummary(Kernel):
    """Backward of a rank-local KDA transition summary.

    The reverse summary construction has two matrix-gradient paths.  Its
    gradients into the local WY representation are fused into the ordinary
    KDA backward, so this kernel only reads the materialized summary/halo
    gradients and does not write another copy of local activation gradients.
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
        summary_dtype: str = "fp32",
    ):
        self.B, self.H, self.S = B, H, S
        self.K, self.V = K, V
        self.rank, self.world = rank, world
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.dtype_ = dtype
        self.state_dtype = state_dtype
        self.summary_dtype = summary_dtype
        super().__init__()

    @property
    def n_chunks(self) -> int:
        return (self.S + self.chunk_size - 1) // self.chunk_size

    @property
    def bf16_flops(self) -> float:
        if self.rank == self.world - 1:
            return 0.0
        forward_per_head = (
            4 * self.S * self.K * self.V
            + 2 * self.n_chunks * self.K * self.V
            + self.S * self.V
            + 2 * self.S * self.K * self.K
            + self.n_chunks * self.K * self.K
        )
        return float(2 * self.B * self.H * forward_per_head)

    @property
    def fp32_flops(self) -> float:
        if self.rank == self.world - 1:
            return 0.0
        return float(4 * self.B * self.H * self.n_chunks * self.K ** 3)

    @property
    def flops(self) -> float:
        return self.bf16_flops + self.fp32_flops

    @property
    def flops_by_dtype(self) -> dict[str, float]:
        return {self.dtype_: self.bf16_flops, "fp32": self.fp32_flops}

    @property
    def input_bytes(self) -> float:
        if self.rank == self.world - 1:
            return 0.0
        summary = self.B * self.H * self.K * (self.K + self.V)
        halo = (self.B * self.H * 3 * self.K
                * max(0, self.conv_size - 1))
        return (summary * dtype_bytes(self.summary_dtype)
                + halo * dtype_bytes(self.state_dtype))

    @property
    def output_bytes(self) -> float:
        return 0.0


class KimiK3DeltaAttnCpMerge(Kernel):
    """Backward of the prefix transition merge for one CP rank."""

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
        forward_per_summary = 2 * self.K * self.K * self.V + self.K * self.V
        return float(2 * self.B * self.H * self.rank * forward_per_summary)

    @property
    def input_bytes(self) -> float:
        summaries = self.B * self.H * self.rank * self.K * (self.K + self.V)
        state = self.B * self.H * self.K * self.V
        return (summaries * dtype_bytes(self.summary_dtype)
                + state * dtype_bytes(self.state_dtype))

    @property
    def output_bytes(self) -> float:
        summaries = self.B * self.H * self.rank * self.K * (self.K + self.V)
        return summaries * dtype_bytes(self.summary_dtype)


class KimiK3DeltaAttnStateStore(Kernel):
    """No-op backward marker for detached KDA cache persistence.

    Recurrent and short-convolution cache state is an inference side effect,
    not a differentiable training output.  No gradient traverses this node.
    """

    _requires_placement = False

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

    @property
    def input_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 0.0


class DpskV4SparseAttn(Kernel):
    """DeepSeek V4 sparse-attention backward with a fused indexer backward.

    flops = 10·B·H·S_q·effective_k_sel·Hd.  During causal prefill,
    only the context-dependent ``causal_k_sel`` portion receives the 0.5
    triangular factor.  The indexer backward recomputes its score matrix
    instead of materializing it: score recompute + dQ + dK costs 6 FLOPs per
    index-head dot-product element, while the ReLU, head-weight, and reduction
    backward costs 5 scalar FLOPs.  Both receive the causal factor during
    prefill.
    bytes (per-tensor read-once / write-once):
        input_bytes reads saved Q at ``q_dtype``, upstream dY at
        ``out_dtype``, and selected KV at ``kv_dtype``.
        output_bytes writes dQ at ``q_dtype`` and dKV at ``kv_dtype``.
    Indexer bytes additionally read quantized index Q/KV, head weights, and
    the upstream reduced-score gradient, then write dQ, dKV, and dWeights at
    their corresponding activation dtypes.

    ``dtype`` selects the device TFLOPS used for compute time.  Q, main KV,
    and attention output storage may independently use ``q_dtype``,
    ``kv_dtype``, and ``out_dtype``; each defaults to ``dtype`` for backward
    compatibility.  Fused indexer FLOPs use ``indexer_compute_dtype``, which
    defaults to the index-cache storage ``indexer_dtype``.
    """

    def __init__(self, B: int, H: int, H_kv: int,
                 S_q: int, k_sel: int, Hd: int,
                 dtype: str = "bf16", kv_factor: int = 2,
                 indexer_s_kv: int = 0, indexer_h: int = 0,
                 indexer_hd: int = 0, indexer_dtype: str = "fp4",
                 indexer_compute_dtype: str | None = None,
                 *, q_dtype: str | None = None,
                 kv_dtype: str | None = None,
                 out_dtype: str | None = None,
                 causal: bool = False, causal_k_sel: int = 0):
        self.B, self.H, self.H_kv = B, H, H_kv
        self.S_q, self.k_sel, self.Hd = S_q, k_sel, Hd
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
    def effective_k_sel(self) -> float:
        if self.causal:
            return self.k_sel - 0.5 * self.causal_k_sel
        return self.k_sel

    @property
    def attention_flops(self) -> float:
        return (10.0 * self.B * self.H * self.S_q
                * self.effective_k_sel * self.Hd)

    @property
    def indexer_flops(self) -> float:
        factor = 0.5 if self.causal else 1.0
        score_backward = (6.0 * self.B * self.S_q * self.indexer_h
                          * self.indexer_s_kv * self.indexer_hd)
        reduce_backward = (5.0 * self.B * self.S_q * self.indexer_h
                           * self.indexer_s_kv)
        return (score_backward + reduce_backward) * factor

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

    @property
    def indexer_input_bytes(self) -> float:
        index_values = (
            self.B * self.S_q * self.indexer_h * self.indexer_hd
            + self.B * self.indexer_s_kv * self.indexer_hd
        ) * dtype_bytes(self.indexer_dtype)
        head_weights = (self.B * self.S_q * self.indexer_h
                        * dtype_bytes(self.q_dtype))
        reduced_score_grad = (self.B * self.S_q * self.indexer_s_kv
                              * dtype_bytes(self.out_dtype))
        return index_values + head_weights + reduced_score_grad

    @property
    def indexer_output_bytes(self) -> float:
        index_q_grad = (self.B * self.S_q * self.indexer_h
                        * self.indexer_hd * dtype_bytes(self.q_dtype))
        index_kv_grad = (self.B * self.indexer_s_kv * self.indexer_hd
                         * dtype_bytes(self.kv_dtype))
        head_weight_grad = (self.B * self.S_q * self.indexer_h
                            * dtype_bytes(self.q_dtype))
        return index_q_grad + index_kv_grad + head_weight_grad

    @property
    def input_bytes(self) -> float:
        q_elements = self.B * self.H * self.S_q * self.Hd
        kv_elements = (self.kv_factor * self.B * self.H_kv * self.S_q
                       * self.effective_k_sel * self.Hd)
        main_attention = (
            q_elements * dtype_bytes(self.q_dtype)
            + q_elements * dtype_bytes(self.out_dtype)
            + kv_elements * dtype_bytes(self.kv_dtype)
        )
        return main_attention + self.indexer_input_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        q_elements = self.B * self.H * self.S_q * self.Hd
        kv_elements = (self.kv_factor * self.B * self.H_kv * self.S_q
                       * self.effective_k_sel * self.Hd)
        main_attention = (
            q_elements * dtype_bytes(self.q_dtype)
            + kv_elements * dtype_bytes(self.kv_dtype)
        )
        return main_attention + self.indexer_output_bytes


class Glm52SparseAttn(Kernel):
    """Backward for GLM-5.2 DSA with full/shared indexer modes."""

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
        grad_dtype: str = "fp32",
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
        self.grad_dtype = grad_dtype
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
        per_pair = 6 * self.kv_cache_dim + 4 * self.kv_lora_rank
        return float(self.B * self.H * self.selected_pairs * per_pair)

    @property
    def kv_transform_flops(self) -> float:
        if self.kv_lora_rank == 0:
            return 0.0
        kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
        return float(
            4 * self.B * self.kv_transform_tokens
            * self.kv_lora_rank * kv_out)

    @property
    def indexer_score_flops(self) -> float:
        return float(
            6 * self.B * self.indexer_h
            * self.indexer_pairs * self.indexer_hd)

    @property
    def indexer_reduce_flops(self) -> float:
        return float(
            5 * self.B * self.indexer_h * self.indexer_pairs)

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
            + self.B * self.S_q * self.H * self.v_head_dim
            * dtype_bytes(self.out_dtype)
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
            + self.B * self.S_q * self.indexer_s_kv
            * dtype_bytes(self.indexer_reduce_dtype)
        )
        return main + indexer

    @property
    def input_bytes(self) -> float:
        main_kv_reads = min(self.S_kv, self.S_q * self.k_sel)
        main = (
            self.B * self.S_q * self.H * self.qk_head_dim
            * dtype_bytes(self.q_dtype)
            + self.B * self.S_q * self.H * self.v_head_dim
            * dtype_bytes(self.out_dtype)
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
            + self.B * self.S_q * self.indexer_s_kv
            * dtype_bytes(self.indexer_reduce_dtype)
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
        main = (
            self.B * self.S_q * self.H * self.qk_head_dim
            * dtype_bytes(self.q_dtype)
            + self.B * self.S_kv * self.kv_cache_dim
            * dtype_bytes(self.kv_dtype)
        )
        if self.kv_lora_rank:
            kv_out = self.H * (self.qk_nope_head_dim + self.v_head_dim)
            main += (self.kv_lora_rank * kv_out
                     * dtype_bytes(self.grad_dtype))
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


class TokenDispatch(Kernel):
    """TokenDispatch backward: gather gradients + routing-score backward.

    Softmax backward costs 4·M·N_experts; sigmoid costs 3·M·N_experts.
    bytes:
        input_bytes  = M·topk·D·sizeof(a_dtype) + M·topk·4 + M·N_experts·4
                       (d_scattered + d_routing_weights + saved routing_probs)
        output_bytes = M·D·sizeof(a_dtype) + M·N_experts·4
                       (d_x + d_routing_logits)
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
        score_flops = 4.0 if self.scoring_func == "softmax" else 3.0
        return (self.M * self.topk * self.D
                + score_flops * self.M * self.N_experts)

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
        self.M_e = Fraction(M * topk, N_experts)
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


class ElementwiseOp(Kernel):
    """Backward of element-wise binary op.

    "add" backward: da = dy, db = dy (gradient copied to both inputs).
      flops = 0
      input  = M·D · sizeof(dtype)          (read dy)
      output = 2·M·D · sizeof(grad_dtype)   (write da, db)

    "mul" backward: da = dy * b, db = dy * a
      flops = 2·M·D
      input  = 3·M·D · sizeof(dtype)        (read dy, saved a, saved b)
      output = 2·M·D · sizeof(grad_dtype)   (write da, db)

    "sigmoid_mul" backward differentiates ``a * sigmoid(b)`` while
    recomputing sigmoid(b).
      flops = 9·M·D
      input  = 3·M·D · sizeof(dtype)
      output = 2·M·D · sizeof(grad_dtype)
    """

    def __init__(self, M: int, D: int, dtype: str = "bf16",
                 op: str = "add", grad_dtype: str = "fp32"):
        self.M, self.D, self.dtype_ = M, D, dtype
        self.op = op
        self.grad_dtype = grad_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        if self.op == "add":
            return 0.0
        if self.op == "sigmoid_mul":
            return 9.0 * self.M * self.D
        return 2.0 * self.M * self.D

    @property
    def input_bytes(self) -> float:
        if self.op == "add":
            return float(self.M * self.D) * dtype_bytes(self.dtype_)
        return 3.0 * self.M * self.D * dtype_bytes(self.dtype_)

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 2.0 * self.M * self.D * dtype_bytes(self.grad_dtype)


class Sampling(Kernel):
    """Backward of sampling: fused softmax-cross-entropy gradient.

    In training, argmax/sampling is replaced by cross-entropy loss on
    logits. The backward computes dL/d_logits = softmax(logits) - one_hot.

    flops = 5·M·V (exp, sum, div, sub, scale per element).
    bytes:
        input_bytes  = M·V · sizeof(dtype) + M · sizeof(idx_dtype)
                       (read logits + target token IDs)
        output_bytes = M·V · sizeof(grad_dtype)  (write dL/d_logits)
    """

    def __init__(self, M: int, V: int,
                 dtype: str = "bf16", idx_dtype: str = "int32",
                 grad_dtype: str = "fp32"):
        self.M, self.V = M, V
        self.dtype_ = dtype
        self.idx_dtype = idx_dtype
        self.grad_dtype = grad_dtype
        super().__init__()

    @property
    def flops(self) -> float:
        return 5.0 * self.M * self.V

    @property
    def input_bytes(self) -> float:
        return (self.M * self.V * dtype_bytes(self.dtype_)
                + self.M * dtype_bytes(self.idx_dtype))

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.M * self.V * dtype_bytes(self.grad_dtype)
