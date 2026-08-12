"""Unit tests for rooflang.language.kernels (Kernel base + all subclasses)."""

import pytest

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.forward import (
    ElementwiseOp, Embedding, Gemm, Nop, ReadInput, RMSNorm, LayerNorm, RoPE,
    Attn, Slice, SparseAttn, Sampling, TokenDispatch, TokenCombine,
)
from rooflang.language.kernels import backward
from rooflang.language.kernels.comm import (
    AllReduce, ReduceScatter, AllGather, AllToAll, Broadcast,
    Scatter, Gather, Reduce, Send, Recv,
)
from rooflang.language.kernels.optimizer import AdamWStep
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.tensor import Tensor


# ── Base Kernel class tests ──────────────────────────────────────────


class TestKernelInit:
    def test_defaults(self):
        k = Kernel()
        assert k.inputs == {}
        assert k.weights == {}
        assert k.outputs == {}
        assert k.has_side_effect is False

    def test_with_tensors(self):
        k = Kernel(inputs={"x": Tensor("bf16", (4,))})
        assert "x" in k.inputs


class TestKernelToDict:
    def test_full_dict(self):
        k = Kernel(inputs={"x": Tensor("bf16", (4,))})
        expected = {
            "flops": 0.0,
            "transferred_bytes": 8.0,
            "input_bytes": 8.0,
            "weight_bytes": 0.0,
            "output_bytes": 0.0,
            "inputs": {"x": {"dtype": "bf16", "shape": (4,)}},
        }
        assert k.to_dict() == expected

    def test_side_effect_included(self):
        assert Kernel(has_side_effect=True).to_dict()["has_side_effect"] is True

    def test_no_side_effect_excluded(self):
        assert "has_side_effect" not in Kernel().to_dict()

    def test_weights_and_outputs_included(self):
        k = Kernel(
            weights={"W": Tensor("bf16", (8, 8))},
            outputs={"y": Tensor("fp32", (4,))},
        )
        d = k.to_dict()
        assert "weights" in d
        assert d["weights"] == {"W": {"dtype": "bf16", "shape": (8, 8)}}
        assert "outputs" in d
        assert d["outputs"] == {"y": {"dtype": "fp32", "shape": (4,)}}


# ── Shared base for all kernel subclass tests ────────────────────────


class TestKernelBase:
    __test__ = False

    kernel = None
    expected_flops = None
    expected_input_bytes = None
    expected_weight_bytes = None
    expected_output_bytes = None
    expected_transferred_bytes = None

    def test_flops(self):
        assert self.kernel.flops == self.expected_flops

    def test_input_bytes(self):
        assert self.kernel.input_bytes == self.expected_input_bytes

    def test_weight_bytes(self):
        assert self.kernel.weight_bytes == self.expected_weight_bytes

    def test_output_bytes(self):
        assert self.kernel.output_bytes == self.expected_output_bytes

    def test_transferred_bytes(self):
        if self.expected_transferred_bytes is not None:
            assert self.kernel.transferred_bytes == self.expected_transferred_bytes
        else:
            assert self.kernel.transferred_bytes == (
                self.expected_input_bytes
                + self.expected_weight_bytes
                + self.expected_output_bytes
            )


# ── Forward kernels ──────────────────────────────────────────────────


class TestNop(TestKernelBase):
    __test__ = True
    kernel = Nop(
        inputs={
            "x": Tensor("bf16", (4, 8)),
            "meta": Tensor("int32", (3,)),
        },
        outputs={
            "done": Tensor("fp32", (7,)),
            "token": Tensor("int32", (1,)),
        },
    )
    expected_flops = 0.0
    expected_input_bytes = 0.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 0.0
    expected_transferred_bytes = 0.0

    def test_requires_no_placement(self):
        assert self.kernel._requires_placement is False


class TestReadInput(TestKernelBase):
    __test__ = True
    kernel = ReadInput(n_elements=64 * 8192, dtype="int32")
    expected_flops = 0.0
    expected_input_bytes = 64 * 8192 * 4.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 64 * 8192 * 4.0


class TestSlice(TestKernelBase):
    __test__ = True
    kernel = Slice()
    kernel.inputs = {"x": Tensor("bf16", (16,))}
    kernel.outputs = {"y": Tensor("bf16", (4,))}
    expected_flops = 0.0
    expected_input_bytes = 16 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 4 * 2.0

    def test_requires_placement_true(self):
        assert self.kernel._requires_placement is True


class TestEmbedding(TestKernelBase):
    __test__ = True
    kernel = Embedding(M=8192, V=129280, D=7168, w_dtype="bf16")
    expected_flops = 0.0
    expected_input_bytes = 8192 * 4.0
    expected_weight_bytes = 129280 * 7168 * 2.0
    expected_output_bytes = 8192 * 7168 * 2.0


class TestGemm(TestKernelBase):
    __test__ = True
    kernel = Gemm(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
    expected_flops = 2.0 * 32 * 64 * 128
    expected_input_bytes = 32 * 128 * 2.0
    expected_weight_bytes = 128 * 64 * 2.0
    expected_output_bytes = 32 * 64 * 2.0


class TestRMSNorm(TestKernelBase):
    __test__ = True
    kernel = RMSNorm(M=16, D=64)
    expected_flops = 4.0 * 16 * 64
    expected_input_bytes = 16 * 64 * 2.0
    expected_weight_bytes = 64 * 2.0
    expected_output_bytes = 16 * 64 * 2.0


class TestLayerNorm(TestKernelBase):
    __test__ = True
    kernel = LayerNorm(M=16, D=64)
    expected_flops = 7.0 * 16 * 64
    expected_input_bytes = 16 * 64 * 2.0
    expected_weight_bytes = 2 * 64 * 2.0
    expected_output_bytes = 16 * 64 * 2.0


class TestRoPE(TestKernelBase):
    __test__ = True
    kernel = RoPE(M=16, D=64)
    expected_flops = 3.0 * 16 * 64
    expected_input_bytes = 16 * 64 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 16 * 64 * 2.0


class TestAttn(TestKernelBase):
    __test__ = True
    kernel = Attn(B=2, H=8, H_kv=8, S_q=256, S_kv=256, Hd=64)
    expected_flops = 4.0 * 2 * 8 * 256 * 256 * 64
    expected_input_bytes = (2 * 8 * 256 * 64 + 2 * 2 * 8 * 256 * 64) * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 2 * 8 * 256 * 64 * 2.0

    def test_flops_causal_halved(self):
        a = Attn(B=2, H=8, H_kv=8, S_q=256, S_kv=256, Hd=64, causal=True)
        assert a.flops == self.expected_flops * 0.5

    def test_flops_causal_asymmetric_not_halved(self):
        a = Attn(B=2, H=8, H_kv=8, S_q=128, S_kv=256, Hd=64, causal=True)
        assert a.flops == 4.0 * 2 * 8 * 128 * 256 * 64


class TestSparseAttn(TestKernelBase):
    __test__ = True
    kernel = SparseAttn(B=2, H=8, H_kv=8, S_q=256, k_sel=64, S_kv=64, Hd=64)
    expected_flops = 4.0 * 2 * 8 * 256 * 64 * 64
    expected_input_bytes = (2 * 8 * 256 * 64 + 2 * 2 * 8 * 64 * 64) * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 2 * 8 * 256 * 64 * 2.0


class TestElementwiseOpAdd(TestKernelBase):
    __test__ = True
    kernel = ElementwiseOp(M=8192, D=7168, dtype="bf16", op="add")
    expected_flops = 8192.0 * 7168
    expected_input_bytes = 2 * 8192 * 7168 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 7168 * 2.0


class TestElementwiseOpMul(TestKernelBase):
    __test__ = True
    kernel = ElementwiseOp(M=8192, D=7168, dtype="bf16", op="mul")
    expected_flops = 8192.0 * 7168
    expected_input_bytes = 2 * 8192 * 7168 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 7168 * 2.0


class TestSampling(TestKernelBase):
    __test__ = True
    kernel = Sampling(M=512, V=129280, dtype="bf16", out_dtype="int32")
    expected_flops = 512.0 * 129280
    expected_input_bytes = 512 * 129280 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 512 * 4.0


# ── Backward kernels ─────────────────────────────────────────────────


class TestBwdNop(TestKernelBase):
    __test__ = True
    kernel = backward.Nop(
        inputs={"dx": Tensor("bf16", (2, 3))},
        outputs={"done": Tensor("int32", (5,))},
    )
    expected_flops = 0.0
    expected_input_bytes = 0.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 0.0
    expected_transferred_bytes = 0.0

    def test_requires_no_placement(self):
        assert self.kernel._requires_placement is False


class TestBwdReadInput(TestKernelBase):
    __test__ = True
    kernel = backward.ReadInput(n_elements=64 * 8192, dtype="int32")
    expected_flops = 0.0
    expected_input_bytes = 64 * 8192 * 4.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 64 * 8192 * 4.0


class TestBwdEmbedding(TestKernelBase):
    __test__ = True
    kernel = backward.Embedding(M=8192, V=129280, D=7168)
    expected_flops = 8192.0 * 7168
    expected_input_bytes = 8192 * 7168 * 2.0 + 8192 * 4.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 7168 * 4.0


class TestGemmDX(TestKernelBase):
    __test__ = True
    kernel = backward.GemmDX(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
    expected_flops = 2.0 * 32 * 64 * 128
    expected_input_bytes = 32 * 64 * 2.0
    expected_weight_bytes = 64 * 128 * 2.0
    expected_output_bytes = 32 * 128 * 2.0


class TestGemmDW(TestKernelBase):
    __test__ = True
    kernel = backward.GemmDW(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
    expected_flops = 2.0 * 32 * 64 * 128
    expected_input_bytes = 32 * 64 * 2.0 + 32 * 128 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 64 * 128 * 4.0


class TestBwdRMSNorm(TestKernelBase):
    __test__ = True
    kernel = backward.RMSNorm(M=16, D=64)
    expected_flops = 9.0 * 16 * 64
    expected_input_bytes = 2 * 16 * 64 * 2.0
    expected_weight_bytes = 64 * 2.0
    expected_output_bytes = 16 * 64 * 2.0 + 64 * 4.0


class TestBwdLayerNorm(TestKernelBase):
    __test__ = True
    kernel = backward.LayerNorm(M=16, D=64)
    expected_flops = 11.0 * 16 * 64
    expected_input_bytes = 2 * 16 * 64 * 2.0
    expected_weight_bytes = 64 * 2.0
    expected_output_bytes = 16 * 64 * 2.0 + 2 * 64 * 4.0


class TestBwdRoPE(TestKernelBase):
    __test__ = True
    kernel = backward.RoPE(M=16, D=64)
    expected_flops = 3.0 * 16 * 64
    expected_input_bytes = 16 * 64 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 16 * 64 * 2.0


class TestBwdAttn(TestKernelBase):
    __test__ = True
    kernel = backward.Attn(B=2, H=8, H_kv=8, S_q=256, S_kv=256, Hd=64)
    expected_flops = 10.0 * 2 * 8 * 256 * 256 * 64
    expected_input_bytes = (2 * 2 * 8 * 256 * 64 + 2 * 2 * 8 * 256 * 64) * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = (2 * 8 * 256 * 64 + 2 * 2 * 8 * 256 * 64) * 2.0

    def test_flops_causal_halved(self):
        k = backward.Attn(B=2, H=8, H_kv=8, S_q=256, S_kv=256, Hd=64,
                          causal=True)
        assert k.flops == self.expected_flops * 0.5


class TestBwdSparseAttn(TestKernelBase):
    __test__ = True
    kernel = backward.SparseAttn(B=2, H=8, H_kv=8, S_q=256, k_sel=64, Hd=64)
    expected_flops = 10.0 * 2 * 8 * 256 * 64 * 64
    expected_input_bytes = (2 * 2 * 8 * 256 * 64 + 2 * 2 * 8 * 256 * 64 * 64) * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = (2 * 8 * 256 * 64 + 2 * 2 * 8 * 256 * 64 * 64) * 2.0


class TestBwdElementwiseOpAdd(TestKernelBase):
    __test__ = True
    kernel = backward.ElementwiseOp(M=8192, D=7168, dtype="bf16", op="add")
    expected_flops = 0.0
    expected_input_bytes = 8192 * 7168 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 2 * 8192 * 7168 * 4.0


class TestBwdElementwiseOpMul(TestKernelBase):
    __test__ = True
    kernel = backward.ElementwiseOp(M=8192, D=7168, dtype="bf16", op="mul")
    expected_flops = 2.0 * 8192 * 7168
    expected_input_bytes = 3 * 8192 * 7168 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 2 * 8192 * 7168 * 4.0


class TestBwdSampling(TestKernelBase):
    __test__ = True
    kernel = backward.Sampling(M=512, V=129280, dtype="bf16")
    expected_flops = 5.0 * 512 * 129280
    expected_input_bytes = 512 * 129280 * 2.0 + 512 * 4.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 512 * 129280 * 4.0


# ── Comm kernels ─────────────────────────────────────────────────────


class TestAllReduce(TestKernelBase):
    __test__ = True
    kernel = AllReduce(total_bytes=1024.0, world=4, dtype="bf16")
    expected_flops = (3 / 4) * (1024.0 / 2.0)
    expected_input_bytes = 1024.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0
    expected_transferred_bytes = 2.0 * (3 / 4) * 1024.0


class TestReduceScatter(TestKernelBase):
    __test__ = True
    kernel = ReduceScatter(total_bytes=1024.0, world=4, dtype="bf16")
    expected_flops = (3 / 4) * (1024.0 / 2.0)
    expected_input_bytes = 1024.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0 / 4
    expected_transferred_bytes = (3 / 4) * 1024.0


class TestAllGather(TestKernelBase):
    __test__ = True
    kernel = AllGather(total_bytes=1024.0, world=4)
    expected_flops = 0.0
    expected_input_bytes = 1024.0 / 4
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0
    expected_transferred_bytes = (3 / 4) * 1024.0


class TestAllToAll(TestKernelBase):
    __test__ = True
    kernel = AllToAll(total_bytes=1024.0, world=4)
    expected_flops = 0.0
    expected_input_bytes = 1024.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0
    expected_transferred_bytes = (3 / 4) * 1024.0


class TestBroadcast(TestKernelBase):
    __test__ = True
    kernel = Broadcast(total_bytes=1024.0, world=4)
    expected_flops = 0.0
    expected_input_bytes = 1024.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0
    expected_transferred_bytes = 1024.0


class TestScatter(TestKernelBase):
    __test__ = True
    kernel = Scatter(total_bytes=1024.0, world=4)
    expected_flops = 0.0
    expected_input_bytes = 1024.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0 / 4
    expected_transferred_bytes = (3 / 4) * 1024.0


class TestGather(TestKernelBase):
    __test__ = True
    kernel = Gather(total_bytes=1024.0, world=4)
    expected_flops = 0.0
    expected_input_bytes = 1024.0 / 4
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0
    expected_transferred_bytes = (3 / 4) * 1024.0


class TestReduce(TestKernelBase):
    __test__ = True
    kernel = Reduce(total_bytes=1024.0, world=4, dtype="bf16")
    expected_flops = (3 / 4) * (1024.0 / 2.0)
    expected_input_bytes = 1024.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 1024.0
    expected_transferred_bytes = 1024.0


class TestSend(TestKernelBase):
    __test__ = True
    kernel = Send(total_bytes=512.0)
    expected_flops = 0.0
    expected_input_bytes = 512.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 0.0
    expected_transferred_bytes = 512.0


class TestRecv(TestKernelBase):
    __test__ = True
    kernel = Recv(total_bytes=512.0)
    expected_flops = 0.0
    expected_input_bytes = 0.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 512.0
    expected_transferred_bytes = 512.0


# ── Optimizer kernels ────────────────────────────────────────────────


class TestAdamWStep(TestKernelBase):
    __test__ = True
    kernel = AdamWStep(n_param=1000)
    expected_flops = 13.0 * 1000
    expected_input_bytes = 1000 * 4.0
    expected_weight_bytes = 1000 * (4.0 + 2 * 4.0)
    expected_output_bytes = 1000 * (4.0 + 2 * 4.0)

    def test_input_bytes_bf16_grad(self):
        opt = AdamWStep(n_param=1000, grad_dtype="bf16")
        assert opt.input_bytes == 1000 * 2.0


# ── MoE dispatch/combine kernels ────────────────────────────────────


class TestTokenDispatch(TestKernelBase):
    __test__ = True
    kernel = TokenDispatch(M=8192, D=7168, N_experts=384, topk=6)
    expected_flops = 5.0 * 8192 * 384
    expected_input_bytes = 8192 * 7168 * 2.0 + 8192 * 384 * 4.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 6 * 7168 * 2.0

    def test_m_e(self):
        assert self.kernel.M_e == 8192 * 6 // 384


class TestTokenCombine(TestKernelBase):
    __test__ = True
    kernel = TokenCombine(M=8192, D=7168, N_experts=384, topk=6)
    expected_flops = 2.0 * 8192 * 6 * 7168
    expected_input_bytes = 8192 * 6 * 7168 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 7168 * 2.0

    def test_m_e(self):
        assert self.kernel.M_e == 8192 * 6 // 384


class TestBwdTokenDispatch(TestKernelBase):
    __test__ = True
    kernel = backward.TokenDispatch(M=8192, D=7168, N_experts=384, topk=6)
    expected_flops = 8192 * 6 * 7168 + 4.0 * 8192 * 384
    expected_input_bytes = (8192 * 6 * 7168 * 2.0
                            + 8192 * 6 * 4.0
                            + 8192 * 384 * 4.0)
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 7168 * 2.0 + 8192 * 384 * 4.0

    def test_m_e(self):
        assert self.kernel.M_e == 8192 * 6 // 384


class TestBwdTokenCombine(TestKernelBase):
    __test__ = True
    kernel = backward.TokenCombine(M=8192, D=7168, N_experts=384, topk=6)
    expected_flops = 3.0 * 8192 * 6 * 7168
    expected_input_bytes = (8192 * 7168 * 2.0
                            + 8192 * 6 * 7168 * 2.0
                            + 8192 * 6 * 4.0)
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 6 * 7168 * 2.0 + 8192 * 6 * 4.0

    def test_m_e(self):
        assert self.kernel.M_e == 8192 * 6 // 384


# ── Identity kernels (Spawn / Concat) ──────────────────────────────────


class TestSpawn(TestKernelBase):
    __test__ = True
    kernel = Spawn(world=4)
    expected_flops = 0.0
    expected_input_bytes = 0.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 0.0
    expected_transferred_bytes = 0.0

    def test_world(self):
        assert self.kernel.world == 4

    def test_requires_placement_false(self):
        assert self.kernel._requires_placement is False


class TestConcat(TestKernelBase):
    __test__ = True
    kernel = Concat()
    expected_flops = 0.0
    expected_input_bytes = 0.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 0.0
    expected_transferred_bytes = 0.0

    def test_requires_placement_false(self):
        assert self.kernel._requires_placement is False


# ── Backward StridedGemm kernels ───────────────────────────────────────


class TestStridedGemmDX(TestKernelBase):
    __test__ = True
    kernel = backward.StridedGemmDX(M=32, N=64, K=128, w_dtype="bf16",
                                     a_dtype="bf16", out_dtype="bf16",
                                     in_elems=32 * 128, out_elems=32 * 64)
    expected_flops = 2.0 * 32 * 64 * 128
    expected_input_bytes = 32 * 64 * 2.0       # out_elems * sizeof(out_dtype)
    expected_weight_bytes = 128 * 64 * 2.0     # K*N * sizeof(w_dtype)
    expected_output_bytes = 32 * 128 * 2.0     # in_elems * sizeof(a_dtype)

    def test_default_elems(self):
        k = backward.StridedGemmDX(M=32, N=64, K=128, w_dtype="bf16")
        assert k._in_elems == 32 * 128
        assert k._out_elems == 32 * 64


class TestStridedGemmDW(TestKernelBase):
    __test__ = True
    kernel = backward.StridedGemmDW(M=32, N=64, K=128, w_dtype="bf16",
                                     a_dtype="bf16", out_dtype="bf16",
                                     in_elems=32 * 128, out_elems=32 * 64)
    expected_flops = 2.0 * 32 * 64 * 128
    expected_input_bytes = 32 * 64 * 2.0 + 32 * 128 * 2.0  # out_elems*out + in_elems*a
    expected_weight_bytes = 0.0
    expected_output_bytes = 128 * 64 * 4.0     # K*N * sizeof(grad_dtype=fp32)

    def test_default_elems(self):
        k = backward.StridedGemmDW(M=32, N=64, K=128, w_dtype="bf16")
        assert k._in_elems == 32 * 128
        assert k._out_elems == 32 * 64
