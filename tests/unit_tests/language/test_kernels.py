# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Unit tests for rooflang.language.kernels (Kernel base + all subclasses)."""

from fractions import Fraction

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.forward import (
    ElementwiseOp, Embedding, Gemm, Nop, ReadInput, RMSNorm, LayerNorm,
    Attn, DpskV4SparseAttn, Glm52SparseAttn, Slice, Sampling, TokenDispatch,
    TokenCombine,
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

    def test_fractional_weight_reads_do_not_change_resident_weights(self):
        k = Kernel(weights={"w": Tensor("bf16", (8,))})
        k.weight_read_fraction = Fraction(3, 8)

        assert k.weight_bytes == 16.0
        assert k.loaded_weight_bytes == 6.0
        assert k.transferred_bytes == 6.0


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

    def test_optional_metrics_and_fractional_reads_included(self):
        class MixedKernel(Kernel):
            @property
            def flops(self):
                return 12.0

            @property
            def flops_by_dtype(self):
                return {"fp8": 12.0}

            def input_read_fraction(self, port):
                assert port == "x"
                return 0.5

        k = MixedKernel(
            inputs={"x": Tensor("bf16", (4,))},
            weights={"w": Tensor("bf16", (4,))},
            outputs={"y": Tensor("bf16", (4,))},
        )
        k.weight_read_fraction = Fraction(1, 2)

        result = k.to_dict()

        assert result["flops_by_dtype"] == {"fp8": 12.0}
        assert result["input_tensor_bytes"] == 8.0
        assert result["loaded_weight_bytes"] == 4.0
        assert result["weight_read_fraction"] == 0.5


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


class TestDpskV4SparseAttn(TestKernelBase):
    __test__ = True
    kernel = DpskV4SparseAttn(
        B=2, H=8, H_kv=8, S_q=256, k_sel=64, S_kv=64, Hd=64)
    expected_flops = 4.0 * 2 * 8 * 256 * 64 * 64
    expected_input_bytes = (2 * 8 * 256 * 64 + 2 * 2 * 8 * 64 * 64) * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 2 * 8 * 256 * 64 * 2.0

    def test_fused_indexer_and_sparse_kv_read(self):
        kernel = DpskV4SparseAttn(
            B=2, H=8, H_kv=1, S_q=1, k_sel=12, S_kv=100, Hd=64,
            kv_factor=1, indexer_s_kv=25, indexer_h=4, indexer_hd=8)
        kernel.inputs = {
            "q": Tensor("bf16", (2, 1, 8 * 64)),
            "kv": Tensor("bf16", (2, 100, 64)),
            "index_kv": Tensor("fp4", (2, 25, 8)),
        }
        kernel.outputs = {"y": Tensor("bf16", (2, 1, 8 * 64))}

        q_bytes = 2 * 1 * 8 * 64 * 2
        full_kv_bytes = 2 * 100 * 64 * 2
        selected_kv_bytes = 2 * 12 * 64 * 2
        index_bytes = 2 * 25 * 8 * 0.5
        attention_flops = 4 * 2 * 8 * 1 * 12 * 64
        indexer_flops = 2 * 2 * 1 * 4 * 25 * 8 + 3 * 2 * 1 * 4 * 25

        assert kernel.attention_flops == attention_flops
        assert kernel.indexer_flops == indexer_flops
        assert kernel.flops == attention_flops + indexer_flops
        assert kernel.flops_by_dtype == {
            "bf16": attention_flops,
            "fp4": indexer_flops,
        }
        assert kernel.input_read_fraction("q") == 1
        assert kernel.input_read_fraction("kv") == Fraction(3, 25)
        assert kernel.input_read_fraction("index_kv") == 1
        assert kernel.input_tensor_bytes == \
            q_bytes + full_kv_bytes + index_bytes
        assert kernel.input_bytes == \
            q_bytes + selected_kv_bytes + index_bytes

        graph = ComputeGraph()
        graph.add_kernel(kernel)
        graph.validate()

    def test_input_tensor_shape_is_checked_independently_of_sparse_read(self):
        kernel = DpskV4SparseAttn(
            B=1, H=1, H_kv=1, S_q=1, k_sel=2, S_kv=8, Hd=4,
            kv_factor=1)
        kernel.inputs = {
            "q": Tensor("bf16", (1, 1, 4)),
            "kv": Tensor("bf16", (1, 7, 4)),
        }
        kernel.outputs = {"y": Tensor("bf16", (1, 1, 4))}
        graph = ComputeGraph()
        graph.add_kernel(kernel)

        with pytest.raises(ValueError, match="input_tensor_bytes"):
            graph.validate()

    def test_compute_q_kv_and_output_dtypes_are_independent(self):
        kernel = DpskV4SparseAttn(
            B=1, H=2, H_kv=1, S_q=3, k_sel=2, S_kv=5, Hd=4,
            dtype="fp8", kv_factor=1,
            q_dtype="bf16", kv_dtype="fp8", out_dtype="bf16")
        kernel.inputs = {
            "q": Tensor("bf16", (1, 3, 8)),
            "kv": Tensor("fp8", (1, 5, 4)),
        }
        kernel.outputs = {"y": Tensor("bf16", (1, 3, 8))}

        assert kernel.dtype_ == "fp8"
        assert kernel.q_dtype == "bf16"
        assert kernel.kv_dtype == "fp8"
        assert kernel.out_dtype == "bf16"
        assert kernel.flops_by_dtype == {"fp8": kernel.flops}
        assert kernel.input_tensor_bytes == 3 * 8 * 2.0 + 5 * 4 * 1.0
        assert kernel.input_bytes == 3 * 8 * 2.0 + 5 * 4 * 1.0
        assert kernel.output_bytes == 3 * 8 * 2.0

        graph = ComputeGraph()
        graph.add_kernel(kernel)
        graph.validate()

    def test_causal_only_halves_context_dependent_sparse_work(self):
        kernel = DpskV4SparseAttn(
            B=1, H=1, H_kv=1, S_q=8, k_sel=4, S_kv=6, Hd=1,
            kv_factor=1, indexer_s_kv=4, indexer_h=1, indexer_hd=1,
            causal=True, causal_k_sel=2)
        attention_flops = 4.0 * 1 * 1 * 8 * (4 - 0.5 * 2) * 1
        indexer_flops = (2.0 * 1 * 8 * 1 * 4 * 1
                         + 3.0 * 1 * 8 * 1 * 4) * 0.5
        assert kernel.flops == attention_flops + indexer_flops


class TestGlm52SparseAttn:
    def test_full_indexer_flops_bytes_and_graph_validation(self):
        kernel = Glm52SparseAttn(
            B=2, H=4, S_q=3, k_sel=2, S_kv=5,
            qk_head_dim=6, v_head_dim=8, kv_cache_dim=10,
            kv_lora_rank=4, qk_nope_head_dim=2,
            indexer_mode="full", indexer_s_kv=5,
            indexer_h=2, indexer_hd=4)
        kernel.inputs = {
            "q": Tensor("bf16", (2, 3, 4 * 6)),
            "kv": Tensor("fp8", (2, 5, 10)),
            "index_q": Tensor("bf16", (2, 3, 2 * 4)),
            "index_kv": Tensor("fp8", (2, 5, 4)),
            "index_weights": Tensor("fp32", (2, 3, 2)),
        }
        kernel.weights = {
            "kv_b": Tensor("fp8", (4, 4 * (2 + 8))),
            "kv_b_scale": Tensor("ue8m0", (1,)),
        }
        kernel.outputs = {"y": Tensor("bf16", (2, 3, 4 * 8))}

        attention = 2 * 2 * 4 * (3 * 2) * (10 + 4)
        transform = 2 * 2 * 3 * 4 * (4 * (2 + 8))
        indexer_score = 2 * 2 * 2 * (3 * 5) * 4
        indexer_reduce = 3 * 2 * 2 * (3 * 5)
        assert kernel.attention_flops == attention
        assert kernel.kv_transform_flops == transform
        assert kernel.indexer_score_flops == indexer_score
        assert kernel.indexer_reduce_flops == indexer_reduce
        assert kernel.indexer_flops == indexer_score + indexer_reduce
        assert kernel.flops_by_dtype == {
            "bf16": attention,
            "fp8": transform + indexer_score,
            "fp32": indexer_reduce,
        }
        assert kernel.input_read_fraction("kv") == 1

        graph = ComputeGraph()
        graph.add_kernel(kernel)
        graph.validate()

    def test_shared_indexer_has_no_indexer_cost_or_bytes(self):
        kernel = Glm52SparseAttn(
            B=1, H=4, S_q=1, k_sel=2, S_kv=8,
            qk_head_dim=6, v_head_dim=8, kv_cache_dim=10,
            indexer_mode="shared")
        kernel.inputs = {
            "q": Tensor("bf16", (1, 1, 4 * 6)),
            "kv": Tensor("fp8", (1, 8, 10)),
        }
        kernel.outputs = {"y": Tensor("bf16", (1, 1, 4 * 8))}

        assert kernel.indexer_flops == 0
        assert kernel.input_tensor_bytes == 4 * 6 * 2 + 8 * 10
        assert kernel.input_bytes == 4 * 6 * 2 + 2 * 10
        assert kernel.input_read_fraction("kv") == Fraction(1, 4)

        graph = ComputeGraph()
        graph.add_kernel(kernel)
        graph.validate()

    def test_causal_topk_pair_counts_are_exact(self):
        kernel = Glm52SparseAttn(
            B=1, H=1, S_q=8, k_sel=4, S_kv=8,
            qk_head_dim=2, v_head_dim=3, kv_cache_dim=5,
            kv_lora_rank=4,
            indexer_mode="full", indexer_s_kv=8,
            indexer_h=1, indexer_hd=2, causal=True)
        assert kernel.selected_pairs == 26
        assert kernel.indexer_pairs == 36
        assert kernel.attention_flops == 2 * 26 * (5 + 4)
        assert kernel.indexer_flops == 2 * 36 * 2 + 3 * 36

    def test_indexer_mode_validation(self):
        kwargs = dict(
            B=1, H=1, S_q=1, k_sel=1, S_kv=1,
            qk_head_dim=2, v_head_dim=2, kv_cache_dim=3)
        with pytest.raises(ValueError, match="full indexer dimensions"):
            Glm52SparseAttn(**kwargs, indexer_mode="full")
        with pytest.raises(ValueError, match="shared indexer"):
            Glm52SparseAttn(
                **kwargs, indexer_mode="shared",
                indexer_s_kv=1, indexer_h=1, indexer_hd=1)


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


class TestBwdPartialRMSNorm(TestKernelBase):
    __test__ = True
    kernel = backward.PartialRMSNorm(M=4, input_dim=10, norm_dim=6)
    expected_flops = 9.0 * 4 * 6
    expected_input_bytes = 2 * 4 * 10 * 2.0
    expected_weight_bytes = 6 * 2.0
    expected_output_bytes = 4 * 10 * 2.0 + 6 * 4.0

    def test_rejects_invalid_norm_dim(self):
        with pytest.raises(ValueError, match="norm_dim"):
            backward.PartialRMSNorm(M=4, input_dim=10, norm_dim=11)


class TestBwdLayerNorm(TestKernelBase):
    __test__ = True
    kernel = backward.LayerNorm(M=16, D=64)
    expected_flops = 11.0 * 16 * 64
    expected_input_bytes = 2 * 16 * 64 * 2.0
    expected_weight_bytes = 64 * 2.0
    expected_output_bytes = 16 * 64 * 2.0 + 2 * 64 * 4.0


class TestBwdAttnRes(TestKernelBase):
    __test__ = True
    kernel = backward.AttnRes(B=2, S=3, D=4, R=2)
    expected_flops = 18.0 * 6 * 3 * 4 + 9.0 * 6 * 3 + 3.0 * 4
    expected_input_bytes = (6 * 3 * 4 + 6 * 4) * 2.0
    expected_weight_bytes = 2 * 4 * 2.0
    expected_output_bytes = 6 * 3 * 4 * 2.0 + 2 * 4 * 4.0

    def test_uses_fp32_compute_with_bf16_storage(self):
        assert self.kernel.dtype_ == "fp32"
        assert self.kernel.storage_dtype == "bf16"


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

    def test_flops_causal_asymmetric_not_halved(self):
        k = backward.Attn(B=2, H=8, H_kv=8, S_q=128, S_kv=256, Hd=64,
                          causal=True)
        assert k.flops == 10.0 * 2 * 8 * 128 * 256 * 64

    def test_explicit_triangular_factor_survives_asymmetric_shard(self):
        k = backward.Attn(B=2, H=8, H_kv=8, S_q=128, S_kv=256, Hd=64,
                          causal=True, triangular=True)
        assert k.flops == 10.0 * 2 * 8 * 128 * 256 * 64 * 0.5


class TestBwdKimiK3MlaAttn(TestKernelBase):
    __test__ = True
    kernel = backward.KimiK3MlaAttn(
        B=2, H=4, S_q=3, S_kv=5,
        qk_head_dim=6, v_head_dim=8, kv_cache_dim=10,
        kv_lora_rank=4, qk_nope_head_dim=2)
    expected_flops = (
        2 * 4 * (3 * 5) * (6 * 10 + 4 * 4)
        + 4 * 2 * 3 * 4 * (4 * (2 + 8))
    )
    expected_input_bytes = (
        2 * 3 * 4 * 6 * 2.0
        + 2 * 3 * 4 * 8 * 2.0
        + 2 * 5 * 10 * 1.0
    )
    expected_weight_bytes = 4 * (4 * (2 + 8)) * 2.0
    expected_output_bytes = (
        2 * 3 * 4 * 6 * 2.0
        + 2 * 5 * 10 * 1.0
        + 4 * (4 * (2 + 8)) * 4.0
    )

    def test_flop_components_and_dtype_merge(self):
        attention = 2 * 4 * (3 * 5) * (6 * 10 + 4 * 4)
        transform = 4 * 2 * 3 * 4 * (4 * (2 + 8))
        assert self.kernel.attention_flops == attention
        assert self.kernel.kv_transform_flops == transform
        assert self.kernel.flops_by_dtype == {
            "bf16": attention + transform,
        }

    def test_causal_dense_pairs(self):
        kernel = backward.KimiK3MlaAttn(
            B=1, H=2, S_q=8, S_kv=8,
            qk_head_dim=6, v_head_dim=8, kv_cache_dim=10,
            kv_lora_rank=4, qk_nope_head_dim=2, causal=True)
        assert kernel.selected_pairs == 36


class TestBwdKimiK3DeltaAttn(TestKernelBase):
    __test__ = True
    kernel = backward.KimiK3DeltaAttn(
        B=2, H=3, S=8, K=4, V=5, mode="chunk",
        chunk_size=4, conv_size=4)
    forward_attention_per_head = (
        6 * 8 * 4 * 5 + 3 * 8 * 4 * 4 + 8 * 4**2)
    forward_preprocessing = (
        3 * (2 * 3 * 8) * 4 * (2 * 4 + 4)
        + 2 * (2 * 3 * 8) * (3 * 4 + 1)
        + (2 * 3 * 8) * (5 * 4 + 3)
    )
    expected_flops = (
        3.0 * 2 * 3 * forward_attention_per_head
        + 2.0 * forward_preprocessing
    )
    expected_input_bytes = (
        (2 * 3 * 8) * (2 * 4 + 3 * 5) * 2.0
        + (2 * 3 * 8) * 4.0
    )
    expected_weight_bytes = (
        3 * (2 * 4 + 5) * (4 + 1) * 2.0
        + 3 * (5 + 1) * 4.0
    )
    expected_output_bytes = (
        (2 * 3 * 8) * (2 * 4 + 2 * 5) * 2.0
        + (2 * 3 * 8) * 4.0
        + (3 * (2 * 4 + 5) * (4 + 1) + 3 * (5 + 1)) * 4.0
    )

    def test_recurrent_includes_state_gradients(self):
        kernel = backward.KimiK3DeltaAttn(
            B=1, H=2, S=1, K=3, V=4, mode="recurrent",
            chunk_size=4, conv_size=2)
        state_elements = 1 * 2 * 3 * 4
        conv_state_elements = 1 * 2 * (2 * 3 + 4) * 2
        state_bytes = (state_elements + conv_state_elements) * 2.0
        base_input = 2 * (2 * 3 + 3 * 4) * 2.0 + 2 * 4.0
        base_output = (
            2 * (2 * 3 + 2 * 4) * 2.0
            + 2 * 4.0
            + (2 * (2 * 3 + 4) * 3 + 2 * (4 + 1)) * 4.0
        )
        assert kernel.input_bytes == base_input + state_bytes
        assert kernel.output_bytes == base_output + state_bytes

    def test_rejects_invalid_mode(self):
        with pytest.raises(ValueError, match="unsupported KDA mode"):
            backward.KimiK3DeltaAttn(
                B=1, H=1, S=1, K=4, V=4, mode="invalid")


class TestBwdKimiK3DeltaAttnCpSummary(TestKernelBase):
    __test__ = True
    kernel = backward.KimiK3DeltaAttnCpSummary(
        B=2, H=3, S=10, K=4, V=5, rank=1, world=3,
        chunk_size=4, conv_size=4)
    forward_bf16_per_head = (
        4 * 10 * 4 * 5
        + 2 * 3 * 4 * 5
        + 10 * 5
        + 2 * 10 * 4 * 4
        + 3 * 4 * 4
    )
    expected_flops = (
        2.0 * 2 * 3 * forward_bf16_per_head
        + 4.0 * 2 * 3 * 3 * 4**3
    )
    expected_input_bytes = (
        2 * 3 * 4 * (4 + 5) * 4.0
        + 2 * 3 * 3 * 4 * (4 - 1) * 2.0
    )
    expected_weight_bytes = 0.0
    expected_output_bytes = 0.0

    def test_flops_by_dtype_and_last_rank_elision(self):
        assert self.kernel.flops_by_dtype == {
            "bf16": 2.0 * 2 * 3 * self.forward_bf16_per_head,
            "fp32": 4.0 * 2 * 3 * 3 * 4**3,
        }
        last = backward.KimiK3DeltaAttnCpSummary(
            B=2, H=3, S=10, K=4, V=5, rank=2, world=3,
            chunk_size=4, conv_size=4)
        assert last.flops == 0
        assert last.input_bytes == 0


class TestBwdKimiK3DeltaAttnCpMerge(TestKernelBase):
    __test__ = True
    kernel = backward.KimiK3DeltaAttnCpMerge(
        B=2, H=3, K=4, V=5, rank=2, world=4)
    expected_flops = 2.0 * 2 * 3 * 2 * (2 * 4 * 4 * 5 + 4 * 5)
    expected_input_bytes = (
        2 * 3 * 2 * 4 * (4 + 5) * 4.0
        + 2 * 3 * 4 * 5 * 2.0
    )
    expected_weight_bytes = 0.0
    expected_output_bytes = 2 * 3 * 2 * 4 * (4 + 5) * 4.0


class TestBwdKimiK3DeltaAttnStateStore(TestKernelBase):
    __test__ = True
    kernel = backward.KimiK3DeltaAttnStateStore(
        B=2, H=3, S=8, K=4, V=5, conv_size=4)
    expected_flops = 0.0
    expected_input_bytes = 0.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 0.0
    expected_transferred_bytes = 0.0

    def test_requires_no_placement(self):
        assert self.kernel._requires_placement is False


class TestBwdDpskV4SparseAttn(TestKernelBase):
    __test__ = True
    kernel = backward.DpskV4SparseAttn(
        B=2, H=8, H_kv=8, S_q=256, k_sel=64, Hd=64)
    expected_flops = 10.0 * 2 * 8 * 256 * 64 * 64
    expected_input_bytes = (2 * 2 * 8 * 256 * 64 + 2 * 2 * 8 * 256 * 64 * 64) * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = (2 * 8 * 256 * 64 + 2 * 2 * 8 * 256 * 64 * 64) * 2.0

    def test_compute_q_kv_and_output_dtypes_are_independent(self):
        k = backward.DpskV4SparseAttn(
            B=1, H=2, H_kv=1, S_q=3, k_sel=2, Hd=4,
            dtype="fp8", kv_factor=1,
            q_dtype="bf16", kv_dtype="fp8", out_dtype="bf16")

        q_elements = 1 * 2 * 3 * 4
        kv_elements = 1 * 1 * 3 * 2 * 4
        assert k.dtype_ == "fp8"
        assert k.q_dtype == "bf16"
        assert k.kv_dtype == "fp8"
        assert k.out_dtype == "bf16"
        assert k.flops_by_dtype == {"fp8": k.flops}
        assert k.input_bytes == (
            q_elements * 2.0 + q_elements * 2.0 + kv_elements * 1.0)
        assert k.output_bytes == (
            q_elements * 2.0 + kv_elements * 1.0)

    def test_fused_indexer_respects_independent_dtypes(self):
        k = backward.DpskV4SparseAttn(
            B=2, H=8, H_kv=1, S_q=4, k_sel=6, Hd=8,
            dtype="fp8", kv_factor=1,
            indexer_s_kv=10, indexer_h=3, indexer_hd=4,
            indexer_dtype="fp4", q_dtype="bf16", kv_dtype="fp8",
            out_dtype="bf16")

        assert k.indexer_input_bytes == (
            (2 * 4 * 3 * 4 + 2 * 10 * 4) * 0.5
            + 2 * 4 * 3 * 2.0
            + 2 * 4 * 10 * 2.0
        )
        assert k.indexer_output_bytes == (
            2 * 4 * 3 * 4 * 2.0
            + 2 * 10 * 4 * 1.0
            + 2 * 4 * 3 * 2.0
        )
        assert k.flops_by_dtype == {
            "fp8": k.attention_flops,
            "fp4": k.indexer_flops,
        }

    def test_causal_only_halves_context_dependent_sparse_work(self):
        k = backward.DpskV4SparseAttn(
            B=2, H=8, H_kv=1, S_q=256, k_sel=64, Hd=64,
            kv_factor=1, causal=True, causal_k_sel=32)
        effective_k_sel = 48
        assert k.effective_k_sel == effective_k_sel
        assert k.flops == 10.0 * 2 * 8 * 256 * effective_k_sel * 64
        assert k.input_bytes == (
            2 * 2 * 8 * 256 * 64
            + 1 * 2 * 1 * 256 * effective_k_sel * 64) * 2.0
        assert k.output_bytes == (
            2 * 8 * 256 * 64
            + 1 * 2 * 1 * 256 * effective_k_sel * 64) * 2.0

    def test_causal_fixed_topk_is_not_halved(self):
        k = backward.DpskV4SparseAttn(
            B=2, H=8, H_kv=1, S_q=256, k_sel=64, Hd=64,
            kv_factor=1, causal=True)
        assert k.effective_k_sel == 64
        assert k.flops == 10.0 * 2 * 8 * 256 * 64 * 64

    def test_fused_indexer_backward_flops_and_bytes(self):
        k = backward.DpskV4SparseAttn(
            B=2, H=8, H_kv=1, S_q=4, k_sel=6, Hd=8,
            kv_factor=1, indexer_s_kv=10, indexer_h=3, indexer_hd=4,
            indexer_dtype="fp4", causal=True, causal_k_sel=2)

        main_flops = 10.0 * 2 * 8 * 4 * 5 * 8
        score_backward = 6.0 * 2 * 4 * 3 * 10 * 4 * 0.5
        reduce_backward = 5.0 * 2 * 4 * 3 * 10 * 0.5
        assert k.indexer_flops == score_backward + reduce_backward
        assert k.flops == main_flops + score_backward + reduce_backward

        indexer_input_bytes = (
            (2 * 4 * 3 * 4 + 2 * 10 * 4) * 0.5
            + (2 * 4 * 3 + 2 * 4 * 10) * 2.0
        )
        indexer_output_bytes = (
            2 * 4 * 3 * 4 + 2 * 10 * 4 + 2 * 4 * 3
        ) * 2.0
        assert k.indexer_input_bytes == indexer_input_bytes
        assert k.indexer_output_bytes == indexer_output_bytes
        assert k.input_bytes == (
            (2 * 2 * 8 * 4 * 8 + 1 * 2 * 1 * 4 * 5 * 8) * 2.0
            + indexer_input_bytes)
        assert k.output_bytes == (
            (2 * 8 * 4 * 8 + 1 * 2 * 1 * 4 * 5 * 8) * 2.0
            + indexer_output_bytes)

    def test_decode_indexer_backward_is_not_causal_halved(self):
        kwargs = dict(
            B=2, H=8, H_kv=1, S_q=1, k_sel=6, Hd=8,
            indexer_s_kv=10, indexer_h=3, indexer_hd=4)
        prefill = backward.DpskV4SparseAttn(**kwargs, causal=True)
        decode = backward.DpskV4SparseAttn(**kwargs, causal=False)
        assert decode.indexer_flops == 2 * prefill.indexer_flops


class TestBwdGlm52SparseAttn:
    def test_full_indexer_flops_bytes_and_graph_validation(self):
        kernel = backward.Glm52SparseAttn(
            B=2, H=4, S_q=3, k_sel=2, S_kv=5,
            qk_head_dim=6, v_head_dim=8, kv_cache_dim=10,
            kv_lora_rank=4, qk_nope_head_dim=2,
            indexer_mode="full", indexer_s_kv=5,
            indexer_h=2, indexer_hd=4)
        kernel.inputs = {
            "q": Tensor("bf16", (2, 3, 4 * 6)),
            "dy": Tensor("bf16", (2, 3, 4 * 8)),
            "kv": Tensor("fp8", (2, 5, 10)),
            "index_q": Tensor("bf16", (2, 3, 2 * 4)),
            "index_kv": Tensor("fp8", (2, 5, 4)),
            "index_weights": Tensor("fp32", (2, 3, 2)),
            "d_index_scores": Tensor("fp32", (2, 3, 5)),
        }
        kernel.weights = {
            "kv_b": Tensor("fp8", (4, 4 * (2 + 8))),
            "kv_b_scale": Tensor("ue8m0", (1,)),
        }
        kernel.outputs = {
            "dq": Tensor("bf16", (2, 3, 4 * 6)),
            "dkv": Tensor("fp8", (2, 5, 10)),
            "d_kv_b": Tensor("fp32", (4, 4 * (2 + 8))),
            "d_index_q": Tensor("bf16", (2, 3, 2 * 4)),
            "d_index_kv": Tensor("fp8", (2, 5, 4)),
            "d_index_weights": Tensor("fp32", (2, 3, 2)),
        }

        attention = 2 * 4 * (3 * 2) * (6 * 10 + 4 * 4)
        transform = 4 * 2 * 3 * 4 * (4 * (2 + 8))
        indexer_score = 6 * 2 * 2 * (3 * 5) * 4
        indexer_reduce = 5 * 2 * 2 * (3 * 5)
        assert kernel.attention_flops == attention
        assert kernel.kv_transform_flops == transform
        assert kernel.indexer_score_flops == indexer_score
        assert kernel.indexer_reduce_flops == indexer_reduce
        assert kernel.indexer_flops == indexer_score + indexer_reduce
        assert kernel.flops_by_dtype == {
            "bf16": attention,
            "fp8": transform + indexer_score,
            "fp32": indexer_reduce,
        }

        graph = ComputeGraph()
        graph.add_kernel(kernel)
        graph.validate()

    def test_shared_causal_has_no_indexer_backward(self):
        kernel = backward.Glm52SparseAttn(
            B=1, H=1, S_q=8, k_sel=4, S_kv=8,
            qk_head_dim=2, v_head_dim=3, kv_cache_dim=5,
            kv_lora_rank=4,
            indexer_mode="shared", causal=True)
        assert kernel.selected_pairs == 26
        assert kernel.indexer_pairs == 0
        assert kernel.indexer_flops == 0
        assert kernel.attention_flops == 26 * (6 * 5 + 4 * 4)


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


class TestBwdElementwiseOpSigmoidMul(TestKernelBase):
    __test__ = True
    kernel = backward.ElementwiseOp(
        M=8192, D=7168, dtype="bf16", op="sigmoid_mul")
    expected_flops = 9.0 * 8192 * 7168
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

    def test_fractional_m_e(self):
        assert TokenDispatch(32, 7168, 384, 6).M_e == Fraction(1, 2)

    def test_sigmoid_scoring(self):
        kernel = TokenDispatch(
            M=32, D=64, N_experts=8, topk=2,
            scoring_func="sigmoid")
        assert kernel.scoring_func == "sigmoid"
        assert kernel.flops == 3.0 * 32 * 8


class TestTokenCombine(TestKernelBase):
    __test__ = True
    kernel = TokenCombine(M=8192, D=7168, N_experts=384, topk=6)
    expected_flops = 2.0 * 8192 * 6 * 7168
    expected_input_bytes = 8192 * 6 * 7168 * 2.0
    expected_weight_bytes = 0.0
    expected_output_bytes = 8192 * 7168 * 2.0

    def test_m_e(self):
        assert self.kernel.M_e == 8192 * 6 // 384

    def test_fractional_m_e(self):
        assert TokenCombine(32, 7168, 384, 6).M_e == Fraction(1, 2)


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

    def test_fractional_m_e(self):
        assert backward.TokenDispatch(
            32, 7168, 384, 6).M_e == Fraction(1, 2)

    def test_sigmoid_scoring(self):
        kernel = backward.TokenDispatch(
            M=32, D=64, N_experts=8, topk=2,
            scoring_func="sigmoid")
        assert kernel.scoring_func == "sigmoid"
        assert kernel.flops == 32 * 2 * 64 + 3.0 * 32 * 8


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

    def test_fractional_m_e(self):
        assert backward.TokenCombine(
            32, 7168, 384, 6).M_e == Fraction(1, 2)


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
