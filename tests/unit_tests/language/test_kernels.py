"""Unit tests for rooflang.language.kernels (Kernel base + subclasses)."""

import pytest

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.forward import Gemm, RMSNorm, Attn, RoPE
from rooflang.language.kernels.backward import GemmDX, GemmDW
from rooflang.language.kernels.comm import (
    AllReduce, ReduceScatter, AllGather, AllToAll, Broadcast, Send, Recv,
)
from rooflang.language.kernels.optimizer import AdamWStep
from rooflang.language.kernels.identity import Move
from rooflang.language.tensor import Tensor
from rooflang.language.utils import dtype_bytes


class TestKernelBase:
    def test_defaults(self):
        k = Kernel()
        assert k.inputs == {}
        assert k.weights == {}
        assert k.outputs == {}
        assert k.has_side_effect is False
        assert k.flops == 0.0

    def test_input_bytes_from_tensors(self):
        k = Kernel(inputs={"x": Tensor("bf16", (4, 4))})
        assert k.input_bytes == 16 * 2.0

    def test_weight_bytes_from_tensors(self):
        k = Kernel(weights={"w": Tensor("fp32", (2, 2))})
        assert k.weight_bytes == 4 * 4.0

    def test_output_bytes_from_tensors(self):
        k = Kernel(outputs={"y": Tensor("bf16", (8,))})
        assert k.output_bytes == 8 * 2.0

    def test_transferred_bytes_sum(self):
        k = Kernel(
            inputs={"x": Tensor("bf16", (4,))},
            weights={"w": Tensor("bf16", (4,))},
            outputs={"y": Tensor("bf16", (4,))},
        )
        assert k.transferred_bytes == 3 * 4 * 2.0

    def test_to_dict_keys(self):
        k = Kernel(inputs={"x": Tensor("bf16", (4,))})
        d = k.to_dict()
        assert "flops" in d
        assert "input_bytes" in d
        assert "inputs" in d

    def test_side_effect_in_to_dict(self):
        k = Kernel(has_side_effect=True)
        assert k.to_dict()["has_side_effect"] is True

    def test_no_side_effect_not_in_to_dict(self):
        k = Kernel()
        assert "has_side_effect" not in k.to_dict()


class TestGemm:
    def test_flops(self):
        g = Gemm(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
        assert g.flops == 2.0 * 32 * 64 * 128

    def test_input_bytes(self):
        g = Gemm(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
        assert g.input_bytes == 32 * 128 * 2.0

    def test_weight_bytes_no_scale(self):
        g = Gemm(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
        assert g.weight_bytes == 128 * 64 * 2.0

    def test_output_bytes(self):
        g = Gemm(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
        assert g.output_bytes == 32 * 64 * 2.0


class TestAttn:
    def test_flops_non_causal(self):
        a = Attn(B=2, H=8, H_kv=8, S_q=256, S_kv=256, Hd=64)
        assert a.flops == 4.0 * 2 * 8 * 256 * 256 * 64

    def test_flops_causal_halved(self):
        a = Attn(B=2, H=8, H_kv=8, S_q=256, S_kv=256, Hd=64, causal=True)
        assert a.flops == 4.0 * 2 * 8 * 256 * 256 * 64 * 0.5

    def test_flops_causal_asymmetric_not_halved(self):
        a = Attn(B=2, H=8, H_kv=8, S_q=128, S_kv=256, Hd=64, causal=True)
        assert a.flops == 4.0 * 2 * 8 * 128 * 256 * 64


class TestCommKernels:
    def test_allreduce_flops(self):
        ar = AllReduce(bytes_per_rank=1024.0, world=4, dtype="bf16")
        n_elements = 1024.0 / 2.0
        assert ar.flops == (4 - 1) / 4 * n_elements

    def test_allreduce_transferred(self):
        ar = AllReduce(bytes_per_rank=1024.0, world=4)
        assert ar.transferred_bytes == 2.0 * (3 / 4) * 1024.0

    def test_allgather_zero_flops(self):
        ag = AllGather(bytes_per_rank=1024.0, world=4)
        assert ag.flops == 0.0

    def test_reducescatter_output(self):
        rs = ReduceScatter(bytes_per_rank=1024.0, world=4)
        assert rs.output_bytes == 1024.0 / 4

    def test_send_recv(self):
        s = Send(bytes_total=512.0)
        r = Recv(bytes_total=512.0)
        assert s.output_bytes == 0.0
        assert r.input_bytes == 0.0
        assert s.transferred_bytes == 512.0
        assert r.transferred_bytes == 512.0


class TestAdamWStep:
    def test_flops(self):
        opt = AdamWStep(n_param=1000)
        assert opt.flops == 13.0 * 1000

    def test_input_bytes_grad(self):
        opt = AdamWStep(n_param=1000, grad_dtype="bf16")
        assert opt.input_bytes == 1000 * 2.0

    def test_weight_bytes(self):
        opt = AdamWStep(n_param=1000)
        assert opt.weight_bytes == 1000 * (4.0 + 2 * 4.0)


class TestMove:
    def test_io(self):
        t = Tensor("bf16", (4, 4))
        m = Move(t, "nvme")
        assert "src" in m.inputs
        assert "dst" in m.outputs
        assert m.outputs["dst"].location == "nvme"
        assert m.has_side_effect is False
