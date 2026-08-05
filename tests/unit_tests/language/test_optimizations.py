"""Unit tests for rooflang.language.optimization (split + optimize_comms)."""

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import (
    AllGather, AllReduce, AllToAll, Broadcast, CommKernel, Gather, Reduce,
    ReduceScatter, Scatter, Send, Recv,
)
from rooflang.language.kernels.forward import (
    Embedding, Gemm, ReadInput, RMSNorm, Slice, SparseAttn, StridedGemm,
    TokenCombine, TokenDispatch,
)
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.tensor import Tensor
from rooflang.language.placement import Placement
from rooflang.language.hardware.component import Compute
from rooflang.language.optimization.comm import optimize_comms
from rooflang.language.optimization.split import (
    batch_split, column_split, head_split, row_split,
)
from rooflang.language.utils import gemm_scale_bytes


SHARD = Tensor("bf16", (4, 4))


def _build_chain(collector, distributor, n, same_devices=True):
    """Build n preds → collector → distributor → n succs with placement."""
    g = ComputeGraph()
    gpus = [Compute(name=f"gpu{i}") for i in range(n)]

    preds = [Kernel(outputs={"y": SHARD}) for _ in range(n)]
    succs = [Kernel(inputs={"a": SHARD}) for _ in range(n)]

    collector.inputs = {f"x{i}": SHARD for i in range(n)}
    collector.outputs = {"z": SHARD}
    distributor.inputs = {"z": SHARD}
    distributor.outputs = {f"y{i}": SHARD for i in range(n)}

    for k in preds + [collector, distributor] + succs:
        g.add_kernel(k)
    for i, p in enumerate(preds):
        g.add_data_edge(p, collector, {"y": f"x{i}"})
    g.add_data_edge(collector, distributor, {"z": "z"})
    for i, s in enumerate(succs):
        g.add_data_edge(distributor, s, {f"y{i}": "a"})

    placement = Placement()
    for i, p in enumerate(preds):
        placement.set_kernel_device(p, gpus[i])
    if same_devices:
        for i, s in enumerate(succs):
            placement.set_kernel_device(s, gpus[i])
    else:
        other_gpus = [Compute(name=f"other{i}") for i in range(n)]
        for i, s in enumerate(succs):
            placement.set_kernel_device(s, other_gpus[i])

    return g, placement, preds, succs


class TestFuseWorld4:
    def test_reduce_broadcast_to_allreduce(self):
        r = Reduce(total_bytes=32, world=4, dtype="bf16")
        b = Broadcast(total_bytes=32, world=4)
        g, p, preds, succs = _build_chain(r, b, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllReduce)

    def test_reduce_scatter_to_reducescatter(self):
        r = Reduce(total_bytes=32, world=4, dtype="bf16")
        s = Scatter(total_bytes=32, world=4, dim=1)
        g, p, preds, succs = _build_chain(r, s, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), ReduceScatter)

    def test_gather_broadcast_to_allgather(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        b = Broadcast(total_bytes=128, world=4)
        g, p, preds, succs = _build_chain(ga, b, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllGather)

    def test_gather_scatter_diff_dim_to_alltoall(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        s = Scatter(total_bytes=128, world=4, dim=1)
        g, p, preds, succs = _build_chain(ga, s, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllToAll)


class TestFuseWorld1Eliminated:
    def test_reduce_broadcast_world1_eliminated(self):
        r = Reduce(total_bytes=32, world=1, dtype="bf16")
        b = Broadcast(total_bytes=32, world=1)
        g, p, preds, succs = _build_chain(r, b, n=1)
        optimize_comms(g, p)
        assert g.kernels == frozenset(preds + succs)
        assert g._out_edges(preds[0])[0].dst is succs[0]

    def test_gather_scatter_same_dim_world1_eliminated(self):
        ga = Gather(total_bytes=32, world=1, dim=0)
        s = Scatter(total_bytes=32, world=1, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g, p)
        assert g.kernels == frozenset(preds + succs)

    def test_gather_broadcast_world1_eliminated(self):
        ga = Gather(total_bytes=32, world=1, dim=0)
        b = Broadcast(total_bytes=32, world=1)
        g, p, preds, succs = _build_chain(ga, b, n=1)
        optimize_comms(g, p)
        assert g.kernels == frozenset(preds + succs)


class TestBypass:
    def test_gather_scatter_same_dim_bypass(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        s = Scatter(total_bytes=128, world=4, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g, p)
        assert ga not in g.kernels
        assert s not in g.kernels
        assert g._out_edges(preds[0])[0].dst is succs[0]


class TestNoFusionDiffDevices:
    def test_diff_devices_no_fusion(self):
        r = Reduce(total_bytes=32, world=4, dtype="bf16")
        b = Broadcast(total_bytes=32, world=4)
        g, p, preds, succs = _build_chain(r, b, n=4, same_devices=False)
        optimize_comms(g, p)
        assert r in g.kernels
        assert b in g.kernels


class TestEliminateDead:
    @pytest.mark.parametrize("comm_cls,kwargs", [
        (Scatter, dict(total_bytes=128, world=4, dim=0)),
        (Broadcast, dict(total_bytes=32, world=4)),
        (Gather, dict(total_bytes=128, world=4, dim=0)),
        (Reduce, dict(total_bytes=32, world=4, dtype="bf16")),
        (AllReduce, dict(total_bytes=32, world=4, dtype="bf16")),
        (AllGather, dict(total_bytes=128, world=4)),
        (ReduceScatter, dict(total_bytes=32, world=4, dtype="bf16")),
        (AllToAll, dict(total_bytes=128, world=4)),
        (Send, dict(total_bytes=32)),
        (Recv, dict(total_bytes=32)),
    ])
    def test_single_edge_eliminated(self, comm_cls, kwargs):
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        pred = Kernel(outputs={"y": SHARD})
        succ = Kernel(inputs={"a": SHARD})
        comm = comm_cls(**kwargs)
        comm.inputs = {"x": SHARD}
        comm.outputs = {"y": SHARD}
        g.add_kernel(pred)
        g.add_kernel(comm)
        g.add_kernel(succ)
        g.add_data_edge(pred, comm, {"y": "x"})
        g.add_data_edge(comm, succ, {"y": "a"})
        p = Placement()
        p.set_kernel_device(pred, gpu)
        p.set_kernel_device(succ, gpu)
        optimize_comms(g, p)
        assert comm not in g.kernels
        assert g._out_edges(pred)[0].dst is succ

    def test_multi_edge_not_eliminated(self):
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        pred = Kernel(outputs={"y": SHARD})
        comm = Scatter(total_bytes=128, world=4, dim=0)
        comm.inputs = {"x": SHARD}
        comm.outputs = {"y0": SHARD, "y1": SHARD}
        succ0 = Kernel(inputs={"a": SHARD})
        succ1 = Kernel(inputs={"a": SHARD})
        g.add_kernel(pred)
        g.add_kernel(comm)
        g.add_kernel(succ0)
        g.add_kernel(succ1)
        g.add_data_edge(pred, comm, {"y": "x"})
        g.add_data_edge(comm, succ0, {"y0": "a"})
        g.add_data_edge(comm, succ1, {"y1": "a"})
        p = Placement()
        p.set_kernel_device(pred, gpu)
        p.set_kernel_device(succ0, gpu)
        p.set_kernel_device(succ1, gpu)
        optimize_comms(g, p)
        assert comm in g.kernels


# ── Guard clause tests (_fuse_pairs skips) ───────────────────────────


class TestFusePairsGuards:
    def test_collector_multi_out_edges_no_fuse(self):
        g = ComputeGraph()
        gpus = [Compute(name=f"gpu{i}") for i in range(2)]
        preds = [Kernel(outputs={"y": SHARD}) for _ in range(2)]
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        r.inputs = {"x0": SHARD, "x1": SHARD}
        r.outputs = {"z": SHARD, "z2": SHARD}
        b = Broadcast(total_bytes=128.0, world=2)
        b.inputs = {"z": SHARD}
        b.outputs = {"y0": SHARD, "y1": SHARD}
        extra = Kernel(inputs={"q": SHARD})
        succs = [Kernel(inputs={"a": SHARD}) for _ in range(2)]
        for k in preds + [r, b, extra] + succs:
            g.add_kernel(k)
        for i, p in enumerate(preds):
            g.add_data_edge(p, r, {"y": f"x{i}"})
        g.add_data_edge(r, b, {"z": "z"})
        g.add_data_edge(r, extra, {"z2": "q"})
        for i, s in enumerate(succs):
            g.add_data_edge(b, s, {f"y{i}": "a"})
        p = Placement()
        for i, pred in enumerate(preds):
            p.set_kernel_device(pred, gpus[i])
        for i, s in enumerate(succs):
            p.set_kernel_device(s, gpus[i])
        p.set_kernel_device(extra, gpus[0])
        optimize_comms(g, p)
        assert r in g.kernels
        assert b in g.kernels

    def test_succ_multi_in_edges_no_fuse(self):
        g = ComputeGraph()
        gpus = [Compute(name=f"gpu{i}") for i in range(2)]
        preds = [Kernel(outputs={"y": SHARD}) for _ in range(2)]
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        r.inputs = {"x0": SHARD, "x1": SHARD}
        r.outputs = {"z": SHARD}
        b = Broadcast(total_bytes=128.0, world=2)
        b.inputs = {"z": SHARD, "q": SHARD}
        b.outputs = {"y0": SHARD, "y1": SHARD}
        extra = Kernel(outputs={"q": SHARD})
        succs = [Kernel(inputs={"a": SHARD}) for _ in range(2)]
        for k in preds + [r, b, extra] + succs:
            g.add_kernel(k)
        for i, p in enumerate(preds):
            g.add_data_edge(p, r, {"y": f"x{i}"})
        g.add_data_edge(r, b, {"z": "z"})
        g.add_data_edge(extra, b, {"q": "q"})
        for i, s in enumerate(succs):
            g.add_data_edge(b, s, {f"y{i}": "a"})
        p = Placement()
        for i, pred in enumerate(preds):
            p.set_kernel_device(pred, gpus[i])
        for i, s in enumerate(succs):
            p.set_kernel_device(s, gpus[i])
        p.set_kernel_device(extra, gpus[0])
        optimize_comms(g, p)
        assert r in g.kernels
        assert b in g.kernels

    def test_world_mismatch_no_fuse(self):
        g = ComputeGraph()
        gpus = [Compute(name=f"gpu{i}") for i in range(2)]
        preds = [Kernel(outputs={"y": SHARD}) for _ in range(2)]
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        r.inputs = {"x0": SHARD, "x1": SHARD}
        r.outputs = {"z": SHARD}
        b = Broadcast(total_bytes=128.0, world=4)
        b.inputs = {"z": SHARD}
        b.outputs = {"y0": SHARD, "y1": SHARD}
        succs = [Kernel(inputs={"a": SHARD}) for _ in range(2)]
        for k in preds + [r, b] + succs:
            g.add_kernel(k)
        for i, p in enumerate(preds):
            g.add_data_edge(p, r, {"y": f"x{i}"})
        g.add_data_edge(r, b, {"z": "z"})
        for i, s in enumerate(succs):
            g.add_data_edge(b, s, {f"y{i}": "a"})
        p = Placement()
        for i, pred in enumerate(preds):
            p.set_kernel_device(pred, gpus[i])
        for i, s in enumerate(succs):
            p.set_kernel_device(s, gpus[i])
        optimize_comms(g, p)
        assert r in g.kernels
        assert b in g.kernels

    def test_pred_succ_length_mismatch_no_fuse(self):
        g = ComputeGraph()
        gpus = [Compute(name=f"gpu{i}") for i in range(3)]
        preds = [Kernel(outputs={"y": SHARD}) for _ in range(3)]
        r = Reduce(total_bytes=128.0, world=3, dtype="bf16")
        r.inputs = {f"x{i}": SHARD for i in range(3)}
        r.outputs = {"z": SHARD}
        b = Broadcast(total_bytes=128.0, world=3)
        b.inputs = {"z": SHARD}
        b.outputs = {"y0": SHARD, "y1": SHARD}
        succs = [Kernel(inputs={"a": SHARD}) for _ in range(2)]
        for k in preds + [r, b] + succs:
            g.add_kernel(k)
        for i, p in enumerate(preds):
            g.add_data_edge(p, r, {"y": f"x{i}"})
        g.add_data_edge(r, b, {"z": "z"})
        for i, s in enumerate(succs):
            g.add_data_edge(b, s, {f"y{i}": "a"})
        p = Placement()
        for i, pred in enumerate(preds):
            p.set_kernel_device(pred, gpus[i])
        for i, s in enumerate(succs):
            p.set_kernel_device(s, gpus[i])
        optimize_comms(g, p)
        assert r in g.kernels
        assert b in g.kernels


# ═══════════════════════════════════════════════════════════════════════
# Split function tests
# ═══════════════════════════════════════════════════════════════════════

N = 4  # split world size for tests


def _make_test_gemm(M=32, N_dim=128, K=64, w_dtype="bf16"):
    k = Gemm(M, N_dim, K, w_dtype, "bf16")
    k.inputs = {"x": Tensor("bf16", (M, K))}
    k.weights = {"w": Tensor(w_dtype, (K, N_dim))}
    sb = gemm_scale_bytes(N_dim, K, w_dtype)
    if sb > 0:
        k.weights["s"] = Tensor("ue8m0", (int(sb),))
    k.outputs = {"y": Tensor("bf16", (M, N_dim))}
    return k


def _make_test_strided_gemm(M=32, N_dim=256, K=64, w_dtype="fp8",
                             out_elems=None):
    out_elems = out_elems or M * 128
    k = StridedGemm(M, N_dim, K, w_dtype, "bf16", out_elems=out_elems)
    k.inputs = {"x": Tensor("bf16", (M, K))}
    k.weights = {"w": Tensor(w_dtype, (K, N_dim))}
    sb = gemm_scale_bytes(N_dim, K, w_dtype)
    if sb > 0:
        k.weights["s"] = Tensor("ue8m0", (int(sb),))
    k.outputs = {"y": Tensor("bf16", (M, 128))}
    return k


# ── column_split ───────────────────────────────────────────────────────


class TestColumnSplitGemm:
    def setup_method(self):
        self.kernel = _make_test_gemm()
        self.prev, self.copies, self.nxt = column_split(self.kernel, N)

    def test_prev_broadcast(self):
        assert "x" in self.prev
        assert isinstance(self.prev["x"], Broadcast)
        assert self.prev["x"].world == N
        assert self.prev["x"].inputs["x"].shape == (32, 64)
        assert self.prev["x"].outputs["o0"].shape == (32, 64)

    def test_copies_count_and_dims(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, Gemm)
            assert c.M == 32
            assert c.N == 32  # 128/4
            assert c.K == 64

    def test_copies_shapes(self):
        for c in self.copies:
            assert c.inputs["x"].shape == (32, 64)
            assert c.outputs["y"].shape == (32, 32)
            assert c.weights["w"].shape == (64, 32)

    def test_next_gather(self):
        assert "y" in self.nxt
        assert isinstance(self.nxt["y"], Gather)
        assert self.nxt["y"].world == N
        assert self.nxt["y"].inputs["i0"].shape == (32, 32)
        assert self.nxt["y"].outputs["y"].shape == (32, 128)

    def test_graph_validates(self):
        g = ComputeGraph()
        pred = Kernel(outputs={"out": Tensor("bf16", (32, 64))})
        succ = Kernel(inputs={"inp": Tensor("bf16", (32, 128))})
        k = _make_test_gemm()
        g.add_kernel(pred)
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(pred, k, {"out": "x"})
        g.add_data_edge(k, succ, {"y": "inp"})
        g.split_kernel(column_split, k, N)
        g.validate()


class TestColumnSplitStridedGemm:
    def setup_method(self):
        self.kernel = _make_test_strided_gemm()
        self.prev, self.copies, self.nxt = column_split(self.kernel, N)

    def test_copies_strided(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, StridedGemm)
            assert c.N == 64  # 256/4
            assert c._out_elems == 32 * 128 // N

    def test_shapes(self):
        for c in self.copies:
            assert c.inputs["x"].shape == (32, 64)
            assert c.outputs["y"].shape == (32, 32)  # 128/4


# ── row_split ──────────────────────────────────────────────────────────


class TestRowSplitGemm:
    def setup_method(self):
        self.kernel = _make_test_gemm()
        self.prev, self.copies, self.nxt = row_split(self.kernel, N)

    def test_prev_scatter(self):
        assert "x" in self.prev
        assert isinstance(self.prev["x"], Scatter)
        assert self.prev["x"].world == N
        assert self.prev["x"].inputs["x"].shape == (32, 64)
        assert self.prev["x"].outputs["o0"].shape == (32, 16)  # K/4

    def test_copies_count_and_dims(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, Gemm)
            assert c.M == 32
            assert c.N == 128
            assert c.K == 16  # 64/4

    def test_copies_shapes(self):
        for c in self.copies:
            assert c.inputs["x"].shape == (32, 16)
            assert c.outputs["y"].shape == (32, 128)
            assert c.weights["w"].shape == (16, 128)

    def test_next_reduce(self):
        assert "y" in self.nxt
        assert isinstance(self.nxt["y"], Reduce)
        assert self.nxt["y"].world == N
        assert self.nxt["y"].inputs["i0"].shape == (32, 128)
        assert self.nxt["y"].outputs["y"].shape == (32, 128)

    def test_graph_validates(self):
        g = ComputeGraph()
        k = _make_test_gemm()
        pred = Kernel(outputs={"out": Tensor("bf16", (32, 64))})
        succ = Kernel(inputs={"inp": Tensor("bf16", (32, 128))})
        g.add_kernel(pred)
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(pred, k, {"out": "x"})
        g.add_data_edge(k, succ, {"y": "inp"})
        g.split_kernel(row_split, k, N)
        g.validate()


class TestRowSplitStridedGemm:
    def setup_method(self):
        self.kernel = _make_test_strided_gemm()
        self.prev, self.copies, self.nxt = row_split(self.kernel, N)

    def test_copies_strided(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, StridedGemm)
            assert c.K == 16  # 64/4
            assert c._in_elems == 32 * 64 // N

    def test_shapes(self):
        for c in self.copies:
            assert c.inputs["x"].shape == (32, 16)
            assert c.outputs["y"].shape == (32, 128)


# ── head_split ─────────────────────────────────────────────────────────


class TestHeadSplit:
    def setup_method(self):
        B, H, H_kv, S_q, k_sel, S_kv, Hd = 1, 8, 1, 64, 16, 80, 64
        self.kernel = SparseAttn(B, H, H_kv, S_q, k_sel, S_kv, Hd, "bf16",
                                 kv_factor=1)
        self.kernel.inputs = {
            "q": Tensor("bf16", (64, H * Hd)),
            "kv": Tensor("bf16", (80, Hd)),
        }
        self.kernel.outputs = {"y": Tensor("bf16", (64, H * Hd))}
        self.prev, self.copies, self.nxt = head_split(self.kernel, N)

    def test_prev_scatter_q(self):
        assert "q" in self.prev
        assert isinstance(self.prev["q"], Scatter)
        assert self.prev["q"].inputs["x"].shape == (64, 512)
        assert self.prev["q"].outputs["o0"].shape == (64, 128)  # H*Hd/4

    def test_prev_broadcast_kv(self):
        assert "kv" in self.prev
        assert isinstance(self.prev["kv"], Broadcast)
        assert self.prev["kv"].inputs["x"].shape == (80, 64)
        assert self.prev["kv"].outputs["o0"].shape == (80, 64)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, SparseAttn)
            assert c.H == 2  # 8/4
            assert c.inputs["q"].shape == (64, 128)
            assert c.inputs["kv"].shape == (80, 64)
            assert c.outputs["y"].shape == (64, 128)

    def test_next_gather(self):
        assert "y" in self.nxt
        assert isinstance(self.nxt["y"], Gather)
        assert self.nxt["y"].inputs["i0"].shape == (64, 128)
        assert self.nxt["y"].outputs["y"].shape == (64, 512)

    def test_graph_validates(self):
        B, H, H_kv, S_q, k_sel, S_kv, Hd = 1, 8, 1, 64, 16, 80, 64
        g = ComputeGraph()
        k = SparseAttn(B, H, H_kv, S_q, k_sel, S_kv, Hd, "bf16", kv_factor=1)
        k.inputs = {"q": Tensor("bf16", (64, H * Hd)),
                    "kv": Tensor("bf16", (80, Hd))}
        k.outputs = {"y": Tensor("bf16", (64, H * Hd))}
        pred_q = Kernel(outputs={"out": Tensor("bf16", (64, 512))})
        pred_kv = Kernel(outputs={"out": Tensor("bf16", (80, 64))})
        succ = Kernel(inputs={"inp": Tensor("bf16", (64, 512))})
        g.add_kernel(pred_q)
        g.add_kernel(pred_kv)
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(pred_q, k, {"out": "q"})
        g.add_data_edge(pred_kv, k, {"out": "kv"})
        g.add_data_edge(k, succ, {"y": "inp"})
        g.split_kernel(head_split, k, N)
        g.validate()


# ── batch_split ────────────────────────────────────────────────────────


class TestBatchSplitGemm:
    def setup_method(self):
        M, K_dim, N_dim = 32, 64, 128
        self.kernel = Gemm(M, N_dim, K_dim, "bf16", "bf16")
        self.kernel.inputs = {"x": Tensor("bf16", (M, K_dim))}
        self.kernel.weights = {"w": Tensor("bf16", (K_dim, N_dim))}
        self.kernel.outputs = {"y": Tensor("bf16", (M, N_dim))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_prev_comms_scatter(self):
        assert "x" in self.prev
        assert isinstance(self.prev["x"], Scatter)
        assert self.prev["x"].world == N
        assert self.prev["x"].inputs["x"].shape == (32, 64)
        assert self.prev["x"].outputs["o0"].shape == (8, 64)

    def test_copies_count_and_M(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, Gemm)
            assert c.M == 8
            assert c.N == 128
            assert c.K == 64

    def test_copies_shapes(self):
        for c in self.copies:
            assert c.inputs["x"].shape == (8, 64)
            assert c.outputs["y"].shape == (8, 128)
            assert c.weights["w"].shape == (64, 128)

    def test_next_comms_gather(self):
        assert "y" in self.nxt
        assert isinstance(self.nxt["y"], Gather)
        assert self.nxt["y"].world == N
        assert self.nxt["y"].inputs["i0"].shape == (8, 128)
        assert self.nxt["y"].outputs["y"].shape == (32, 128)


class TestBatchSplitStridedGemm:
    def setup_method(self):
        M = 32
        self.kernel = StridedGemm(M, 256, 64, "fp8", "bf16",
                                  in_elems=M * 64, out_elems=M * 128)
        self.kernel.inputs = {"x": Tensor("bf16", (M, 64))}
        self.kernel.weights = {"w": Tensor("fp8", (64, 256))}
        scale_bytes = gemm_scale_bytes(256, 64, "fp8")
        if scale_bytes > 0:
            self.kernel.weights["s"] = Tensor("ue8m0", (int(scale_bytes),))
        self.kernel.outputs = {"y": Tensor("bf16", (M, 128))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_copies_strided(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, StridedGemm)
            assert c.M == 8
            assert c._in_elems == 32 * 64 // N
            assert c._out_elems == 32 * 128 // N

    def test_copies_shapes(self):
        for c in self.copies:
            assert c.inputs["x"].shape == (8, 64)
            assert c.outputs["y"].shape == (8, 128)


class TestBatchSplitRMSNorm:
    def setup_method(self):
        M, D = 32, 64
        self.kernel = RMSNorm(M, D, "bf16")
        self.kernel.inputs = {"x": Tensor("bf16", (M, D))}
        self.kernel.weights = {"g": Tensor("bf16", (D,))}
        self.kernel.outputs = {"y": Tensor("bf16", (M, D))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, RMSNorm)
            assert c.M == 8
            assert c.D == 64

    def test_weights_replicated(self):
        for c in self.copies:
            assert c.weights["g"].shape == (64,)

    def test_shapes(self):
        assert self.prev["x"].outputs["o0"].shape == (8, 64)
        assert self.nxt["y"].inputs["i0"].shape == (8, 64)


class TestBatchSplitEmbedding:
    def setup_method(self):
        M, V, D = 32, 1000, 64
        self.kernel = Embedding(M, V, D)
        self.kernel.inputs = {"idx": Tensor("int32", (M,))}
        self.kernel.weights = {"emb": Tensor("bf16", (V, D))}
        self.kernel.outputs = {"y": Tensor("bf16", (M, D))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_prev_scatter_idx(self):
        assert "idx" in self.prev
        assert isinstance(self.prev["idx"], Scatter)
        assert self.prev["idx"].inputs["x"].shape == (32,)
        assert self.prev["idx"].outputs["o0"].shape == (8,)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, Embedding)
            assert c.M == 8
            assert c.V == 1000
            assert c.D == 64

    def test_copies_weight_shape(self):
        for c in self.copies:
            assert c.weights["emb"].shape == (1000, 64)


class TestBatchSplitReadInput:
    def setup_method(self):
        self.kernel = ReadInput(32, "int32")
        self.kernel.inputs = {"tokens": Tensor("int32", (4, 8))}
        self.kernel.outputs = {"tokens": Tensor("int32", (4, 8))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, ReadInput)
            assert c.n_elements == 8

    def test_shapes(self):
        assert self.prev["tokens"].outputs["o0"].shape == (1, 8)
        assert self.nxt["tokens"].inputs["i0"].shape == (1, 8)


class TestBatchSplitSpawn:
    def setup_method(self):
        self.kernel = Spawn(world=2)
        self.kernel.inputs = {"x": Tensor("bf16", (32, 64))}
        self.kernel.outputs = {"y": Tensor("bf16", (32, 64)),
                               "y2": Tensor("bf16", (32, 64))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_prev_comms_single_input(self):
        assert len(self.prev) == 1
        assert "x" in self.prev
        assert isinstance(self.prev["x"], Scatter)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, Spawn)
            assert c.world == 2
            assert c.inputs["x"].shape == (8, 64)
            assert c.outputs["y"].shape == (8, 64)
            assert c.outputs["y2"].shape == (8, 64)

    def test_next_comms_per_output(self):
        assert len(self.nxt) == 2
        assert "y" in self.nxt and "y2" in self.nxt
        for port in ("y", "y2"):
            assert isinstance(self.nxt[port], Gather)
            assert self.nxt[port].inputs["i0"].shape == (8, 64)
            assert self.nxt[port].outputs["y"].shape == (32, 64)


class TestBatchSplitConcat:
    def setup_method(self):
        self.kernel = Concat()
        self.kernel.inputs = {"a": Tensor("bf16", (32, 64)),
                              "b": Tensor("bf16", (16, 64))}
        self.kernel.outputs = {"y": Tensor("bf16", (48, 64))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_prev_comms_per_input(self):
        assert len(self.prev) == 2
        assert "a" in self.prev and "b" in self.prev
        assert isinstance(self.prev["a"], Scatter)
        assert isinstance(self.prev["b"], Scatter)
        assert self.prev["a"].outputs["o0"].shape == (8, 64)
        assert self.prev["b"].outputs["o0"].shape == (4, 64)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, Concat)
            assert c.inputs["a"].shape == (8, 64)
            assert c.inputs["b"].shape == (4, 64)
            assert c.outputs["y"].shape == (12, 64)

    def test_next_comms(self):
        assert "y" in self.nxt
        assert self.nxt["y"].inputs["i0"].shape == (12, 64)
        assert self.nxt["y"].outputs["y"].shape == (48, 64)


class TestBatchSplitSlice:
    def setup_method(self):
        self.kernel = Slice()
        self.kernel.inputs = {"x": Tensor("bf16", (32, 64))}
        self.kernel.outputs = {"y": Tensor("bf16", (32, 8))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_copies(self):
        assert len(self.copies) == N
        for copy in self.copies:
            assert isinstance(copy, Slice)
            assert copy.inputs["x"].shape == (8, 64)
            assert copy.outputs["y"].shape == (8, 8)
            assert copy._requires_placement is True


class TestBatchSplitTokenDispatch:
    def setup_method(self):
        M, D, N_experts, topk = 32, 64, 8, 2
        self.kernel = TokenDispatch(M, D, N_experts, topk)
        self.kernel.inputs = {"x": Tensor("bf16", (M, D)),
                              "routing": Tensor("fp32", (M, N_experts))}
        M_e = M * topk // N_experts  # 8
        self.kernel.outputs = {f"o{i}": Tensor("bf16", (M_e, D))
                               for i in range(N_experts)}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_prev_comms(self):
        assert len(self.prev) == 2
        assert "x" in self.prev and "routing" in self.prev
        assert self.prev["x"].outputs["o0"].shape == (8, 64)
        assert self.prev["routing"].outputs["o0"].shape == (8, 8)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, TokenDispatch)
            assert c.M == 8
            assert c.M_e == 2  # 8*2/8

    def test_next_comms_per_expert(self):
        assert len(self.nxt) == 8
        for i in range(8):
            g = self.nxt[f"o{i}"]
            assert isinstance(g, Gather)
            assert g.inputs["i0"].shape == (2, 64)
            assert g.outputs["y"].shape == (8, 64)


class TestBatchSplitTokenCombine:
    def setup_method(self):
        M, D, N_experts, topk = 32, 64, 8, 2
        M_e = M * topk // N_experts  # 8
        self.kernel = TokenCombine(M, D, N_experts, topk)
        self.kernel.inputs = {f"i{i}": Tensor("bf16", (M_e, D))
                              for i in range(N_experts)}
        self.kernel.outputs = {"y": Tensor("bf16", (M, D))}
        self.prev, self.copies, self.nxt = batch_split(self.kernel, N)

    def test_prev_comms_per_expert(self):
        assert len(self.prev) == 8
        for i in range(8):
            s = self.prev[f"i{i}"]
            assert isinstance(s, Scatter)
            assert s.inputs["x"].shape == (8, 64)
            assert s.outputs["o0"].shape == (2, 64)

    def test_copies(self):
        assert len(self.copies) == N
        for c in self.copies:
            assert isinstance(c, TokenCombine)
            assert c.M == 8
            assert c.M_e == 2

    def test_next_comms(self):
        assert len(self.nxt) == 1
        g = self.nxt["y"]
        assert isinstance(g, Gather)
        assert g.inputs["i0"].shape == (8, 64)
        assert g.outputs["y"].shape == (32, 64)


class TestBatchSplitGraphIntegration:
    def test_gemm_split_validates(self):
        g = ComputeGraph()
        M, K_dim, N_dim = 32, 64, 128
        pred = Kernel(outputs={"out": Tensor("bf16", (M, K_dim))})
        succ = Kernel(inputs={"inp": Tensor("bf16", (M, N_dim))})
        k = Gemm(M, N_dim, K_dim, "bf16", "bf16")
        k.inputs = {"x": Tensor("bf16", (M, K_dim))}
        k.weights = {"w": Tensor("bf16", (K_dim, N_dim))}
        k.outputs = {"y": Tensor("bf16", (M, N_dim))}
        g.add_kernel(pred)
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(pred, k, {"out": "x"})
        g.add_data_edge(k, succ, {"y": "inp"})
        prev, copies, nxt = g.split_kernel(batch_split, k, N)
        g.validate()
        assert k not in g.kernels
        assert len(copies) == N
        assert all(c in g.kernels for c in copies)

    def test_spawn_split_validates(self):
        g = ComputeGraph()
        pred = Kernel(outputs={"out": Tensor("bf16", (32, 64))})
        succ1 = Kernel(inputs={"a": Tensor("bf16", (32, 64))})
        succ2 = Kernel(inputs={"b": Tensor("bf16", (32, 64))})
        k = Spawn(world=2)
        k.inputs = {"x": Tensor("bf16", (32, 64))}
        k.outputs = {"y": Tensor("bf16", (32, 64)),
                     "y2": Tensor("bf16", (32, 64))}
        g.add_kernel(pred)
        g.add_kernel(k)
        g.add_kernel(succ1)
        g.add_kernel(succ2)
        g.add_data_edge(pred, k, {"out": "x"})
        g.add_data_edge(k, succ1, {"y": "a"})
        g.add_data_edge(k, succ2, {"y2": "b"})
        g.split_kernel(batch_split, k, N)
        g.validate()

    def test_concat_split_validates(self):
        g = ComputeGraph()
        pred_a = Kernel(outputs={"out": Tensor("bf16", (32, 64))})
        pred_b = Kernel(outputs={"out": Tensor("bf16", (16, 64))})
        succ = Kernel(inputs={"inp": Tensor("bf16", (48, 64))})
        k = Concat()
        k.inputs = {"a": Tensor("bf16", (32, 64)),
                    "b": Tensor("bf16", (16, 64))}
        k.outputs = {"y": Tensor("bf16", (48, 64))}
        g.add_kernel(pred_a)
        g.add_kernel(pred_b)
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(pred_a, k, {"out": "a"})
        g.add_data_edge(pred_b, k, {"out": "b"})
        g.add_data_edge(k, succ, {"y": "inp"})
        g.split_kernel(batch_split, k, N)
        g.validate()

    def test_unsupported_kernel_raises(self):
        k = Kernel()
        k.inputs = {"x": Tensor("bf16", (32, 64))}
        k.outputs = {"y": Tensor("bf16", (32, 64))}
        with pytest.raises(TypeError, match="unsupported kernel type"):
            batch_split(k, N)


# ── weight_id propagation tests ────────────────────────────────────────


class TestWeightIdBatchSplit:
    """batch_split preserves weight_id unchanged (DP = replicate)."""

    def test_gemm_weight_id_preserved(self):
        k = Gemm(32, 64, 128, "fp8", "bf16")
        k.inputs = {"x": Tensor("bf16", (32, 128))}
        k.weights = {"w": Tensor("fp8", (128, 64),
                                 weight_id="L0_wq_a_w")}
        k.outputs = {"y": Tensor("bf16", (32, 64))}
        _, copies, _ = batch_split(k, N)
        for c in copies:
            assert c.weights["w"].weight_id == "L0_wq_a_w"

    def test_embedding_weight_id_preserved(self):
        k = Embedding(32, 1000, 64)
        k.inputs = {"idx": Tensor("int32", (32,))}
        k.weights = {"emb": Tensor("bf16", (1000, 64),
                                   weight_id="L-1_emb_emb")}
        k.outputs = {"y": Tensor("bf16", (32, 64))}
        _, copies, _ = batch_split(k, N)
        for c in copies:
            assert c.weights["emb"].weight_id == "L-1_emb_emb"


class TestWeightIdColumnSplit:
    """column_split appends /col:{i} shard tag."""

    def test_gemm_col_tag(self):
        k = Gemm(32, 64, 128, "fp8", "bf16")
        k.inputs = {"x": Tensor("bf16", (32, 128))}
        k.weights = {"w": Tensor("fp8", (128, 64),
                                 weight_id="L0_wq_b_w")}
        k.outputs = {"y": Tensor("bf16", (32, 64))}
        _, copies, _ = column_split(k, N)
        for i, c in enumerate(copies):
            assert c.weights["w"].weight_id == f"L0_wq_b_w/col:{i}"

    def test_no_weight_id_stays_none(self):
        k = Gemm(32, 64, 128, "fp8", "bf16")
        k.inputs = {"x": Tensor("bf16", (32, 128))}
        k.weights = {"w": Tensor("fp8", (128, 64))}
        k.outputs = {"y": Tensor("bf16", (32, 64))}
        _, copies, _ = column_split(k, N)
        for c in copies:
            assert c.weights["w"].weight_id is None


class TestWeightIdRowSplit:
    """row_split appends /row:{i} shard tag."""

    def test_gemm_row_tag(self):
        k = Gemm(32, 64, 128, "fp8", "bf16")
        k.inputs = {"x": Tensor("bf16", (32, 128))}
        k.weights = {"w": Tensor("fp8", (128, 64),
                                 weight_id="L0_wo_b_w")}
        k.outputs = {"y": Tensor("bf16", (32, 64))}
        _, copies, _ = row_split(k, N)
        for i, c in enumerate(copies):
            assert c.weights["w"].weight_id == f"L0_wo_b_w/row:{i}"
