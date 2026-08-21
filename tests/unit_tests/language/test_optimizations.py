"""Unit tests for rooflang.language.optimization (split + optimize_comms)."""

from fractions import Fraction

import pytest

from rooflang.language.graph import ComputeGraph, FabricEdge, HardwareGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import (
    AllGather, AllReduce, AllToAll, Broadcast, CommKernel, Gather, Reduce,
    ReduceScatter, Scatter, Send, Recv,
)
from rooflang.language.kernels.forward import (
    Attn, DpskV4SparseAttn, Embedding, Gemm, Glm52SparseAttn, Nop, ReadInput,
    RMSNorm, Slice, StridedGemm, TokenCombine, TokenDispatch,
)
from rooflang.language.kernels.identity import Concat, Spawn
from rooflang.language.tensor import Tensor
from rooflang.language.placement import Placement
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.optimization.comm import (
    _create_collective, optimize_comms,
)
from rooflang.language.optimization.split import (
    batch_split, column_split, context_split_decode, context_split_prefill,
    general_dup, head_split, kv_persistence_split, row_split,
)
from rooflang.language.utils import gemm_scale_bytes


SHARD = Tensor("bf16", (4, 4))


def _build_chain(collector, distributor, n, same_devices=True):
    """Build n preds → collector → distributor → n succs with placement."""
    g = ComputeGraph()
    gpus = [Compute(name=f"gpu{i}") for i in range(n)]

    preds = [Kernel(outputs={"y": Tensor("bf16", SHARD.shape)})
             for _ in range(n)]
    succs = [Kernel(inputs={"a": Tensor("bf16", SHARD.shape)})
             for _ in range(n)]

    collector.inputs = {
        f"x{i}": Tensor("bf16", SHARD.shape) for i in range(n)
    }
    collector.outputs = {"z": Tensor("bf16", SHARD.shape)}
    distributor.inputs = {"z": Tensor("bf16", SHARD.shape)}
    distributor.outputs = {
        f"y{i}": Tensor("bf16", SHARD.shape) for i in range(n)
    }

    for k in preds + [collector, distributor] + succs:
        g.add_kernel(k)
    for i, p in enumerate(preds):
        g.add_data_edge(p, collector, {"y": f"x{i}"})
    g.add_data_edge(collector, distributor, {"z": "z"})
    for i, s in enumerate(succs):
        g.add_data_edge(distributor, s, {f"y{i}": "a"})

    other_gpus = []
    if not same_devices:
        other_gpus = [Compute(name=f"other{i}") for i in range(n)]
    hw = HardwareGraph()
    memories = {}
    for device in gpus + other_gpus:
        memory = Memory(name=f"{device.name}-hbm", capacity_gb=1.0)
        hw.add_node(device)
        hw.add_node(memory)
        hw.add_edge(FabricEdge(
            name=f"{device.name}-hbm", src=device, dst=memory,
            src_to_dst_bandwidth_gbs=1.0,
            dst_to_src_bandwidth_gbs=1.0,
            is_full_duplex=False,
        ))
        memories[device] = memory

    placement = Placement(hardware=hw, graph=g)
    for i, p in enumerate(preds):
        placement.set_kernel_device(p, gpus[i])
    if same_devices:
        for i, s in enumerate(succs):
            placement.set_kernel_device(s, gpus[i])
        successor_devices = gpus
    else:
        for i, s in enumerate(succs):
            placement.set_kernel_device(s, other_gpus[i])
        successor_devices = other_gpus

    for i, tensor in enumerate(collector.inputs.values()):
        placement.set_tensor_memory(tensor, memories[gpus[i]])
    placement.set_tensor_memory(collector.outputs["z"], memories[gpus[0]])
    placement.set_tensor_memory(distributor.inputs["z"], memories[gpus[0]])
    for device, tensor in zip(
            successor_devices, distributor.outputs.values()):
        placement.set_tensor_memory(tensor, memories[device])

    return g, placement, preds, succs


class TestFuseWorld4:
    def test_reduce_broadcast_to_allreduce(self):
        r = Reduce(total_bytes=32, world=4, dtype="bf16")
        b = Broadcast(total_bytes=32, world=4)
        g, p, preds, succs = _build_chain(r, b, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllReduce)

    def test_reduce_scatter_to_reducescatter(self):
        r = Reduce(total_bytes=32, world=4, dtype="bf16")
        s = Scatter(total_bytes=32, world=4, dim=1)
        g, p, preds, succs = _build_chain(r, s, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), ReduceScatter)

    def test_gather_broadcast_to_allgather(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        b = Broadcast(total_bytes=128, world=4)
        g, p, preds, succs = _build_chain(ga, b, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllGather)

    def test_gather_scatter_diff_dim_to_alltoall(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        s = Scatter(total_bytes=128, world=4, dim=1)
        g, p, preds, succs = _build_chain(ga, s, n=4)
        optimize_comms(g)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllToAll)

    def test_fused_collective_does_not_require_placement(self):
        gather = Gather(total_bytes=128, world=2, dim=0)
        broadcast = Broadcast(total_bytes=128, world=2)
        graph, placement, preds, succs = _build_chain(
            gather, broadcast, n=2)

        graph_only = ComputeGraph()
        for kernel in (*preds, gather, broadcast, *succs):
            graph_only.add_kernel(kernel)
        for index, pred in enumerate(preds):
            graph_only.add_data_edge(
                pred, gather, {"y": f"x{index}"})
        graph_only.add_data_edge(gather, broadcast, {"z": "z"})
        for index, succ in enumerate(succs):
            graph_only.add_data_edge(
                broadcast, succ, {f"y{index}": "a"})

        optimize_comms(graph_only)

        collective = next(
            kernel for kernel in graph_only.kernels
            if isinstance(kernel, AllGather)
        )
        assert len(collective.inputs) == 2
        assert len(collective.outputs) == 2
        assert all(
            placement.get_tensor_memory(tensor) is None
            for tensor in (*collective.inputs.values(),
                           *collective.outputs.values())
        )


class TestFuseWorld1Eliminated:
    def test_reduce_broadcast_world1_eliminated(self):
        r = Reduce(total_bytes=32, world=1, dtype="bf16")
        b = Broadcast(total_bytes=32, world=1)
        g, p, preds, succs = _build_chain(r, b, n=1)
        optimize_comms(g)
        assert g.kernels == frozenset(preds + succs)
        assert g._out_edges(preds[0])[0].dst is succs[0]

    def test_gather_scatter_same_dim_world1_eliminated(self):
        ga = Gather(total_bytes=32, world=1, dim=0)
        s = Scatter(total_bytes=32, world=1, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g)
        assert g.kernels == frozenset(preds + succs)

    def test_gather_broadcast_world1_eliminated(self):
        ga = Gather(total_bytes=32, world=1, dim=0)
        b = Broadcast(total_bytes=32, world=1)
        g, p, preds, succs = _build_chain(ga, b, n=1)
        optimize_comms(g)
        assert g.kernels == frozenset(preds + succs)


class TestBypass:
    def test_gather_scatter_same_dim_bypass(self):
        ga = Gather(total_bytes=128, world=4, dim=0)
        s = Scatter(total_bytes=128, world=4, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g)
        assert ga not in g.kernels
        assert s not in g.kernels
        assert g._out_edges(preds[0])[0].dst is succs[0]


class TestPlacementIndependentRewrite:
    @pytest.mark.parametrize("comm_cls", [Gather, Scatter])
    def test_same_direction_hierarchy_is_preserved(self, comm_cls):
        first = comm_cls(total_bytes=128, world=4, dim=0)
        second = comm_cls(total_bytes=128, world=2, dim=1)
        graph, placement, preds, succs = _build_chain(
            first, second, n=4, same_devices=False)

        optimize_comms(graph)

        assert first in graph.kernels
        assert second in graph.kernels
        assert set(preds + succs + [first, second]) == graph.kernels

    def test_gather_scatter_becomes_rank_direct_edges(self):
        gather = Gather(total_bytes=128, world=4, dim=0)
        scatter = Scatter(total_bytes=128, world=4, dim=0)
        g, p, preds, succs = _build_chain(
            gather, scatter, n=4, same_devices=False)

        optimize_comms(g)

        assert gather not in g.kernels
        assert scatter not in g.kernels
        for pred, succ in zip(preds, succs):
            assert g._out_edges(pred)[0].dst is succ

    def test_different_memories_do_not_affect_graph_rewrite(self):
        gather = Gather(total_bytes=SHARD.size_bytes, world=1, dim=0)
        scatter = Scatter(total_bytes=SHARD.size_bytes, world=1, dim=0)
        graph, placement, preds, succs = _build_chain(
            gather, scatter, n=1, same_devices=False)
        pred, succ = preds[0], succs[0]
        optimize_comms(graph)

        assert graph.kernels == frozenset({pred, succ})
        assert graph._out_edges(pred)[0].dst is succ

    def test_same_memory_uses_direct_edge(self):
        gather = Gather(total_bytes=128, world=1, dim=0)
        scatter = Scatter(total_bytes=128, world=1, dim=0)
        graph, placement, preds, succs = _build_chain(
            gather, scatter, n=1, same_devices=False)
        shared = Memory(name="shared", capacity_gb=1.0)
        placement.set_tensor_memory(preds[0].outputs["y"], shared)
        placement.set_tensor_memory(succs[0].inputs["a"], shared)

        optimize_comms(graph)

        assert graph.kernels == frozenset(preds + succs)
        assert graph._out_edges(preds[0])[0].dst is succs[0]

    def test_unrecognized_pair_is_preserved(self):
        first = AllGather(total_bytes=32, world=4)
        second = AllReduce(total_bytes=32, world=4, dtype="bf16")
        g, p, _, _ = _build_chain(
            first, second, n=4, same_devices=False)
        optimize_comms(g)
        assert first in g.kernels
        assert second in g.kernels

    def test_gather_scatter_different_dims_becomes_alltoall(self):
        gather = Gather(total_bytes=128, world=4, dim=0)
        scatter = Scatter(total_bytes=128, world=4, dim=1)
        g, p, _, _ = _build_chain(
            gather, scatter, n=4, same_devices=False)
        optimize_comms(g)
        assert any(isinstance(kernel, AllToAll) for kernel in g.kernels)
        assert gather not in g.kernels
        assert scatter not in g.kernels

    def test_gather_scatter_different_worlds_is_preserved(self):
        gather = Gather(total_bytes=128, world=4, dim=0)
        scatter = Scatter(total_bytes=128, world=2, dim=0)
        g, p, preds, succs = _build_chain(
            gather, scatter, n=4, same_devices=False)
        optimize_comms(g)
        assert gather in g.kernels
        assert scatter in g.kernels


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
        optimize_comms(g)
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
        optimize_comms(g)
        assert comm in g.kernels

    def test_removed_neighbor_can_make_an_earlier_comm_trivial(self):
        graph = ComputeGraph()
        pred = Kernel(outputs={"y": SHARD})
        first = Scatter(total_bytes=128, world=2, dim=0)
        first.inputs = {"x": SHARD}
        first.outputs = {"y0": SHARD, "y1": SHARD}
        second = Send(total_bytes=32)
        second.inputs = {"x": SHARD}
        second.outputs = {"y": SHARD}
        succ = Kernel(inputs={"a": SHARD, "b": SHARD})
        for kernel in (pred, first, second, succ):
            graph.add_kernel(kernel)
        graph.add_data_edge(pred, first, {"y": "x"})
        graph.add_data_edge(first, second, {"y0": "x"})
        graph.add_data_edge(first, succ, {"y1": "b"})
        graph.add_data_edge(second, succ, {"y": "a"})

        optimize_comms(graph)

        assert graph.kernels == frozenset({pred, succ})


# ── Guard clause tests (_fuse_pairs skips) ───────────────────────────


class TestFusePairsGuards:
    def test_create_collective_rejects_unexpected_pair(self):
        with pytest.raises(ValueError, match="Unexpected pair"):
            _create_collective(
                Broadcast(total_bytes=8, world=2),
                Gather(total_bytes=8, world=2, dim=0),
            )

    def test_same_dim_mismatched_shards_are_preserved(self):
        gather = Gather(total_bytes=128.0, world=2, dim=0)
        scatter = Scatter(total_bytes=128.0, world=2, dim=0)
        graph, _, _, _ = _build_chain(gather, scatter, n=2)
        scatter.outputs["y0"] = Tensor("bf16", (2, 8))

        optimize_comms(graph)

        assert gather in graph.kernels
        assert scatter in graph.kernels

    def test_bypass_tolerates_an_unconsumed_rank(self):
        gather = Gather(total_bytes=128.0, world=2, dim=0)
        scatter = Scatter(total_bytes=128.0, world=2, dim=0)
        graph, _, preds, succs = _build_chain(gather, scatter, n=2)
        graph.remove_kernel(succs[1])

        optimize_comms(graph)

        assert gather not in graph.kernels
        assert scatter not in graph.kernels
        assert graph._out_edges(preds[0])[0].dst is succs[0]
        assert not graph._out_edges(preds[1])

    def test_collector_multi_successor_is_preserved(self):
        gather = Gather(total_bytes=128.0, world=2, dim=0)
        scatter = Scatter(total_bytes=128.0, world=2, dim=0)
        graph, placement, preds, succs = _build_chain(
            gather, scatter, n=2, same_devices=False)
        gather.outputs["extra"] = SHARD
        extra = Kernel(inputs={"x": SHARD})
        graph.add_kernel(extra)
        graph.add_data_edge(gather, extra, {"extra": "x"})
        source_device = placement.get_kernel_device(preds[0]).device
        placement.set_kernel_device(extra, source_device)
        placement.set_tensor_memory(
            gather.outputs["extra"],
            placement.get_tensor_memory(extra.inputs["x"]))

        optimize_comms(graph)
        assert gather in graph.kernels
        assert scatter in graph.kernels
        assert set(preds + succs + [extra, gather, scatter]) == graph.kernels

    def test_collector_multi_out_edges_no_fuse(self):
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=2)
        g, p, preds, succs = _build_chain(r, b, n=2)
        r.outputs["z2"] = Tensor("bf16", SHARD.shape)
        extra = Kernel(inputs={"q": Tensor("bf16", SHARD.shape)})
        g.add_kernel(extra)
        g.add_data_edge(r, extra, {"z2": "q"})
        device = p.get_kernel_device(preds[0]).device
        p.set_kernel_device(extra, device)
        p.set_tensor_memory(
            r.outputs["z2"], p.get_tensor_memory(extra.inputs["q"]))
        optimize_comms(g)
        assert r in g.kernels
        assert b in g.kernels

    def test_succ_multi_in_edges_no_fuse(self):
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=2)
        g, p, preds, succs = _build_chain(r, b, n=2)
        b.inputs["q"] = Tensor("bf16", SHARD.shape)
        extra = Kernel(outputs={"q": Tensor("bf16", SHARD.shape)})
        g.add_kernel(extra)
        g.add_data_edge(extra, b, {"q": "q"})
        device = p.get_kernel_device(preds[0]).device
        p.set_kernel_device(extra, device)
        p.set_tensor_memory(
            b.inputs["q"], p.get_tensor_memory(extra.outputs["q"]))
        optimize_comms(g)
        assert r in g.kernels
        assert b in g.kernels

    def test_world_mismatch_no_fuse(self):
        r = Reduce(total_bytes=128.0, world=2, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=4)
        g, p, _, _ = _build_chain(r, b, n=2)
        optimize_comms(g)
        assert r in g.kernels
        assert b in g.kernels

    def test_incomplete_rank_ports_are_not_fused(self):
        r = Reduce(total_bytes=128.0, world=3, dtype="bf16")
        b = Broadcast(total_bytes=128.0, world=3)
        g, p, _, succs = _build_chain(r, b, n=3)
        g.remove_kernel(succs[2])
        del b.outputs["y2"]
        optimize_comms(g)
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
        self.kernel = DpskV4SparseAttn(
            B, H, H_kv, S_q, k_sel, S_kv, Hd, "bf16", kv_factor=1)
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
            assert isinstance(c, DpskV4SparseAttn)
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
        k = DpskV4SparseAttn(
            B, H, H_kv, S_q, k_sel, S_kv, Hd, "bf16", kv_factor=1)
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


# ── context_split ─────────────────────────────────────────────────────


class TestContextSplitGemm:
    def setup_method(self):
        self.kernel = Gemm(32, 128, 64, "bf16", "bf16")
        self.kernel.inputs = {"x": Tensor("bf16", (2, 16, 64))}
        self.kernel.weights = {"w": Tensor("bf16", (64, 128))}
        self.kernel.outputs = {"y": Tensor("bf16", (2, 16, 128))}
        self.prev, self.copies, self.nxt = context_split_prefill(
            self.kernel, N)

    def test_context_axis_and_kernel_work_are_sharded(self):
        assert isinstance(self.prev["x"], Scatter)
        assert self.prev["x"].dim == 1
        assert self.prev["x"].outputs["o0"].shape == (2, 4, 64)
        for copy in self.copies:
            assert copy.M == 8
            assert copy.inputs["x"].shape == (2, 4, 64)
            assert copy.outputs["y"].shape == (2, 4, 128)
        assert isinstance(self.nxt["y"], Gather)
        assert self.nxt["y"].dim == 1


@pytest.mark.parametrize("attn_cls", [Attn, DpskV4SparseAttn])
def test_context_split_attention_uses_full_logical_kv(attn_cls):
    kwargs = dict(B=2, H=8, H_kv=1, S_q=16, S_kv=24, Hd=64,
                  dtype="bf16")
    if attn_cls is DpskV4SparseAttn:
        kwargs["k_sel"] = 8
        kwargs["kv_factor"] = 1
    kernel = attn_cls(**kwargs)
    kv_width = 64 if attn_cls is DpskV4SparseAttn else 2 * 64
    kernel.inputs = {
        "q": Tensor("bf16", (2, 16, 8 * 64)),
        "kv": Tensor("bf16", (2, 24, kv_width)),
    }
    kernel.outputs = {"y": Tensor("bf16", (2, 16, 8 * 64))}

    prev, copies, nxt = context_split_prefill(kernel, N)

    assert isinstance(prev["q"], Scatter)
    assert isinstance(prev["kv"], Broadcast)
    assert prev["kv"].outputs["o0"].shape == (2, 24, kv_width)
    assert isinstance(nxt["y"], Gather)
    assert sum(copy.flops for copy in copies) == kernel.flops
    for copy in copies:
        assert copy.S_q == 4
        assert copy.S_kv == 24
        assert copy.inputs["q"].shape == (2, 4, 8 * 64)
        assert copy.inputs["kv"].shape == (2, 24, kv_width)
        assert copy.outputs["y"].shape == (2, 4, 8 * 64)
        assert copy.input_bytes == sum(
            tensor.size_bytes for tensor in copy.inputs.values())


@pytest.mark.parametrize("attn_cls", [Attn, DpskV4SparseAttn])
def test_context_split_preserves_causal_flop_factor(attn_cls):
    kwargs = dict(B=1, H=8, H_kv=1, S_q=16, S_kv=16, Hd=64,
                  dtype="bf16", causal=True)
    if attn_cls is DpskV4SparseAttn:
        kwargs.update(
            S_kv=12, k_sel=8, kv_factor=1,
            indexer_s_kv=8, indexer_h=4, indexer_hd=8,
            causal_k_sel=4)
    kernel = attn_cls(**kwargs)
    kv_width = 64 if attn_cls is DpskV4SparseAttn else 2 * 64
    kernel.inputs = {
        "q": Tensor("bf16", (1, 16, 8 * 64)),
        "kv": Tensor("bf16", (1, kwargs["S_kv"], kv_width)),
    }
    if attn_cls is DpskV4SparseAttn:
        kernel.inputs["index_kv"] = Tensor("fp4", (1, 8, 8))
    kernel.outputs = {"y": Tensor("bf16", (1, 16, 8 * 64))}

    _, copies, _ = context_split_prefill(kernel, N)

    assert all(copy.causal for copy in copies)
    if attn_cls is DpskV4SparseAttn:
        assert all(copy.causal_k_sel == 4 for copy in copies)
        assert all(copy.indexer_compute_dtype == "fp4" for copy in copies)
        assert sum(copy.flops_by_dtype["fp4"] for copy in copies) \
            == kernel.flops_by_dtype["fp4"]
    assert sum(copy.flops for copy in copies) == kernel.flops


def test_context_split_dispatch_and_combine_port_axes():
    dispatch = TokenDispatch(32, 64, 8, 2)
    dispatch.inputs = {
        "x": Tensor("bf16", (2, 16, 64)),
        "routing": Tensor("fp32", (2, 16, 8)),
    }
    dispatch.outputs = {
        f"o{i}": Tensor("bf16", (8, 64)) for i in range(8)}
    _, dispatch_copies, dispatch_next = context_split_prefill(dispatch, N)
    assert dispatch_copies[0].inputs["x"].shape == (2, 4, 64)
    assert dispatch_copies[0].outputs["o0"].shape == (2, 64)
    assert dispatch_next["o0"].dim == 0

    combine = TokenCombine(32, 64, 8, 2)
    combine.inputs = {
        f"i{i}": Tensor("bf16", (8, 64)) for i in range(8)}
    combine.outputs = {"y": Tensor("bf16", (2, 16, 64))}
    combine_prev, combine_copies, _ = context_split_prefill(combine, N)
    assert combine_prev["i0"].dim == 0
    assert combine_copies[0].inputs["i0"].shape == (2, 64)
    assert combine_copies[0].outputs["y"].shape == (2, 4, 64)


def test_batch_split_preserves_fractional_expert_token_workload():
    dispatch = TokenDispatch(M=4, D=64, N_experts=8, topk=1)
    dispatch.inputs = {
        "x": Tensor("bf16", (4, 64)),
        "routing": Tensor("fp32", (4, 8)),
    }
    dispatch.outputs = {
        f"o{i}": Tensor("bf16", (Fraction(1, 2), 64))
        for i in range(8)
    }

    _, copies, next_comms = batch_split(dispatch, 4)

    assert all(copy.M_e == Fraction(1, 8) for copy in copies)
    assert all(copy.outputs["o0"].shape == (Fraction(1, 8), 64)
               for copy in copies)
    assert next_comms["o0"].inputs["i0"].shape == (Fraction(1, 8), 64)
    assert next_comms["o0"].outputs["y"].shape == (Fraction(1, 2), 64)


def test_context_split_nop_keeps_dummy_output_shape():
    nop = Nop(
        inputs={"kv": Tensor("bf16", (8, 16, 64))},
        outputs={"done": Tensor("int32", (1,))},
    )
    prev, copies, nxt = context_split_prefill(nop, N)

    assert isinstance(prev["kv"], Scatter)
    assert all(copy.inputs["kv"].shape == (8, 4, 64)
               for copy in copies)
    assert all(copy.outputs["done"].shape == (1,) for copy in copies)
    assert isinstance(nxt["done"], Gather)
    assert all(tensor.shape == (1,)
               for tensor in nxt["done"].inputs.values())


def test_batch_split_supports_terminal_nop_sink_without_gather():
    tensor = Tensor("bf16", (8, 16, 64))
    source = Nop(outputs={"y": Tensor(tensor.dtype, tensor.shape)})
    sink = Nop(inputs={"x": Tensor(tensor.dtype, tensor.shape)})
    graph = ComputeGraph()
    graph.add_kernel(source)
    graph.add_kernel(sink)
    graph.add_data_edge(source, sink, {"y": "x"})

    _, copies, next_comms = graph.split_kernel(batch_split, sink, N)

    assert not next_comms
    assert sink not in graph.kernels
    assert all(not copy.outputs and not graph._out_edges(copy)
               for copy in copies)


def test_context_split_decode_attention_broadcasts_q_and_shards_kv():
    kernel = DpskV4SparseAttn(
        8, 8, 1, 1, 8, 12, 64, "bf16", kv_factor=1,
        indexer_s_kv=16, indexer_h=4, indexer_hd=8)
    kernel.inputs = {
        "q": Tensor("bf16", (8, 1, 8 * 64)),
        "kv": Tensor("bf16", (8, 12, 64)),
        "index_kv": Tensor("fp4", (8, 16, 8)),
    }
    kernel.outputs = {"y": Tensor("bf16", (8, 1, 8 * 64))}
    prev, copies, nxt = context_split_decode(kernel, N)

    assert isinstance(prev["q"], Broadcast)
    assert isinstance(prev["kv"], Scatter)
    assert isinstance(prev["index_kv"], Scatter)
    assert all(copy.S_kv == 3 for copy in copies)
    assert all(copy.k_sel == 2 for copy in copies)
    assert all(copy.indexer_s_kv == 4 for copy in copies)
    assert all(copy.indexer_compute_dtype == "fp4" for copy in copies)
    assert sum(copy.flops_by_dtype["fp4"] for copy in copies) \
        == kernel.flops_by_dtype["fp4"]
    assert all(copy.inputs["index_kv"].shape == (8, 4, 8)
               for copy in copies)
    assert sum(copy.flops for copy in copies) == kernel.flops
    assert isinstance(nxt["y"], Reduce)


def _make_glm52_sparse_attn(B=1, S_q=8, S_kv=8, causal=True):
    kernel = Glm52SparseAttn(
        B=B, H=8, S_q=S_q, k_sel=4, S_kv=S_kv,
        qk_head_dim=6, v_head_dim=8, kv_cache_dim=10,
        kv_lora_rank=4, qk_nope_head_dim=2,
        indexer_mode="full", indexer_s_kv=S_kv,
        indexer_h=4, indexer_hd=4, causal=causal)
    kernel.inputs = {
        "q": Tensor("bf16", (B, S_q, 8 * 6)),
        "kv": Tensor("fp8", (B, S_kv, 10)),
        "index_q": Tensor("bf16", (B, S_q, 4 * 4)),
        "index_kv": Tensor("fp8", (B, S_kv, 4)),
        "index_weights": Tensor("fp32", (B, S_q, 4)),
    }
    kernel.weights = {
        "kv_b": Tensor("fp8", (4, 8 * (2 + 8))),
        "kv_b_scale": Tensor("ue8m0", (1,)),
    }
    kernel.outputs = {"y": Tensor("bf16", (B, S_q, 8 * 8))}
    return kernel


def test_glm52_head_split_preserves_full_indexer_work():
    kernel = _make_glm52_sparse_attn()

    prev, copies, nxt = head_split(kernel, N)

    assert set(prev) == {
        "q", "kv", "index_q", "index_kv", "index_weights"}
    assert isinstance(nxt["y"], Gather)
    assert sum(copy.flops for copy in copies) == kernel.flops
    for copy in copies:
        assert isinstance(copy, Glm52SparseAttn)
        assert copy.H == 2
        assert copy.indexer_h == 1
        assert copy.kv_cache_dim == 10
        assert copy.inputs["q"].shape == (1, 8, 2 * 6)
        assert copy.inputs["index_q"].shape == (1, 8, 4)
        assert copy.outputs["y"].shape == (1, 8, 2 * 8)
        assert copy.weights["kv_b"].shape == (4, 2 * (2 + 8))


def test_glm52_context_split_prefill_preserves_exact_causal_pairs():
    kernel = _make_glm52_sparse_attn()
    assert kernel.selected_pairs == 26
    assert kernel.indexer_pairs == 36

    prev, copies, nxt = context_split_prefill(kernel, N)

    assert isinstance(prev["q"], Scatter)
    assert isinstance(prev["kv"], Broadcast)
    assert isinstance(prev["index_q"], Scatter)
    assert isinstance(prev["index_kv"], Broadcast)
    assert isinstance(nxt["y"], Gather)
    assert sum(copy.flops for copy in copies) == kernel.flops
    assert sum(copy.selected_pairs for copy in copies) == 26
    assert sum(copy.indexer_pairs for copy in copies) == 36
    assert all(copy.S_q == 2 for copy in copies)


def test_glm52_context_split_decode_shards_both_caches():
    kernel = _make_glm52_sparse_attn(S_q=1, causal=False)

    prev, copies, nxt = context_split_decode(kernel, N)

    assert isinstance(prev["q"], Broadcast)
    assert isinstance(prev["kv"], Scatter)
    assert isinstance(prev["index_q"], Broadcast)
    assert isinstance(prev["index_kv"], Scatter)
    assert isinstance(nxt["y"], Reduce)
    assert sum(copy.flops for copy in copies) == kernel.flops
    assert all(copy.S_kv == 2 for copy in copies)
    assert all(copy.k_sel == 1 for copy in copies)
    assert all(copy.indexer_s_kv == 2 for copy in copies)


def test_glm52_batch_split_preserves_modes_dtypes_and_work():
    kernel = _make_glm52_sparse_attn(B=8, S_q=1, causal=False)

    _, copies, _ = batch_split(kernel, N)

    assert sum(copy.flops for copy in copies) == kernel.flops
    assert all(copy.B == 2 for copy in copies)
    assert all(copy.indexer_mode == "full" for copy in copies)
    assert all(copy.kv_dtype == "fp8" for copy in copies)
    assert all(copy.indexer_compute_dtype == "fp8" for copy in copies)
    assert all(copy.indexer_reduce_dtype == "fp32" for copy in copies)


@pytest.mark.parametrize("output_name", ["prefill_output", "decode_output"])
def test_kv_persistence_split_shards_kv_and_replicates_output(output_name):
    kernel = Nop(
        inputs={
            "kv": Tensor("bf16", (8, 16, 64)),
            output_name: Tensor("int32", (8, 1, 1)),
        },
        outputs={"done": Tensor("int32", (1,))},
    )
    prev, copies, nxt = kv_persistence_split(kernel, N)

    assert isinstance(prev["kv"], Scatter)
    assert isinstance(prev[output_name], Broadcast)
    assert all(copy.inputs["kv"].shape == (8, 4, 64)
               for copy in copies)
    assert all(copy.inputs[output_name].shape == (8, 1, 1)
               for copy in copies)
    assert all(copy.outputs["done"].shape == (1,) for copy in copies)
    assert isinstance(nxt["done"], Gather)


def test_optimize_comms_bypasses_rank_aligned_gather_scatter():
    gather = Gather(total_bytes=SHARD.size_bytes, world=N, dim=0)
    scatter = Scatter(total_bytes=SHARD.size_bytes, world=N, dim=0)
    graph, _, preds, succs = _build_chain(gather, scatter, N)
    optimize_comms(graph)

    assert gather not in graph.kernels
    assert scatter not in graph.kernels
    for pred, succ in zip(preds, succs):
        assert any(edge.src is pred and edge.dst is succ
                   for edge in graph._out_edges(pred))


def test_general_dup_builds_broadcast_and_identical_copies():
    kernel = Gemm(8, 16, 32, "bf16", "bf16")
    kernel.inputs = {"x": Tensor("bf16", (8, 32))}
    kernel.weights = {"w": Tensor("bf16", (32, 16))}
    kernel.outputs = {"y": Tensor("bf16", (8, 16))}

    broadcast, copies = general_dup(kernel, N)

    assert isinstance(broadcast, Broadcast)
    assert len(copies) == N
    assert all(copy.to_dict() == kernel.to_dict() for copy in copies)


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
