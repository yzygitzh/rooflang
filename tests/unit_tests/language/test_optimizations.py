"""Unit tests for rooflang.language.optimization (optimize_comms)."""

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import (
    AllGather, AllReduce, AllToAll, Broadcast, CommKernel, Gather, Reduce,
    ReduceScatter, Scatter, Send, Recv,
)
from rooflang.language.tensor import Tensor
from rooflang.language.placement import Placement
from rooflang.language.hardware.component import Compute
from rooflang.language.optimization.comm import optimize_comms


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
        placement.set(p, gpus[i])
    if same_devices:
        for i, s in enumerate(succs):
            placement.set(s, gpus[i])
    else:
        other_gpus = [Compute(name=f"other{i}") for i in range(n)]
        for i, s in enumerate(succs):
            placement.set(s, other_gpus[i])

    return g, placement, preds, succs


class TestFuseWorld4:
    def test_reduce_broadcast_to_allreduce(self):
        r = Reduce(bytes_per_rank=32, world=4, dtype="bf16")
        b = Broadcast(bytes_per_rank=32, world=4)
        g, p, preds, succs = _build_chain(r, b, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllReduce)

    def test_reduce_scatter_to_reducescatter(self):
        r = Reduce(bytes_per_rank=32, world=4, dtype="bf16")
        s = Scatter(bytes_per_rank=32, world=4, dim=1)
        g, p, preds, succs = _build_chain(r, s, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), ReduceScatter)

    def test_gather_broadcast_to_allgather(self):
        ga = Gather(bytes_per_rank=128, world=4, dim=0)
        b = Broadcast(bytes_per_rank=128, world=4)
        g, p, preds, succs = _build_chain(ga, b, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllGather)

    def test_gather_scatter_diff_dim_to_alltoall(self):
        ga = Gather(bytes_per_rank=128, world=4, dim=0)
        s = Scatter(bytes_per_rank=128, world=4, dim=1)
        g, p, preds, succs = _build_chain(ga, s, n=4)
        optimize_comms(g, p)
        comms = g.kernels - set(preds) - set(succs)
        assert len(comms) == 1
        assert isinstance(next(iter(comms)), AllToAll)


class TestFuseWorld1Eliminated:
    def test_reduce_broadcast_world1_eliminated(self):
        r = Reduce(bytes_per_rank=32, world=1, dtype="bf16")
        b = Broadcast(bytes_per_rank=32, world=1)
        g, p, preds, succs = _build_chain(r, b, n=1)
        optimize_comms(g, p)
        assert g.kernels == frozenset(preds + succs)
        assert g._out_edges(preds[0])[0].dst is succs[0]

    def test_gather_scatter_same_dim_world1_eliminated(self):
        ga = Gather(bytes_per_rank=32, world=1, dim=0)
        s = Scatter(bytes_per_rank=32, world=1, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g, p)
        assert g.kernels == frozenset(preds + succs)

    def test_gather_broadcast_world1_eliminated(self):
        ga = Gather(bytes_per_rank=32, world=1, dim=0)
        b = Broadcast(bytes_per_rank=32, world=1)
        g, p, preds, succs = _build_chain(ga, b, n=1)
        optimize_comms(g, p)
        assert g.kernels == frozenset(preds + succs)


class TestBypass:
    def test_gather_scatter_same_dim_bypass(self):
        ga = Gather(bytes_per_rank=128, world=4, dim=0)
        s = Scatter(bytes_per_rank=128, world=4, dim=0)
        g, p, preds, succs = _build_chain(ga, s, n=1)
        optimize_comms(g, p)
        assert ga not in g.kernels
        assert s not in g.kernels
        assert g._out_edges(preds[0])[0].dst is succs[0]


class TestNoFusionDiffDevices:
    def test_diff_devices_no_fusion(self):
        r = Reduce(bytes_per_rank=32, world=4, dtype="bf16")
        b = Broadcast(bytes_per_rank=32, world=4)
        g, p, preds, succs = _build_chain(r, b, n=4, same_devices=False)
        optimize_comms(g, p)
        assert r in g.kernels
        assert b in g.kernels


class TestEliminateDead:
    @pytest.mark.parametrize("comm_cls,kwargs", [
        (Scatter, dict(bytes_per_rank=128, world=4, dim=0)),
        (Broadcast, dict(bytes_per_rank=32, world=4)),
        (Gather, dict(bytes_per_rank=128, world=4, dim=0)),
        (Reduce, dict(bytes_per_rank=32, world=4, dtype="bf16")),
        (AllReduce, dict(bytes_per_rank=32, world=4, dtype="bf16")),
        (AllGather, dict(bytes_per_rank=128, world=4)),
        (ReduceScatter, dict(bytes_per_rank=32, world=4, dtype="bf16")),
        (AllToAll, dict(bytes_per_rank=128, world=4)),
        (Send, dict(bytes_total=32)),
        (Recv, dict(bytes_total=32)),
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
        p.set(pred, gpu)
        p.set(succ, gpu)
        optimize_comms(g, p)
        assert comm not in g.kernels
        assert g._out_edges(pred)[0].dst is succ

    def test_multi_edge_not_eliminated(self):
        g = ComputeGraph()
        gpu = Compute(name="gpu0")
        pred = Kernel(outputs={"y": SHARD})
        comm = Scatter(bytes_per_rank=128, world=4, dim=0)
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
        p.set(pred, gpu)
        p.set(succ0, gpu)
        p.set(succ1, gpu)
        optimize_comms(g, p)
        assert comm in g.kernels
