"""Unit tests for rooflang.language.graph (ComputeGraph)."""

import pytest

from rooflang.language.graph import ComputeGraph, DataEdge
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.kernels.comm import Broadcast
from rooflang.language.tensor import Tensor


T = Tensor("bf16", (4, 4))


def _k(ins=None, outs=None, side_effect=False):
    inputs = {k: Tensor("bf16", (4, 4)) for k in (ins or [])}
    outputs = {k: Tensor("bf16", (4, 4)) for k in (outs or [])}
    return Kernel(inputs=inputs, outputs=outputs, has_side_effect=side_effect)


class _Broadcast(Broadcast):
    def __init__(self, inputs, outputs):
        Kernel.__init__(self, inputs=inputs, outputs=outputs)


# ── Node API ─────────────────────────────────────────────────────────


class TestNodeAPI:
    def test_add_kernel(self):
        g = ComputeGraph()
        k = _k()
        g.add_kernel(k)
        assert k in g.kernels

    def test_add_duplicate_raises(self):
        g = ComputeGraph()
        k = _k()
        g.add_kernel(k)
        with pytest.raises(ValueError):
            g.add_kernel(k)

    def test_remove_kernel(self):
        g = ComputeGraph()
        k = _k()
        g.add_kernel(k)
        g.remove_kernel(k)
        assert k not in g.kernels

    def test_remove_nonexistent_raises(self):
        g = ComputeGraph()
        with pytest.raises(ValueError):
            g.remove_kernel(_k())


# ── Data Edge API ────────────────────────────────────────────────────


class TestDataEdgeAPI:
    def test_add_data_edge(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        edges = g._in_edges(b)
        assert len(edges) == 1
        assert edges[0].src is a
        assert edges[0].mapping == {"y": "x"}

    def test_shape_mismatch_raises(self):
        g = ComputeGraph()
        a = Kernel(outputs={"y": Tensor("bf16", (4, 4))})
        b = Kernel(inputs={"x": Tensor("bf16", (8, 8))})
        g.add_kernel(a)
        g.add_kernel(b)
        with pytest.raises(ValueError, match="Shape mismatch"):
            g.add_data_edge(a, b, {"y": "x"})

    def test_dtype_mismatch_raises(self):
        g = ComputeGraph()
        a = Kernel(outputs={"y": Tensor("bf16", (4, 4))})
        b = Kernel(inputs={"x": Tensor("fp32", (4, 4))})
        g.add_kernel(a)
        g.add_kernel(b)
        with pytest.raises(ValueError, match="Dtype mismatch"):
            g.add_data_edge(a, b, {"y": "x"})

    def test_empty_mapping_raises(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        with pytest.raises(ValueError):
            g.add_data_edge(a, b, {})

    def test_merge_data_edge(self):
        g = ComputeGraph()
        a = _k(outs=["y1", "y2"])
        b = _k(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y1": "x1"})
        g.add_data_edge(a, b, {"y2": "x2"})
        edges = g._out_edges(a)
        assert len(edges) == 1
        assert edges[0].mapping == {"y1": "x1", "y2": "x2"}

    def test_data_edge_replaces_control(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_control_edge(a, b)
        g.add_data_edge(a, b, {"y": "x"})
        edges = g._out_edges(a)
        assert len(edges) == 1
        assert edges[0].mapping == {"y": "x"}


# ── Control Edge API ─────────────────────────────────────────────────


class TestControlEdgeAPI:
    def test_add_control_edge(self):
        g = ComputeGraph()
        a, b = _k(), _k()
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_control_edge(a, b)
        assert g._dag.has_edge(a, b)
        assert g._dag.edges[a, b]["mapping"] == {}

    def test_add_control_noop_if_data_exists(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        g.add_control_edge(a, b)
        assert g._dag.edges[a, b]["mapping"] == {"y": "x"}

    def test_remove_control_edge(self):
        g = ComputeGraph()
        a, b = _k(), _k()
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_control_edge(a, b)
        g.remove_control_edge(a, b)
        assert not g._dag.has_edge(a, b)

    def test_remove_data_edge_as_control_raises(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        with pytest.raises(ValueError):
            g.remove_control_edge(a, b)


# ── Insert/Remove Identity ───────────────────────────────────────────


class TestInsertRemoveIdentity:
    def test_insert_identity(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        mid = _k(ins=["src"], outs=["dst"])
        g.insert_identity(mid, a, b, {"y": "x"})
        assert mid in g.kernels
        assert g._in_edges(mid)[0].src is a
        assert g._out_edges(mid)[0].dst is b

    def test_insert_identity_multi_output(self):
        g = ComputeGraph()
        a = _k(outs=["y1", "y2"])
        b = _k(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y1": "x1", "y2": "x2"})
        mid = _k(ins=["i1", "i2"], outs=["o1", "o2"])
        g.insert_identity(mid, a, b, {"y1": "x1", "y2": "x2"})
        assert g._in_edges(mid)[0].mapping == {"y1": "i1", "y2": "i2"}
        assert g._out_edges(mid)[0].mapping == {"o1": "x1", "o2": "x2"}

    def test_remove_identity(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"])
        mid = _k(ins=["src"], outs=["dst"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        g.insert_identity(mid, a, b, {"y": "x"})
        g.remove_identity(mid)
        assert mid not in g.kernels
        edges = g._out_edges(a)
        assert len(edges) == 1
        assert edges[0].dst is b
        assert edges[0].mapping == {"y": "x"}

    def test_remove_identity_not_single_edge_raises(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x1"])
        c = _k(ins=["x2"])
        mid = _k(ins=["src"], outs=["d1", "d2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_kernel(mid)
        g.add_data_edge(a, mid, {"y": "src"})
        g.add_data_edge(mid, b, {"d1": "x1"})
        g.add_data_edge(mid, c, {"d2": "x2"})
        with pytest.raises(ValueError):
            g.remove_identity(mid)


# ── Fuse Kernels ─────────────────────────────────────────────────────


class TestFuseKernels:
    def test_fuse_chain(self):
        g = ComputeGraph()
        a = _k(outs=["y"])
        b = _k(ins=["x"], outs=["z"])
        c = _k(ins=["w"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_data_edge(a, b, {"y": "x"})
        g.add_data_edge(b, c, {"z": "w"})

        def fuse_class(kl):
            return _k(ins=["x"], outs=["z"])

        fused = g.fuse_kernels(fuse_class, [a, b])
        assert a not in g.kernels
        assert b not in g.kernels
        assert fused in g.kernels
        out_e = g._out_edges(fused)
        assert len(out_e) == 1
        assert out_e[0].dst is c

    def test_fuse_forest(self):
        g = ComputeGraph()
        a = _k(outs=["y1"])
        b = _k(outs=["y2"])
        c = _k(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_data_edge(a, c, {"y1": "x1"})
        g.add_data_edge(b, c, {"y2": "x2"})

        def fuse_class(kl):
            return _k(outs=["y1", "y2"])

        fused = g.fuse_kernels(fuse_class, [a, b])
        assert fused in g.kernels
        out_e = g._out_edges(fused)
        assert len(out_e) == 1
        assert out_e[0].dst is c

    def test_fuse_less_than_2_raises(self):
        g = ComputeGraph()
        a = _k()
        g.add_kernel(a)
        with pytest.raises(ValueError):
            g.fuse_kernels(lambda kl: _k(), [a])

    def test_fuse_not_in_graph_raises(self):
        g = ComputeGraph()
        a = _k()
        b = _k()
        g.add_kernel(a)
        with pytest.raises(ValueError):
            g.fuse_kernels(lambda kl: _k(), [a, b])


# ── Split Kernel ─────────────────────────────────────────────────────


class TestSplitKernel:
    def _make_split_class(self):
        def split_class(kernel, n):
            prev = _k(
                ins=list(kernel.inputs),
                outs=[f"o{i}_{j}" for i in range(n)
                      for j in kernel.inputs],
            )
            copies = [_k(ins=list(kernel.inputs), outs=list(kernel.outputs))
                      for _ in range(n)]
            nxt = _k(
                ins=[f"i{i}_{j}" for i in range(n)
                     for j in kernel.outputs],
                outs=list(kernel.outputs),
            )
            return prev, copies, nxt
        return split_class

    def test_split_basic(self):
        g = ComputeGraph()
        pred = _k(outs=["y"])
        k = _k(ins=["x"], outs=["z"])
        succ = _k(ins=["w"])
        g.add_kernel(pred)
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(pred, k, {"y": "x"})
        g.add_data_edge(k, succ, {"z": "w"})

        prev_comm, copies, next_comm = g.split_kernel(
            self._make_split_class(), k, 2)
        assert k not in g.kernels
        assert prev_comm in g.kernels
        assert next_comm in g.kernels
        assert all(c in g.kernels for c in copies)
        assert len(copies) == 2

    def test_split_rewires_predecessors(self):
        g = ComputeGraph()
        pred = _k(outs=["y"])
        k = _k(ins=["x"], outs=["z"])
        g.add_kernel(pred)
        g.add_kernel(k)
        g.add_data_edge(pred, k, {"y": "x"})

        prev_comm, _, _ = g.split_kernel(self._make_split_class(), k, 2)
        in_e = g._in_edges(prev_comm)
        assert len(in_e) == 1
        assert in_e[0].src is pred

    def test_split_rewires_successors(self):
        g = ComputeGraph()
        k = _k(ins=["x"], outs=["z"])
        succ = _k(ins=["w"])
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(k, succ, {"z": "w"})

        _, _, next_comm = g.split_kernel(self._make_split_class(), k, 2)
        out_e = g._out_edges(next_comm)
        assert len(out_e) == 1
        assert out_e[0].dst is succ

    def test_split_count_mismatch_raises(self):
        g = ComputeGraph()
        k = _k(ins=["x"], outs=["z"])
        g.add_kernel(k)

        def bad_split(kernel, n):
            prev = _k(ins=["x"], outs=["o0_x"])
            copies = [_k(ins=["x"], outs=["z"])]
            nxt = _k(ins=["i0_z"], outs=["z"])
            return prev, copies, nxt

        with pytest.raises(ValueError, match="expected 2"):
            g.split_kernel(bad_split, k, 2)

    def test_split_none_comm_raises(self):
        g = ComputeGraph()
        k = _k(ins=["x"], outs=["z"])
        g.add_kernel(k)

        def bad_split(kernel, n):
            return None, [_k(), _k()], _k()

        with pytest.raises(AssertionError):
            g.split_kernel(bad_split, k, 2)
