"""Unit tests for rooflang.language.graph (ComputeGraph)."""

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor


def make_kernel(ins=None, outs=None, side_effect=False):
    inputs = {k: Tensor("bf16", (4, 4)) for k in (ins or [])}
    outputs = {k: Tensor("bf16", (4, 4)) for k in (outs or [])}
    return Kernel(inputs=inputs, outputs=outputs, has_side_effect=side_effect)


# ── Node API ─────────────────────────────────────────────────────────


class TestAddKernel:
    def test_add(self):
        g = ComputeGraph()
        k = make_kernel()
        g.add_kernel(k)
        assert k in g.kernels

    def test_duplicate_raises(self):
        g = ComputeGraph()
        k = make_kernel()
        g.add_kernel(k)
        with pytest.raises(ValueError):
            g.add_kernel(k)


class TestRemoveKernel:
    def test_remove(self):
        g = ComputeGraph()
        k = make_kernel()
        g.add_kernel(k)
        g.remove_kernel(k)
        assert k not in g.kernels

    def test_nonexistent_raises(self):
        g = ComputeGraph()
        with pytest.raises(ValueError):
            g.remove_kernel(make_kernel())


class TestKernelsProperty:
    def test_returns_frozenset(self):
        g = ComputeGraph()
        a = make_kernel()
        b = make_kernel()
        g.add_kernel(a)
        g.add_kernel(b)
        assert g.kernels == frozenset([a, b])


# ── Data Edge API ────────────────────────────────────────────────────


class TestAddDataEdge:
    def test_basic(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
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
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        with pytest.raises(ValueError):
            g.add_data_edge(a, b, {})

    def test_merge_existing_edge(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y1", "y2"])
        b = make_kernel(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y1": "x1"})
        g.add_data_edge(a, b, {"y2": "x2"})
        edges = g._out_edges(a)
        assert len(edges) == 1
        assert edges[0].mapping == {"y1": "x1", "y2": "x2"}

    def test_replaces_control_edge(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_control_edge(a, b)
        g.add_data_edge(a, b, {"y": "x"})
        edges = g._out_edges(a)
        assert len(edges) == 1
        assert edges[0].mapping == {"y": "x"}


# ── Control Edge API ─────────────────────────────────────────────────


class TestAddControlEdge:
    def test_basic(self):
        g = ComputeGraph()
        a, b = make_kernel(), make_kernel()
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_control_edge(a, b)
        assert g._dag.has_edge(a, b)
        assert g._dag.edges[a, b]["mapping"] == {}

    def test_noop_if_data_exists(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        g.add_control_edge(a, b)
        assert g._dag.edges[a, b]["mapping"] == {"y": "x"}


class TestRemoveControlEdge:
    def test_basic(self):
        g = ComputeGraph()
        a, b = make_kernel(), make_kernel()
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_control_edge(a, b)
        g.remove_control_edge(a, b)
        assert not g._dag.has_edge(a, b)

    def test_data_edge_raises(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        with pytest.raises(ValueError):
            g.remove_control_edge(a, b)


# ── Mutation: insert_identity ─────────────────────────────────────────


class TestInsertIdentity:
    def test_basic(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        mid = make_kernel(ins=["src"], outs=["dst"])
        g.insert_identity(mid, a, b, {"y": "x"})
        assert mid in g.kernels
        assert g._in_edges(mid)[0].src is a
        assert g._out_edges(mid)[0].dst is b

    def test_multi_output(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y1", "y2"])
        b = make_kernel(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y1": "x1", "y2": "x2"})
        mid = make_kernel(ins=["i1", "i2"], outs=["o1", "o2"])
        g.insert_identity(mid, a, b, {"y1": "x1", "y2": "x2"})
        assert g._in_edges(mid)[0].mapping == {"y1": "i1", "y2": "i2"}
        assert g._out_edges(mid)[0].mapping == {"o1": "x1", "o2": "x2"}

    def test_partial_mapping(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y1", "y2"])
        b = make_kernel(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y1": "x1", "y2": "x2"})
        mid = make_kernel(ins=["src"], outs=["dst"])
        g.insert_identity(mid, a, b, {"y1": "x1"})
        assert g._in_edges(mid)[0].mapping == {"y1": "src"}
        assert g._out_edges(mid)[0].mapping == {"dst": "x1"}
        remaining = g._dag.edges[a, b]["mapping"]
        assert remaining == {"y2": "x2"}

    def test_mapping_size_mismatch_raises(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y1", "y2"])
        b = make_kernel(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y1": "x1", "y2": "x2"})
        mid = make_kernel(ins=["src"], outs=["dst"])
        with pytest.raises(ValueError, match="mapping has 2 entries"):
            g.insert_identity(mid, a, b, {"y1": "x1", "y2": "x2"})


# ── Mutation: remove_identity ─────────────────────────────────────────


class TestRemoveIdentity:
    def test_basic(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        mid = make_kernel(ins=["src"], outs=["dst"])
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

    def test_not_single_edge_raises(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x1"])
        c = make_kernel(ins=["x2"])
        mid = make_kernel(ins=["src"], outs=["d1", "d2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_kernel(mid)
        g.add_data_edge(a, mid, {"y": "src"})
        g.add_data_edge(mid, b, {"d1": "x1"})
        g.add_data_edge(mid, c, {"d2": "x2"})
        with pytest.raises(ValueError):
            g.remove_identity(mid)


# ── Mutation: fuse_kernels ────────────────────────────────────────────


class TestFuseKernels:
    def test_chain(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"], outs=["z"])
        c = make_kernel(ins=["w"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_data_edge(a, b, {"y": "x"})
        g.add_data_edge(b, c, {"z": "w"})

        fused = g.fuse_kernels(lambda kl: make_kernel(ins=["x"], outs=["z"]),
                               [a, b])
        assert a not in g.kernels and b not in g.kernels
        assert fused in g.kernels
        assert g._out_edges(fused)[0].dst is c

    def test_forest(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y1"])
        b = make_kernel(outs=["y2"])
        c = make_kernel(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_data_edge(a, c, {"y1": "x1"})
        g.add_data_edge(b, c, {"y2": "x2"})

        fused = g.fuse_kernels(lambda kl: make_kernel(outs=["y1", "y2"]),
                               [a, b])
        assert fused in g.kernels
        assert g._out_edges(fused)[0].dst is c

    def test_less_than_2_raises(self):
        g = ComputeGraph()
        a = make_kernel()
        g.add_kernel(a)
        with pytest.raises(ValueError):
            g.fuse_kernels(lambda kl: make_kernel(), [a])

    def test_not_in_graph_raises(self):
        g = ComputeGraph()
        a = make_kernel()
        b = make_kernel()
        g.add_kernel(a)
        with pytest.raises(ValueError):
            g.fuse_kernels(lambda kl: make_kernel(), [a, b])


# ── Mutation: split_kernel ────────────────────────────────────────────


class TestSplitKernel:
    def _make_split_class(self):
        def split_class(kernel, n):
            prev = make_kernel(
                ins=list(kernel.inputs),
                outs=[f"o{i}_{j}" for i in range(n) for j in kernel.inputs],
            )
            copies = [make_kernel(ins=list(kernel.inputs),
                                  outs=list(kernel.outputs))
                      for _ in range(n)]
            next = make_kernel(
                ins=[f"i{i}_{j}" for i in range(n) for j in kernel.outputs],
                outs=list(kernel.outputs),
            )
            return prev, copies, next
        return split_class

    def test_basic(self):
        g = ComputeGraph()
        pred = make_kernel(outs=["y"])
        k = make_kernel(ins=["x"], outs=["z"])
        succ = make_kernel(ins=["w"])
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

    def test_rewires_predecessors_and_successors(self):
        g = ComputeGraph()
        pred = make_kernel(outs=["y"])
        k = make_kernel(ins=["x"], outs=["z"])
        succ = make_kernel(ins=["w"])
        g.add_kernel(pred)
        g.add_kernel(k)
        g.add_kernel(succ)
        g.add_data_edge(pred, k, {"y": "x"})
        g.add_data_edge(k, succ, {"z": "w"})

        prev_comm, _, next_comm = g.split_kernel(
            self._make_split_class(), k, 2)
        assert g._in_edges(prev_comm)[0].src is pred
        assert g._out_edges(next_comm)[0].dst is succ

    def test_count_mismatch_raises(self):
        g = ComputeGraph()
        k = make_kernel(ins=["x"], outs=["z"])
        g.add_kernel(k)

        def bad_split(kernel, n):
            prev = make_kernel(ins=["x"], outs=["o0_x"])
            copies = [make_kernel(ins=["x"], outs=["z"])]
            next = make_kernel(ins=["i0_z"], outs=["z"])
            return prev, copies, next

        with pytest.raises(ValueError, match="expected 2"):
            g.split_kernel(bad_split, k, 2)

    def test_none_comm_raises(self):
        g = ComputeGraph()
        k = make_kernel(ins=["x"], outs=["z"])
        g.add_kernel(k)
        with pytest.raises(AssertionError):
            g.split_kernel(lambda ker, n: (None, [make_kernel()]*n, make_kernel()),
                           k, 2)


# ── Query: topological_sort ───────────────────────────────────────────


class TestTopologicalSort:
    def test_linear(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"], outs=["z"])
        c = make_kernel(ins=["w"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_data_edge(a, b, {"y": "x"})
        g.add_data_edge(b, c, {"z": "w"})
        assert g.topological_sort() == [a, b, c]

    def test_diamond(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y1", "y2"])
        b = make_kernel(ins=["x1"], outs=["z1"])
        c = make_kernel(ins=["x2"], outs=["z2"])
        d = make_kernel(ins=["w1", "w2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_kernel(d)
        g.add_data_edge(a, b, {"y1": "x1"})
        g.add_data_edge(a, c, {"y2": "x2"})
        g.add_data_edge(b, d, {"z1": "w1"})
        g.add_data_edge(c, d, {"z2": "w2"})
        order = g.topological_sort()
        assert order[0] is a
        assert order[-1] is d


# ── Validation: validate ──────────────────────────────────────────────


class TestValidate:
    def test_valid_graph(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        g.validate()

    def test_cycle_raises(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"], outs=["z"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x"})
        g._dag.add_edge(b, a, mapping={"z": "y"})
        with pytest.raises(ValueError, match="cycle"):
            g.validate()

    def test_output_connected_to_multiple_raises(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x"])
        c = make_kernel(ins=["w"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_kernel(c)
        g.add_data_edge(a, b, {"y": "x"})
        g.add_data_edge(a, c, {"y": "w"})
        with pytest.raises(ValueError, match="connected to multiple"):
            g.validate()

    def test_input_not_connected_raises(self):
        g = ComputeGraph()
        a = make_kernel(outs=["y"])
        b = make_kernel(ins=["x1", "x2"])
        g.add_kernel(a)
        g.add_kernel(b)
        g.add_data_edge(a, b, {"y": "x1"})
        with pytest.raises(ValueError, match="not connected"):
            g.validate()
