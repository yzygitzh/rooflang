"""Unit tests for rooflang.runtime.graph_export."""

import os

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.forward import Gemm, RMSNorm
from rooflang.language.kernels.identity import Spawn, Concat
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.tensor import Tensor
from rooflang.runtime.graph_export import (
    _kernel_label,
    _layered_layout,
    bfs_subgraph,
    export_graph,
)


def _chain_graph(n=3):
    """Build a linear chain: k0 -> k1 -> ... -> k(n-1)."""
    g = ComputeGraph()
    kernels = []
    for i in range(n):
        k = RMSNorm(M=16, D=64)
        k.inputs = {"x": Tensor("bf16", (16, 64))}
        k.weights = {"g": Tensor("bf16", (64,))}
        k.outputs = {"y": Tensor("bf16", (16, 64))}
        g.add_kernel(k)
        kernels.append(k)
    for i in range(n - 1):
        g.add_data_edge(kernels[i], kernels[i + 1], {"y": "x"})
    return g, kernels


def _diamond_graph():
    """Build a diamond: src -> (a, b) -> dst."""
    g = ComputeGraph()
    src = Gemm(M=8, N=16, K=32, w_dtype="bf16", a_dtype="bf16")
    src.inputs = {"x": Tensor("bf16", (8, 32))}
    src.weights = {"w": Tensor("bf16", (32, 16))}
    src.outputs = {"y": Tensor("bf16", (8, 16)), "y2": Tensor("bf16", (8, 16))}
    g.add_kernel(src)

    a = RMSNorm(M=8, D=16)
    a.inputs = {"x": Tensor("bf16", (8, 16))}
    a.weights = {"g": Tensor("bf16", (16,))}
    a.outputs = {"y": Tensor("bf16", (8, 16))}
    g.add_kernel(a)

    b = RMSNorm(M=8, D=16)
    b.inputs = {"x": Tensor("bf16", (8, 16))}
    b.weights = {"g": Tensor("bf16", (16,))}
    b.outputs = {"y": Tensor("bf16", (8, 16))}
    g.add_kernel(b)

    dst = Concat()
    dst.inputs = {"a": Tensor("bf16", (8, 16)), "b": Tensor("bf16", (8, 16))}
    dst.outputs = {"y": Tensor("bf16", (16, 16))}
    g.add_kernel(dst)

    g.add_data_edge(src, a, {"y": "x"})
    g.add_data_edge(src, b, {"y2": "x"})
    g.add_data_edge(a, dst, {"y": "a"})
    g.add_data_edge(b, dst, {"y": "b"})
    return g, src, a, b, dst


# ── _kernel_label ───────────────────────────────────────────────────


class TestKernelLabel:
    def test_gemm_label(self):
        k = Gemm(M=32, N=64, K=128, w_dtype="bf16", a_dtype="bf16")
        label = _kernel_label(k)
        assert "Gemm" in label
        assert "M=32" in label
        assert "N=64" in label
        assert "K=128" in label

    def test_rmsnorm_label(self):
        k = RMSNorm(M=16, D=64)
        label = _kernel_label(k)
        assert "RMSNorm" in label
        assert "M=16" in label
        assert "D=64" in label

    def test_spawn_label(self):
        k = Spawn(world=2)
        label = _kernel_label(k)
        assert "Spawn" in label

    def test_bare_kernel_label(self):
        k = Kernel()
        label = _kernel_label(k)
        assert label == "Kernel"


# ── _layered_layout ─────────────────────────────────────────────────


class TestLayeredLayout:
    def test_chain_positions_descend(self):
        g, kernels = _chain_graph(4)
        pos = _layered_layout(g._dag)
        for i in range(3):
            assert pos[kernels[i]][1] > pos[kernels[i + 1]][1]

    def test_diamond_layers(self):
        g, src, a, b, dst = _diamond_graph()
        pos = _layered_layout(g._dag)
        assert pos[src][1] > pos[a][1]
        assert pos[src][1] > pos[b][1]
        assert pos[a][1] > pos[dst][1]
        assert pos[b][1] > pos[dst][1]

    def test_same_layer_different_x(self):
        g, src, a, b, dst = _diamond_graph()
        pos = _layered_layout(g._dag)
        assert pos[a][0] != pos[b][0]

    def test_stagger_across_layers(self):
        g, kernels = _chain_graph(5)
        pos = _layered_layout(g._dag)
        xs = [pos[k][0] for k in kernels]
        assert len(set(xs)) > 1

    def test_all_nodes_have_positions(self):
        g, kernels = _chain_graph(3)
        pos = _layered_layout(g._dag)
        for k in kernels:
            assert k in pos
            assert len(pos[k]) == 2


# ── bfs_subgraph ────────────────────────────────────────────────────


class TestBfsSubgraph:
    def test_single_seed_full_chain(self):
        g, kernels = _chain_graph(5)
        result = bfs_subgraph(g, {kernels[2]})
        assert result == set(kernels)

    def test_max_depth_limits_reach(self):
        g, kernels = _chain_graph(5)
        result = bfs_subgraph(g, {kernels[0]}, max_depth=1)
        assert kernels[0] in result
        assert kernels[1] in result
        assert kernels[2] not in result

    def test_diamond_from_src(self):
        g, src, a, b, dst = _diamond_graph()
        result = bfs_subgraph(g, {src}, max_depth=1)
        assert src in result
        assert a in result
        assert b in result
        assert dst not in result

    def test_empty_seeds(self):
        g, kernels = _chain_graph(3)
        result = bfs_subgraph(g, set())
        assert result == set()


# ── export_graph ────────────────────────────────────────────────────


class TestExportGraph:
    def test_export_svg(self, tmp_path):
        g, kernels = _chain_graph(3)
        path = str(tmp_path / "graph.svg")
        export_graph(g, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_export_png(self, tmp_path):
        g, kernels = _chain_graph(3)
        path = str(tmp_path / "graph.png")
        export_graph(g, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_export_with_kernel_subset(self, tmp_path):
        g, kernels = _chain_graph(5)
        subset = {kernels[0], kernels[1], kernels[2]}
        path = str(tmp_path / "sub.svg")
        export_graph(g, path, kernels=subset)
        assert os.path.exists(path)

    def test_export_diamond(self, tmp_path):
        g, src, a, b, dst = _diamond_graph()
        path = str(tmp_path / "diamond.svg")
        export_graph(g, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_no_edge_labels_for_large_graph(self, tmp_path):
        g, kernels = _chain_graph(3)
        path = str(tmp_path / "nolabels.svg")
        export_graph(g, path, show_edge_labels=False)
        assert os.path.exists(path)

    def test_empty_graph(self, tmp_path):
        g = ComputeGraph()
        path = str(tmp_path / "empty.svg")
        export_graph(g, path)
        assert os.path.exists(path)

    def test_single_node(self, tmp_path):
        g = ComputeGraph()
        k = Gemm(M=16, N=32, K=64, w_dtype="bf16", a_dtype="bf16")
        k.inputs = {"x": Tensor("bf16", (16, 64))}
        k.weights = {"w": Tensor("bf16", (64, 32))}
        k.outputs = {"y": Tensor("bf16", (16, 32))}
        g.add_kernel(k)
        path = str(tmp_path / "single.svg")
        export_graph(g, path)
        assert os.path.exists(path)

    def test_custom_figsize(self, tmp_path):
        g, kernels = _chain_graph(3)
        path = str(tmp_path / "custom.svg")
        export_graph(g, path, figsize=(12, 8))
        assert os.path.exists(path)
