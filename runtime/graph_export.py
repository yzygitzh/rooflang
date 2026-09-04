# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Export ComputeGraph visualization using networkx + matplotlib."""

from __future__ import annotations

from collections import deque
from typing import Optional, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel


def _kernel_label(kernel: Kernel) -> str:
    """Short human-readable label for a kernel node."""
    name = type(kernel).__name__
    attrs = []
    if hasattr(kernel, "M"):
        attrs.append(f"M={kernel.M}")
    if hasattr(kernel, "N") and not hasattr(kernel, "topk"):
        attrs.append(f"N={kernel.N}")
    if hasattr(kernel, "K"):
        attrs.append(f"K={kernel.K}")
    if hasattr(kernel, "V"):
        attrs.append(f"V={kernel.V}")
    if hasattr(kernel, "D") and not hasattr(kernel, "N"):
        attrs.append(f"D={kernel.D}")
    if attrs:
        return f"{name}\n{','.join(attrs)}"
    return name


def _layered_layout(dag: nx.DiGraph) -> dict:
    """Assign positions using topological layers (top-to-bottom).

    Wide layers (experts) use compact spacing; narrow layers (spine) are
    staggered horizontally so labels in the vertical chain don't overlap.
    """
    topo = list(nx.topological_sort(dag))
    layer_of = {}
    for node in topo:
        preds = list(dag.predecessors(node))
        if not preds:
            layer_of[node] = 0
        else:
            layer_of[node] = max(layer_of[p] for p in preds) + 1

    from collections import defaultdict
    layers = defaultdict(list)
    for node, layer in layer_of.items():
        layers[layer].append(node)

    max_width = max(len(nodes) for nodes in layers.values()) if layers else 1

    x_spacing = max(2.0, max_width * 0.15)

    pos = {}
    for layer, nodes in layers.items():
        n = len(nodes)
        x_stagger = (layer % 4 - 1.5) * 1.5
        for i, node in enumerate(nodes):
            x = (i - (n - 1) / 2.0) * x_spacing + x_stagger
            y_offset = (i % 4) * 1.2
            y = -layer * 5.0 - y_offset
            pos[node] = (x, y)
    return pos


def bfs_subgraph(
    graph: ComputeGraph,
    seeds: Set[Kernel],
    max_depth: int = -1,
) -> Set[Kernel]:
    """BFS from seed kernels, collecting all reachable nodes (undirected).

    Args:
        graph: The ComputeGraph.
        seeds: Starting kernels for BFS.
        max_depth: Max BFS hops (-1 for unlimited).

    Returns:
        Set of reachable kernels (including seeds).
    """
    dag = graph._dag
    visited = set(seeds)
    queue = deque((k, 0) for k in seeds)

    while queue:
        node, depth = queue.popleft()
        if 0 <= max_depth <= depth:
            continue
        for neighbor in list(dag.predecessors(node)) + list(dag.successors(node)):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    return visited


def export_graph(
    graph: ComputeGraph,
    path: str,
    kernels: Optional[Set[Kernel]] = None,
    figsize: tuple = (24, 16),
    show_edge_labels: bool = True,
) -> None:
    """Render ComputeGraph to an image file (PNG/SVG/PDF).

    Args:
        graph: The ComputeGraph to visualize.
        path: Output file path (extension determines format: .svg, .pdf, .png).
        kernels: If provided, only show this subset of kernels and edges
                 between them. Otherwise show the full graph.
        figsize: Figure size in inches.
        show_edge_labels: Whether to draw edge mapping labels.
    """
    dag = graph._dag

    if kernels is not None:
        sub = dag.subgraph(kernels).copy()
    else:
        sub = dag

    labels = {k: _kernel_label(k) for k in sub.nodes}
    pos = _layered_layout(sub)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    data_edges = [(u, v) for u, v, d in sub.edges(data=True) if d.get("mapping")]
    ctrl_edges = [(u, v) for u, v, d in sub.edges(data=True)
                  if not d.get("mapping")]

    node_size = 800 if sub.number_of_nodes() < 50 else 200
    font_size = 6 if sub.number_of_nodes() < 50 else 4

    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=node_size,
                           node_color="lightblue", edgecolors="black",
                           linewidths=0.5)
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax,
                            font_size=font_size)
    nx.draw_networkx_edges(sub, pos, edgelist=data_edges, ax=ax,
                           edge_color="black", arrows=True, arrowsize=6,
                           width=0.5)
    if ctrl_edges:
        nx.draw_networkx_edges(sub, pos, edgelist=ctrl_edges, ax=ax,
                               edge_color="gray", style="dashed",
                               arrows=True, arrowsize=5, width=0.3)

    if show_edge_labels and sub.number_of_nodes() < 100:
        edge_labels = {}
        for u, v, d in sub.edges(data=True):
            mapping = d.get("mapping")
            if mapping:
                lbl = ", ".join(f"{s}→{t}" for s, t in mapping.items())
                edge_labels[(u, v)] = lbl
        nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=5)

    ax.set_title(f"ComputeGraph ({sub.number_of_nodes()} nodes, "
                 f"{sub.number_of_edges()} edges)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph exported to {path}")
