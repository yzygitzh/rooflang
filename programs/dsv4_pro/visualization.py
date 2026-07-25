"""DeepSeek V4 Pro inference — Visualization Phase."""

from collections import deque
from typing import Set

from rooflang.language.kernels.kernel import Kernel
from rooflang.runtime.graph_export import export_graph


def _collect_kernels(obj) -> Set[Kernel]:
    """BFS over an object's attributes to find all Kernel instances."""
    found = set()
    queue = deque([obj])
    seen_ids = {id(obj)}
    while queue:
        cur = queue.popleft()
        if isinstance(cur, Kernel):
            found.add(cur)
        elif hasattr(cur, "__dict__"):
            for v in vars(cur).values():
                if id(v) not in seen_ids:
                    seen_ids.add(id(v))
                    queue.append(v)
        elif isinstance(cur, (list, tuple)):
            for item in cur:
                if id(item) not in seen_ids:
                    seen_ids.add(id(item))
                    queue.append(item)
    return found


def visualize_layer(g, layer_meta, extra_seeds=None,
                    path="dsv4_pro_graph_layer0.svg"):
    """Visualize a single layer's subgraph."""
    kernels = _collect_kernels(layer_meta)
    if extra_seeds:
        kernels.update(extra_seeds)
    # Include Spawn/Concat identity kernels adjacent to collected kernels
    frozen = frozenset(kernels)
    for k in g.topological_sort():
        if k in frozen:
            continue
        if type(k).__name__ not in ("Spawn", "Concat", "Slice"):
            continue
        neighbors = set(g._dag.predecessors(k)) | set(g._dag.successors(k))
        if neighbors & frozen:
            kernels.add(k)
    figsize = (max(48, len(kernels) // 8), 32) if len(kernels) > 50 else (24, 16)
    export_graph(g, path, kernels=kernels, figsize=figsize)
