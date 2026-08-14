"""Communication optimization pass — fuse primitive pairs into collectives.

Recognizes (Gather|Reduce) → (Scatter|Broadcast) pairs introduced by
split_kernel and rewrites them into single fused collectives:

    Gather  + Scatter (same dim)  → identity (eliminate both)
    Gather  + Scatter (diff dim)  → AllToAll
    Gather  + Broadcast           → AllGather
    Reduce  + Scatter             → ReduceScatter
    Reduce  + Broadcast           → AllReduce

Preconditions for fusion:
  - The collector's sole data successor is the distributor.
  - The distributor's sole data predecessor is the collector.
  - Both have the same world.
  - Their outer rank ports are complete and the connecting tensor is compatible.

This pass is a pure data-dependency graph rewrite and intentionally runs before
placement. A same-dimension Gather → Scatter is rewired rank by rank; later
placement may keep that edge local or model it as a remote read.

Also eliminates dead communication nodes (single-edge Broadcast/Scatter/
Gather/Reduce left after fuse_kernels).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from rooflang.language.kernels.comm import (
    AllGather, AllReduce, AllToAll, Broadcast, CommKernel, Gather, Reduce,
    ReduceScatter, Scatter,
)
from rooflang.language.tensor import Tensor

if TYPE_CHECKING:
    from rooflang.language.graph import ComputeGraph
    from rooflang.language.kernels.kernel import Kernel


def optimize_comms(graph: ComputeGraph) -> None:
    """Fuse primitive comm pairs and eliminate dead comms before placement."""
    changed = True
    while changed:
        changed = _fuse_pairs(graph)
        changed |= _eliminate_dead(graph)


def _create_collective(collector: Kernel, distributor: Kernel) -> Kernel | None:
    """Create a fused collective from a (collector, distributor) pair.

    Returns None for Gather+Scatter same-dim (identity case).
    """
    bpr = collector.total_bytes
    world = collector.world

    if isinstance(collector, Reduce) and isinstance(distributor, Broadcast):
        dtype = getattr(collector, "dtype_", "bf16")
        return AllReduce(total_bytes=bpr, world=world, dtype=dtype)
    elif isinstance(collector, Reduce) and isinstance(distributor, Scatter):
        dtype = getattr(collector, "dtype_", "bf16")
        return ReduceScatter(total_bytes=bpr, world=world, dtype=dtype)
    elif isinstance(collector, Gather) and isinstance(distributor, Broadcast):
        return AllGather(total_bytes=bpr, world=world)
    elif isinstance(collector, Gather) and isinstance(distributor, Scatter):
        if collector.dim == distributor.dim:
            return None
        return AllToAll(total_bytes=bpr, world=world)
    raise ValueError(f"Unexpected pair: {type(collector)}, {type(distributor)}")


def _single_data_neighbor(adjacency) -> Kernel | None:
    """Return the sole neighbor connected by data, without edge wrappers."""
    result = None
    for neighbor, attributes in adjacency.items():
        if not attributes["mapping"]:
            continue
        if result is not None:
            return None
        result = neighbor
    return result


def _fuse_pairs(graph: ComputeGraph) -> bool:
    """Fuse adjacent (Gather|Reduce) → (Scatter|Broadcast) pairs."""
    did_change = False
    while True:
        pairs = []
        for kernel in list(graph._dag.nodes):
            if not isinstance(kernel, (Gather, Reduce)):
                continue
            successor = _single_data_neighbor(graph._dag.succ[kernel])
            if successor is None:
                continue
            if not isinstance(successor, (Scatter, Broadcast)):
                continue
            if _single_data_neighbor(graph._dag.pred[successor]) is not kernel:
                continue
            if kernel.world != successor.world:
                continue
            if len(kernel.inputs) != kernel.world \
                    or len(successor.outputs) != successor.world:
                continue
            pairs.append((kernel, successor))

        if not pairs:
            break

        round_changed = False
        for kernel, successor in pairs:
            if not graph._dag.has_node(kernel) or not graph._dag.has_node(successor):
                continue
            collective = _create_collective(kernel, successor)
            if collective is None:
                collector_shard = next(iter(kernel.inputs.values())).shape
                distributor_shard = next(iter(successor.outputs.values())).shape
                if collector_shard != distributor_shard:
                    continue
                _bypass_pair(graph, kernel, successor)
            else:
                _replace_pair(graph, kernel, successor, collective)
            round_changed = True

        did_change |= round_changed
        if not round_changed:
            break

    return did_change


def _replace_pair(
    graph: ComputeGraph, collector: Kernel, distributor: Kernel,
    collective: Kernel,
) -> None:
    """Replace (collector, distributor) with a fused collective."""
    collective.inputs = {
        name: Tensor(tensor.dtype, tensor.shape)
        for name, tensor in collector.inputs.items()
    }
    collective.outputs = {
        name: Tensor(tensor.dtype, tensor.shape)
        for name, tensor in distributor.outputs.items()
    }
    graph.add_kernel(collective)

    for edge in graph._in_edges(collector):
        graph.add_data_edge(edge.src, collective, edge.mapping)
    for edge in graph._out_edges(distributor):
        graph.add_data_edge(collective, edge.dst, edge.mapping)

    graph.remove_kernel(collector)
    graph.remove_kernel(distributor)


def _bypass_pair(
    graph: ComputeGraph,
    collector: Kernel, distributor: Kernel,
) -> None:
    """Eliminate a Gather+Scatter same-dim pair (identity — no comm needed).

    Wires each predecessor directly to the corresponding successor by rank
    index: the positional index of the Gather input port maps to the same
    positional index of the Scatter output port.
    """
    in_edges = graph._in_edges(collector)
    out_edges = graph._out_edges(distributor)

    collector_in_keys = list(collector.inputs.keys())
    distributor_out_keys = list(distributor.outputs.keys())
    collector_rank = {
        input_name: rank
        for rank, input_name in enumerate(collector_in_keys)
    }
    destination_by_output = {
        output_name: (edge.dst, input_name)
        for edge in out_edges
        for output_name, input_name in edge.mapping.items()
    }

    for ie in in_edges:
        for src_out, coll_in in ie.mapping.items():
            rank_idx = collector_rank[coll_in]
            dist_out = distributor_out_keys[rank_idx]
            destination = destination_by_output.get(dist_out)
            if destination is not None:
                dst, dst_input = destination
                graph.add_data_edge(ie.src, dst, {src_out: dst_input})

    graph.remove_kernel(collector)
    graph.remove_kernel(distributor)


def _eliminate_dead(graph: ComputeGraph) -> bool:
    """Remove trivial comm nodes (single in-edge and single out-edge)."""
    did_change = False
    queue = deque(
        kernel for kernel in graph._dag.nodes
        if isinstance(kernel, CommKernel)
    )
    pending = set(queue)
    while queue:
        kernel = queue.popleft()
        pending.discard(kernel)
        if not graph._dag.has_node(kernel):
            continue
        predecessor = _single_data_neighbor(graph._dag.pred[kernel])
        successor = _single_data_neighbor(graph._dag.succ[kernel])
        if predecessor is None or successor is None:
            continue
        neighbors = (predecessor, successor)
        graph.remove_identity(kernel)
        did_change = True
        for neighbor in neighbors:
            if isinstance(neighbor, CommKernel) \
                    and neighbor not in pending \
                    and graph._dag.has_node(neighbor):
                queue.append(neighbor)
                pending.add(neighbor)

    return did_change
