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
  - The device set of the collector's predecessors matches the device set
    of the distributor's successors (checked via Placement).

Also eliminates dead communication nodes (single-edge Broadcast/Scatter/
Gather/Reduce left after fuse_kernels).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rooflang.language.kernels.comm import (
    AllGather, AllReduce, AllToAll, Broadcast, CommKernel, Gather, Reduce,
    ReduceScatter, Scatter,
)

if TYPE_CHECKING:
    from rooflang.language.graph import ComputeGraph
    from rooflang.language.kernels.kernel import Kernel
    from rooflang.language.placement import Placement


def optimize_comms(graph: ComputeGraph, placement: Placement) -> None:
    """Fuse primitive comm pairs into collectives and eliminate dead comms.

    Mutates graph in place. Reads placement for device-set checks only.
    """
    changed = True
    while changed:
        changed = _fuse_pairs(graph, placement)
        changed |= _eliminate_dead(graph)


def _same_device_set(graph: ComputeGraph, collector: Kernel,
                     distributor: Kernel, placement: Placement) -> bool:
    """Check if collector's predecessors and distributor's successors share the same devices."""
    preds = sorted((placement.get_kernel_device(e.src).device for e in graph._in_edges(collector)), key=id)
    succs = sorted((placement.get_kernel_device(e.dst).device for e in graph._out_edges(distributor)), key=id)
    if len(preds) != len(succs):
        return False
    return all(a is b for a, b in zip(preds, succs))


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


def _fuse_pairs(graph: ComputeGraph, placement: Placement) -> bool:
    """Fuse adjacent (Gather|Reduce) → (Scatter|Broadcast) pairs."""
    changed = True
    did_change = False
    while changed:
        changed = False
        for kernel in list(graph.kernels):
            if not isinstance(kernel, (Gather, Reduce)):
                continue
            out_edges = graph._out_edges(kernel)
            if len(out_edges) != 1:
                continue
            successor = out_edges[0].dst
            if not isinstance(successor, (Scatter, Broadcast)):
                continue
            in_edges_of_succ = graph._in_edges(successor)
            if len(in_edges_of_succ) != 1:
                continue
            if in_edges_of_succ[0].src is not kernel:
                continue
            if kernel.world != successor.world:
                continue
            if not _same_device_set(graph, kernel, successor, placement):
                continue

            collective = _create_collective(kernel, successor)

            if collective is None:
                # Identity case — verify shard shapes match before bypass
                collector_shard = next(iter(kernel.inputs.values())).shape
                distributor_shard = next(iter(successor.outputs.values())).shape
                if collector_shard != distributor_shard:
                    continue
                _bypass_pair(graph, kernel, successor)
            else:
                _replace_pair(graph, kernel, successor, collective)
            changed = True
            did_change = True
            break
    return did_change


def _replace_pair(
    graph: ComputeGraph,
    collector: Kernel, distributor: Kernel, collective: Kernel,
) -> None:
    """Replace (collector, distributor) with a fused collective."""
    collective.inputs = dict(collector.inputs)
    collective.outputs = dict(distributor.outputs)
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

    for ie in in_edges:
        for src_out, coll_in in ie.mapping.items():
            rank_idx = collector_in_keys.index(coll_in)
            dist_out = distributor_out_keys[rank_idx]
            for oe in out_edges:
                if dist_out in oe.mapping:
                    graph.add_data_edge(ie.src, oe.dst,
                                        {src_out: oe.mapping[dist_out]})
                    break

    graph.remove_kernel(collector)
    graph.remove_kernel(distributor)


def _eliminate_dead(graph: ComputeGraph) -> bool:
    """Remove trivial comm nodes (single in-edge and single out-edge)."""
    changed = True
    did_change = False
    while changed:
        changed = False
        for kernel in list(graph.kernels):
            if not isinstance(kernel, CommKernel):
                continue
            in_edges = graph._in_edges(kernel)
            out_edges = graph._out_edges(kernel)
            if len(in_edges) == 1 and len(out_edges) == 1:
                graph.remove_identity(kernel)
                changed = True
                did_change = True
                break
    return did_change
