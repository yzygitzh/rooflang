"""Communication optimization pass — fuse primitive pairs into collectives.

Recognizes (Gather|Reduce) → (Scatter|Broadcast) pairs introduced by
split_kernel and rewrites them into single fused collectives:

    Gather  + Scatter (same dim)  → identity (eliminate both)
    Gather  + Scatter (diff dim)  → AllToAll
    Gather  + Broadcast           → AllGather
    Reduce  + Scatter             → ReduceScatter
    Reduce  + Broadcast           → AllReduce

Preconditions for intra-device-set fusion:
  - The collector's sole data successor is the distributor.
  - The distributor's sole data predecessor is the collector.
  - Both have the same world.
  - The device set of the collector's predecessors matches the device set
    of the distributor's successors (checked via Placement).

Any adjacent communication kernels with different device sets use an explicit
whitelist.  Currently only a same-dimension Gather → Scatter pair is supported;
corresponding ranks are wired directly when their tensors share a memory, with
a Move inserted otherwise.  Other cross-device-set pairs raise instead of
silently omitting the group-to-group transfer.

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
from rooflang.language.kernels.identity import Move
from rooflang.language.tensor import Tensor

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


def canonicalize_split_comms(graph: ComputeGraph, axis: str) -> None:
    """Eliminate rank-aligned Gather/Scatter artifacts for one split axis.

    Sequentially splitting adjacent kernels wraps both sides of an otherwise
    local edge. Unlike optimize_comms(), this rewrite is placement-free and is
    restricted to communication kernels explicitly tagged with the same
    logical parallel axis by the caller.
    """
    while True:
        pairs = []
        for collector in list(graph._dag.nodes):
            if not isinstance(collector, Gather) \
                    or getattr(collector, "_split_axis", None) != axis:
                continue
            out_edges = graph._out_edges(collector)
            if len(out_edges) != 1:
                continue
            distributor = out_edges[0].dst
            if not isinstance(distributor, Scatter) \
                    or getattr(distributor, "_split_axis", None) != axis:
                continue
            if len(graph._in_edges(distributor)) != 1 \
                    or collector.world != distributor.world \
                    or collector.dim != distributor.dim:
                continue
            collector_shard = next(iter(collector.inputs.values())).shape
            distributor_shard = next(
                iter(distributor.outputs.values())).shape
            if collector_shard != distributor_shard:
                continue
            pairs.append((collector, distributor))

        if not pairs:
            break
        for collector, distributor in pairs:
            if not graph._dag.has_node(collector) \
                    or not graph._dag.has_node(distributor):
                continue
            _bypass_pair(graph, collector, distributor)


def _same_device_set(
    collector: Kernel, distributor: Kernel, placement: Placement,
    device_sets=None,
) -> bool:
    """Check whether two comm kernels use the same participant devices."""
    if device_sets is None:
        device_sets = {}
    for kernel in (collector, distributor):
        if kernel not in device_sets:
            device_sets[kernel] = frozenset(
                placement.infer_comm_devices(kernel))
    return device_sets[collector] == device_sets[distributor]


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
    did_change = False
    device_sets = {}
    while True:
        pairs = []
        for kernel in list(graph._dag.nodes):
            out_edges = graph._out_edges(kernel)

            # Check every adjacent pair of communication kernels before any
            # ordinary fusion guards, so cross-device traffic cannot be
            # silently skipped by the more specific fusion patterns below.
            if isinstance(kernel, CommKernel):
                for edge in out_edges:
                    successor = edge.dst
                    if not isinstance(successor, CommKernel):
                        continue
                    if not _same_device_set(
                            kernel, successor, placement, device_sets):
                        pairs.append((kernel, successor, True))

            if not isinstance(kernel, (Gather, Reduce)):
                continue
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
            if not _same_device_set(
                    kernel, successor, placement, device_sets):
                continue
            pairs.append((kernel, successor, False))

        if not pairs:
            break

        round_changed = False
        for kernel, successor, cross_device_bypass in pairs:
            if not graph._dag.has_node(kernel) or not graph._dag.has_node(successor):
                continue
            if cross_device_bypass:
                round_changed |= _bypass_cross_device_pair(
                    graph, kernel, successor, placement)
                continue
            collective = _create_collective(kernel, successor)
            if collective is None:
                collector_shard = next(iter(kernel.inputs.values())).shape
                distributor_shard = next(iter(successor.outputs.values())).shape
                if collector_shard != distributor_shard:
                    continue
                _bypass_pair(graph, kernel, successor)
            else:
                _replace_pair(
                    graph, kernel, successor, collective, placement)
            round_changed = True

        did_change |= round_changed
        if not round_changed:
            break

    return did_change


def _bypass_cross_device_pair(
    graph: ComputeGraph,
    collector: Kernel,
    distributor: Kernel,
    placement: Placement,
) -> bool:
    """Apply the cross-device-set whitelist or reject the pair."""
    # Sequential one-dimensional splits intentionally retain hierarchical
    # collectors/distributors. Each level has its own world and cost.
    if (
        isinstance(collector, Gather)
        and isinstance(distributor, Gather)
    ) or (
        isinstance(collector, Scatter)
        and isinstance(distributor, Scatter)
    ):
        return False
    elif (
        isinstance(collector, Gather)
        and isinstance(distributor, Scatter)
        and collector.dim == distributor.dim
    ):
        sources = {}
        for edge in graph._in_edges(collector):
            for src_out, collector_in in edge.mapping.items():
                sources[collector_in] = (edge.src, src_out)

        destinations = {}
        for edge in graph._out_edges(distributor):
            for distributor_out, dst_in in edge.mapping.items():
                destinations[distributor_out] = (edge.dst, dst_in)

        collector_ports = list(collector.inputs)
        distributor_ports = list(distributor.outputs)
        for collector_in, distributor_out in zip(
                collector_ports, distributor_ports):
            src_kernel, src_out = sources[collector_in]
            dst_kernel, dst_in = destinations[distributor_out]
            src_tensor = src_kernel.outputs[src_out]
            dst_tensor = dst_kernel.inputs[dst_in]
            src_memory = placement.get_tensor_memory(src_tensor)
            dst_memory = placement.get_tensor_memory(dst_tensor)
            if src_memory is None or dst_memory is None \
               or src_memory is dst_memory:
                graph.add_data_edge(src_kernel, dst_kernel, {src_out: dst_in})
                continue

            move = Move()
            move.inputs = {"src0": src_tensor}
            move.outputs = {
                "dst0": Tensor(src_tensor.dtype, src_tensor.shape),
            }
            graph.add_kernel(move)
            graph.add_data_edge(src_kernel, move, {src_out: "src0"})
            graph.add_data_edge(move, dst_kernel, {"dst0": dst_in})
            placement.set_tensor_memory(move.inputs["src0"], src_memory)
            placement.set_tensor_memory(move.outputs["dst0"], dst_memory)
            stream = 0
            if src_kernel in placement.placed_kernels:
                stream = placement.get_kernel_device(src_kernel).stream
            placement.set_kernel_device(
                move, placement.get_tensor_device(src_tensor), stream=stream)

        graph.remove_kernel(collector)
        graph.remove_kernel(distributor)
        return True
    else:
        raise ValueError(
            "Unsupported cross-device-set communication pair: "
            f"{type(collector).__name__} -> "
            f"{type(distributor).__name__}")


def _replace_pair(
    graph: ComputeGraph, collector: Kernel, distributor: Kernel,
    collective: Kernel, placement: Placement,
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
    for old_tensor, new_tensor in zip(
        list(collector.inputs.values()) + list(distributor.outputs.values()),
        list(collective.inputs.values()) + list(collective.outputs.values()),
    ):
        memory = placement.get_tensor_memory(old_tensor)
        if memory is not None:
            placement.set_tensor_memory(new_tensor, memory)
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
        in_edges = graph._in_edges(kernel)
        out_edges = graph._out_edges(kernel)
        if len(in_edges) != 1 or len(out_edges) != 1:
            continue
        neighbors = (in_edges[0].src, out_edges[0].dst)
        graph.remove_identity(kernel)
        did_change = True
        for neighbor in neighbors:
            if isinstance(neighbor, CommKernel) \
                    and neighbor not in pending \
                    and graph._dag.has_node(neighbor):
                queue.append(neighbor)
                pending.add(neighbor)

    return did_change
