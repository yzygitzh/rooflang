"""ComputeGraph — DAG of kernels with typed edges.

Two edge types (distinguished by whether mapping is empty):
  - Data edge (mapping non-empty): kernel B consumes kernel A's output.
    Carries a mapping {src_output_name: dst_input_name} with shape+dtype
    validation.
  - Control edge (mapping empty): kernel B must execute after kernel A
    (ordering only, no data flow).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

import networkx as nx

from rooflang.language.kernels.comm import Broadcast, CommKernel
from rooflang.language.kernels.kernel import Kernel
from rooflang.language.hardware.component import Compute, HardwareComponent, Memory


class DataEdge(NamedTuple):
    """A data-dependency edge between two kernels."""
    src: Kernel
    dst: Kernel
    mapping: Dict[str, str]  # {src_output_name: dst_input_name}


class ComputeGraph:
    """DAG of Kernel nodes with data-dependency and control-dependency edges.

    Edges are stored as nx edge attributes with a single field `mapping`:
      - mapping non-empty → data edge
      - mapping empty → control edge
    """

    def __init__(self) -> None:
        self._dag: nx.DiGraph = nx.DiGraph()

    # ── Node API ──────────────────────────────────────────────────────

    def add_kernel(self, kernel: Kernel) -> None:
        if self._dag.has_node(kernel):
            raise ValueError(f"Kernel already in graph: {kernel}")
        self._dag.add_node(kernel)

    def remove_kernel(self, kernel: Kernel) -> None:
        if not self._dag.has_node(kernel):
            raise ValueError(f"Kernel not in graph: {kernel}")
        self._dag.remove_node(kernel)

    @property
    def kernels(self) -> FrozenSet[Kernel]:
        return frozenset(self._dag.nodes)

    # ── Data edge API ─────────────────────────────────────────────────

    def add_data_edge(
        self, src: Kernel, dst: Kernel, mapping: Dict[str, str]
    ) -> None:
        """Add a data-dependency edge with output→input mapping.

        mapping: {src.outputs key -> dst.inputs key}.
        Validates shape and dtype match for each pair.
        """
        self._check_in_graph(src)
        self._check_in_graph(dst)
        if not mapping:
            raise ValueError("mapping must be non-empty for a data edge")
        for out_name, in_name in mapping.items():
            if out_name not in src.outputs:
                raise ValueError(
                    f"'{out_name}' not in src.outputs: {list(src.outputs)}")
            if in_name not in dst.inputs:
                raise ValueError(
                    f"'{in_name}' not in dst.inputs: {list(dst.inputs)}")
            src_td = src.outputs[out_name]
            dst_td = dst.inputs[in_name]
            if src_td.shape != dst_td.shape:
                raise ValueError(
                    f"Shape mismatch: src.outputs['{out_name}'].shape="
                    f"{src_td.shape} != dst.inputs['{in_name}'].shape="
                    f"{dst_td.shape}")
            if src_td.dtype != dst_td.dtype:
                raise ValueError(
                    f"Dtype mismatch: src.outputs['{out_name}'].dtype="
                    f"'{src_td.dtype}' != dst.inputs['{in_name}'].dtype="
                    f"'{dst_td.dtype}'")
        if self._dag.has_edge(src, dst):
            existing = self._dag.edges[src, dst]["mapping"]
            if existing:
                existing.update(mapping)
                return
            else:
                self._dag.remove_edge(src, dst)
        self._dag.add_edge(src, dst, mapping=dict(mapping))

    # ── Control edge API ──────────────────────────────────────────────

    def add_control_edge(self, src: Kernel, dst: Kernel) -> None:
        self._check_in_graph(src)
        self._check_in_graph(dst)
        if self._dag.has_edge(src, dst):
            return  # data edge already implies ordering
        self._dag.add_edge(src, dst, mapping={})

    def remove_control_edge(self, src: Kernel, dst: Kernel) -> None:
        if not self._dag.has_edge(src, dst):
            raise ValueError("No edge between src and dst")
        if self._dag.edges[src, dst]["mapping"]:
            raise ValueError("Edge is a data edge, not a control edge")
        self._dag.remove_edge(src, dst)

    # ── Mutation API ──────────────────────────────────────────────────

    def insert_identity(
        self, kernel: Kernel, k1: Kernel, k2: Kernel,
        mapping: Dict[str, str],
    ) -> None:
        """Insert an identity kernel between k1 and k2.

        mapping: {k1_output_name: k2_input_name} — connections to intercept.
        The identity kernel's inputs/outputs are matched by iteration order
        against the mapping entries.
        """
        k_ins = list(kernel.inputs)
        k_outs = list(kernel.outputs)
        if len(mapping) != len(k_ins) or len(mapping) != len(k_outs):
            raise ValueError(
                f"mapping has {len(mapping)} entries but kernel has "
                f"{len(k_ins)} inputs and {len(k_outs)} outputs")
        self.add_kernel(kernel)
        in_map = {}
        out_map = {}
        for (k1_out, k2_in), k_in, k_out in zip(mapping.items(), k_ins, k_outs):
            in_map[k1_out] = k_in
            out_map[k_out] = k2_in
        self.add_data_edge(k1, kernel, in_map)
        self.add_data_edge(kernel, k2, out_map)
        for k1_out in mapping:
            self._remove_data_mapping(k1, k2, k1_out)

    def remove_identity(self, kernel: Kernel) -> None:
        """Remove an identity kernel, reconnecting predecessor to successor."""
        in_edges = self._in_edges(kernel)
        out_edges = self._out_edges(kernel)
        if len(in_edges) != 1 or len(out_edges) != 1:
            raise ValueError(
                "Identity kernel must have exactly one in and one out data edge")
        src_edge = in_edges[0]
        dst_edge = out_edges[0]
        reconnect = {}
        k_ins = list(kernel.inputs)
        k_outs = list(kernel.outputs)
        src_output_by_input = {}
        for output_name, input_name in src_edge.mapping.items():
            src_output_by_input.setdefault(input_name, output_name)
        for k_in, k_out in zip(k_ins, k_outs):
            reconnect[src_output_by_input[k_in]] = dst_edge.mapping[k_out]
        self.add_data_edge(src_edge.src, dst_edge.dst, reconnect)
        self.remove_kernel(kernel)

    def fuse_kernels(
        self,
        fuse_class: Callable[[List[Kernel]], Kernel],
        kernel_list: List[Kernel],
    ) -> Kernel:
        """Fuse a subgraph of kernels into one, eliminating intermediate HBM traffic.

        fuse_class: callable that takes the list of kernels and returns a new
        Kernel instance representing the fused operation.
        kernel_list: kernels forming a subgraph (not necessarily connected —
        can be a forest). External inputs (edges from outside the subgraph
        into roots) are rewired to the fused kernel. External outputs (edges
        from exits to outside the subgraph) are rewired from the fused kernel.

        Returns the fused kernel (already inserted into the graph).
        """
        if len(kernel_list) < 2:
            raise ValueError("Need at least 2 kernels to fuse")
        for k in kernel_list:
            self._check_in_graph(k)
        subgraph = set(kernel_list)
        fused = fuse_class(kernel_list)
        self.add_kernel(fused)
        for k in kernel_list:
            for edge in self._in_edges(k):
                if edge.src not in subgraph:
                    self.add_data_edge(edge.src, fused, edge.mapping)
            for edge in self._out_edges(k):
                if edge.dst not in subgraph:
                    self.add_data_edge(fused, edge.dst, edge.mapping)
        for k in kernel_list:
            self.remove_kernel(k)
        return fused

    def split_kernel(
        self,
        split_class: Callable[[Kernel, int], tuple],
        kernel: Kernel,
        n: int,
    ) -> tuple:
        """Split a kernel into n copies with per-port communication kernels.

        split_class: callable that takes (kernel, n) and returns
        (prev_comms, kernels, next_comms) where:
          - prev_comms: Dict[str, Kernel] — one comm kernel per input port.
            Each has 1 input ("x") and n outputs ("o0".."o{n-1}").
            Keyed by the original kernel's input port name.
          - kernels: List[Kernel] of length n — the split copies.
          - next_comms: Dict[str, Kernel] — one comm kernel per output port.
            Each has n inputs ("i0".."i{n-1}") and 1 output ("y").
            Keyed by the original kernel's output port name. May be empty for
            a terminal sink kernel with no output ports.

        Returns (prev_comms, kernels, next_comms) — all already in the graph.
        """
        self._check_in_graph(kernel)
        prev_comms, copies, next_comms = split_class(kernel, n)
        if len(copies) != n:
            raise ValueError(
                f"split_class returned {len(copies)} kernels, expected {n}")
        if not prev_comms:
            raise ValueError("split_class must return non-empty prev_comms")
        if not next_comms and kernel.outputs:
            raise ValueError(
                "split_class must return next_comms for kernel outputs")

        in_edges = self._in_edges(kernel)
        out_edges = self._out_edges(kernel)

        # Wire predecessors → prev_comms (each in_edge port maps to its comm)
        for port_name, comm in prev_comms.items():
            self.add_kernel(comm)
        for edge in in_edges:
            for src_out, dst_in in edge.mapping.items():
                comm = prev_comms[dst_in]
                self.add_data_edge(edge.src, comm, {src_out: "x"})

        # Wire prev_comms → copies
        for i, c in enumerate(copies):
            self.add_kernel(c)
            for port_name, comm in prev_comms.items():
                self.add_data_edge(comm, c, {f"o{i}": port_name})

        # Wire copies → next_comms
        for port_name, comm in next_comms.items():
            self.add_kernel(comm)
        for i, c in enumerate(copies):
            for port_name, comm in next_comms.items():
                self.add_data_edge(c, comm, {port_name: f"i{i}"})

        # Wire next_comms → successors
        for edge in out_edges:
            for src_out, dst_in in edge.mapping.items():
                comm = next_comms[src_out]
                self.add_data_edge(comm, edge.dst, {"y": dst_in})

        self.remove_kernel(kernel)
        return prev_comms, copies, next_comms

    def dedup(
        self,
        dedup_class: Callable[[List[Kernel]], tuple],
        kernel_list: List[Kernel],
    ) -> tuple:
        """Merge redundant kernels into one, adding a broadcast after.

        Before: preds → OldBroadcast → [K1, ..., Kn] → [S1, ...]
            or: [K1, ..., Kn] (roots) → [S1, ...]
        After:  preds → survivor → PostBroadcast → [S1, ...]

        Preconditions (checked):
        - All kernels have no side effects.
        - All kernels are computationally identical (same type + to_dict).
        - All share the same Broadcast predecessor, or all are roots.

        dedup_class(kernel_list) -> (survivor, post_broadcast) where:
          - survivor: Kernel to keep (may be from kernel_list or new).
          - post_broadcast: Kernel placed after survivor. Its inputs are
            ordered to match survivor.outputs. Its outputs are ordered by
            successor: for each kernel in kernel_list order, for each out-edge
            in order, the corresponding output chunk feeds that successor.

        Returns (survivor, post_broadcast).
        """
        if len(kernel_list) < 2:
            raise ValueError("Need at least 2 kernels to dedup")
        for k in kernel_list:
            self._check_in_graph(k)
            if k.has_side_effect:
                raise ValueError(f"Cannot dedup kernel with side effect: {k}")
        ref_dict = kernel_list[0].to_dict()
        ref_type = type(kernel_list[0])
        for k in kernel_list[1:]:
            if type(k) != ref_type or k.to_dict() != ref_dict:
                raise ValueError(
                    f"Dedup precondition violated: {k} is not computationally "
                    f"identical to {kernel_list[0]}")
        old_broadcast = None
        for k in kernel_list:
            in_edges = self._in_edges(k)
            if not in_edges:
                if old_broadcast is not None:
                    raise ValueError(
                        "Dedup precondition violated: mixed root and "
                        "non-root kernels")
            elif len(in_edges) == 1 and isinstance(in_edges[0].src, Broadcast):
                if old_broadcast is None:
                    old_broadcast = in_edges[0].src
                elif old_broadcast is not in_edges[0].src:
                    raise ValueError(
                        "Dedup precondition violated: kernels have different "
                        "Broadcast predecessors")
            else:
                raise ValueError(
                    f"Dedup precondition violated: {k} must have either no "
                    f"predecessor or a single Broadcast predecessor")

        all_out_edges = []
        for k in kernel_list:
            all_out_edges.extend(self._out_edges(k))

        survivor, post_broadcast = dedup_class(kernel_list)
        if not self._dag.has_node(survivor):
            self.add_kernel(survivor)
        self.add_kernel(post_broadcast)

        self.add_data_edge(
            survivor, post_broadcast,
            dict(zip(survivor.outputs, post_broadcast.inputs)))

        post_out_keys = list(post_broadcast.outputs)
        offset = 0
        for edge in all_out_edges:
            n = len(edge.mapping)
            chunk = post_out_keys[offset:offset + n]
            si_in_keys = list(edge.mapping.values())
            self.add_data_edge(post_broadcast, edge.dst, dict(zip(chunk, si_in_keys)))
            offset += n

        if old_broadcast is not None:
            bcast_in_edges = self._in_edges(old_broadcast)
            for edge in bcast_in_edges:
                self.add_data_edge(edge.src, survivor, edge.mapping)
            self.remove_kernel(old_broadcast)

        for k in kernel_list:
            if k is not survivor:
                self.remove_kernel(k)

        return survivor, post_broadcast

    def dup(
        self,
        dup_class: Callable[[Kernel, int], tuple],
        kernel: Kernel,
    ) -> tuple:
        """Duplicate a kernel, moving broadcast from after to before.

        Before: preds → kernel → OldBroadcast → [S1, ..., Sn]
        After:  preds → PreBroadcast → [copy_1, ..., copy_n] → [S1, ..., Sn]

        Preconditions (checked):
        - kernel has no side effects.
        - kernel's only data successor is a Broadcast.

        dup_class(kernel, n) -> (pre_broadcast, copies) where:
          - pre_broadcast: Kernel placed before copies. Its inputs are ordered
            to match kernel's inputs. Its outputs are ordered by copy (chunked:
            first len(copy.inputs) for copy_0, next for copy_1, etc.).
          - copies: List[Kernel] of length n.

        Each copy_i connects to S_i using ordered matching: zip(copy_i.outputs,
        S_i's input keys from the old Broadcast→S_i edge).

        Returns (pre_broadcast, copies).
        """
        self._check_in_graph(kernel)
        if kernel.has_side_effect:
            raise ValueError(f"Cannot dup kernel with side effect: {kernel}")
        out_edges = self._out_edges(kernel)
        if len(out_edges) != 1 or not isinstance(out_edges[0].dst, Broadcast):
            raise ValueError(
                "Dup precondition violated: kernel's only data successor "
                "must be a Broadcast")
        old_broadcast = out_edges[0].dst
        bcast_out_edges = self._out_edges(old_broadcast)
        n = len(bcast_out_edges)

        pre_broadcast, copies = dup_class(kernel, n)
        if len(copies) != n:
            raise ValueError(
                f"dup_class returned {len(copies)} copies, expected {n}")

        in_edges = self._in_edges(kernel)

        self.add_kernel(pre_broadcast)
        for edge in in_edges:
            self.add_data_edge(edge.src, pre_broadcast, edge.mapping)

        pre_out_keys = list(pre_broadcast.outputs)
        n_inputs_per_copy = len(copies[0].inputs)
        for i, c in enumerate(copies):
            self.add_kernel(c)
            chunk = pre_out_keys[i * n_inputs_per_copy:(i + 1) * n_inputs_per_copy]
            self.add_data_edge(pre_broadcast, c, dict(zip(chunk, c.inputs)))

        for c, edge in zip(copies, bcast_out_edges):
            self.add_data_edge(
                c, edge.dst, dict(zip(c.outputs, edge.mapping.values())))

        self.remove_kernel(old_broadcast)
        self.remove_kernel(kernel)
        return pre_broadcast, copies

    # ── Query API ─────────────────────────────────────────────────────

    def topological_sort(self) -> List[Kernel]:
        return list(nx.topological_sort(self._dag))

    # ── Validation ────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate graph integrity.

        Checks:
        1. DAG is acyclic.
        2. Partition constraint: every output slot is consumed by exactly
           one data edge; every input slot is provided by exactly one data
           edge. Root kernels (no incoming data edges) are exempt from the
           input constraint. Leaf kernels (no outgoing data edges) are
           exempt from the output constraint.
        """
        if not nx.is_directed_acyclic_graph(self._dag):
            raise ValueError("Graph contains a cycle")

        for kernel in self._dag.nodes:
            out_data = self._out_edges(kernel)
            if kernel.outputs and out_data:
                covered: Set[str] = set()
                for edge in out_data:
                    for out_name in edge.mapping:
                        if out_name in covered:
                            raise ValueError(
                                f"Output '{out_name}' of {kernel} is "
                                f"connected to multiple data edges")
                        covered.add(out_name)
                missing = set(kernel.outputs) - covered
                if missing:
                    raise ValueError(
                        f"Outputs {missing} of {kernel} are not connected")

            in_data = self._in_edges(kernel)
            if kernel.inputs and in_data:
                covered_in: Set[str] = set()
                for edge in in_data:
                    for in_name in edge.mapping.values():
                        if in_name in covered_in:
                            raise ValueError(
                                f"Input '{in_name}' of {kernel} is "
                                f"connected to multiple data edges")
                        covered_in.add(in_name)
                missing = set(kernel.inputs) - covered_in
                if missing:
                    raise ValueError(
                        f"Inputs {missing} of {kernel} are not connected")

        for kernel in self._dag.nodes:
            if not kernel._requires_placement:
                continue
            tensor_in = sum(t.size_bytes for t in kernel.inputs.values())
            tensor_read = sum(
                t.size_bytes * kernel.input_read_fraction(name)
                for name, t in kernel.inputs.items()
            )
            tensor_w = sum(t.size_bytes for t in kernel.weights.values())
            tensor_out = sum(t.size_bytes for t in kernel.outputs.values())
            if kernel.inputs and kernel.input_tensor_bytes != tensor_in:
                raise ValueError(
                    f"{type(kernel).__name__}: input_tensor_bytes property "
                    f"({kernel.input_tensor_bytes}) != tensor sum "
                    f"({tensor_in})")
            if kernel.inputs and kernel.input_bytes != tensor_read:
                raise ValueError(
                    f"{type(kernel).__name__}: input_bytes property "
                    f"({kernel.input_bytes}) != fraction-adjusted tensor "
                    f"sum ({tensor_read})")
            if kernel.weights and kernel.weight_bytes != tensor_w:
                raise ValueError(
                    f"{type(kernel).__name__}: weight_bytes property "
                    f"({kernel.weight_bytes}) != tensor sum ({tensor_w})")
            if kernel.outputs and kernel.output_bytes != tensor_out:
                raise ValueError(
                    f"{type(kernel).__name__}: output_bytes property "
                    f"({kernel.output_bytes}) != tensor sum ({tensor_out})")

        for kernel in self._dag.nodes:
            if isinstance(kernel, CommKernel):
                kernel.validate_ports()

    # ── Internals ─────────────────────────────────────────────────────

    def _in_edges(self, kernel: Kernel) -> List[DataEdge]:
        self._check_in_graph(kernel)
        edges = []
        for src, attr in self._dag.pred[kernel].items():
            if attr["mapping"]:
                edges.append(DataEdge(src=src, dst=kernel,
                                      mapping=attr["mapping"]))
        return edges

    def _out_edges(self, kernel: Kernel) -> List[DataEdge]:
        self._check_in_graph(kernel)
        edges = []
        for dst, attr in self._dag.succ[kernel].items():
            if attr["mapping"]:
                edges.append(DataEdge(src=kernel, dst=dst,
                                      mapping=attr["mapping"]))
        return edges

    def _check_in_graph(self, kernel: Kernel) -> None:
        if not self._dag.has_node(kernel):
            raise ValueError(f"Kernel not in graph: {kernel}")

    def _remove_data_mapping(
        self, src: Kernel, dst: Kernel, src_output: str
    ) -> None:
        if not self._dag.has_edge(src, dst):
            raise ValueError("No edge between src and dst")
        mapping = self._dag.edges[src, dst]["mapping"]
        if src_output not in mapping:
            raise ValueError(f"'{src_output}' not in edge mapping")
        del mapping[src_output]
        if not mapping:
            self._dag.remove_edge(src, dst)


# ═══════════════════════════════════════════════════════════════════════
# Hardware Graph
# ═══════════════════════════════════════════════════════════════════════


class FabricEdge:
    """An edge in the hardware graph (NVLink, PCIe, IB, etc.).

    Bandwidth model:
      - is_full_duplex=True: time = alpha + max(fwd_time, rev_time)
      - is_full_duplex=False: time = alpha + fwd_time + rev_time
    """

    def __init__(
        self,
        name: str,
        src: HardwareComponent,
        dst: HardwareComponent,
        src_to_dst_bandwidth_gbs: float,
        dst_to_src_bandwidth_gbs: float,
        is_full_duplex: bool,
        alpha_us: float = 0.0,
    ) -> None:
        self.name = name
        self.src = src
        self.dst = dst
        self.src_to_dst_bandwidth_gbs = src_to_dst_bandwidth_gbs
        self.dst_to_src_bandwidth_gbs = dst_to_src_bandwidth_gbs
        self.is_full_duplex = is_full_duplex
        self.alpha_us = alpha_us

    def transfer_time_us(self, src_to_dst_bytes: float = 0.0,
                         dst_to_src_bytes: float = 0.0) -> float:
        """Estimate transfer time (microseconds) for bidirectional traffic."""
        t_fwd = (src_to_dst_bytes / (self.src_to_dst_bandwidth_gbs * 1e3)
                 if self.src_to_dst_bandwidth_gbs > 0 and src_to_dst_bytes > 0
                 else 0.0)
        t_rev = (dst_to_src_bytes / (self.dst_to_src_bandwidth_gbs * 1e3)
                 if self.dst_to_src_bandwidth_gbs > 0 and dst_to_src_bytes > 0
                 else 0.0)
        if self.is_full_duplex:
            return self.alpha_us + max(t_fwd, t_rev)
        else:
            return self.alpha_us + t_fwd + t_rev


class HardwareGraph:
    """Undirected graph of hardware components connected by fabric edges.

    Nodes: Compute / Memory instances.
    Edges: FabricEdge instances (directional bandwidth stored per-edge).
    Supports path-finding for effective bandwidth between any two nodes.
    """

    def __init__(self) -> None:
        self._graph: nx.Graph = nx.Graph()
        self._names: Set[str] = set()

    def add_node(self, component: HardwareComponent) -> None:
        if component.name in self._names:
            raise ValueError(
                f"Duplicate hardware component name: '{component.name}'")
        self._names.add(component.name)
        self._graph.add_node(component)
        self._clear_lookup_caches()

    def add_edge(self, edge: FabricEdge) -> None:
        if not self._graph.has_node(edge.src):
            raise ValueError(f"Node not in graph: {edge.src.name}")
        if not self._graph.has_node(edge.dst):
            raise ValueError(f"Node not in graph: {edge.dst.name}")
        if self._graph.has_edge(edge.src, edge.dst):
            self._graph.edges[edge.src, edge.dst]["fabrics"].append(edge)
        else:
            self._graph.add_edge(edge.src, edge.dst, fabrics=[edge])
        self._clear_lookup_caches()

    @property
    def nodes(self) -> FrozenSet[HardwareComponent]:
        return frozenset(self._graph.nodes)

    @lru_cache(maxsize=None)
    def _find_route(
        self, src: HardwareComponent, dst: HardwareComponent,
    ) -> Tuple[Tuple[HardwareComponent, ...], Tuple[FabricEdge, ...]]:
        """Select the route with the lowest aggregate routing weight."""
        if not self._graph.has_node(src) or not self._graph.has_node(dst):
            raise ValueError(f"No path between {src.name} and {dst.name}")
        if src is dst:
            return (src,), ()

        try:
            path = nx.shortest_path(
                self._graph, src, dst, weight=self._routing_weight)
        except nx.NetworkXNoPath:
            raise ValueError(f"No path between {src.name} and {dst.name}")
        fabrics = tuple(
            self._best_fabric(a, b) for a, b in zip(path, path[1:]))
        return tuple(path), fabrics

    def _routing_weight(
        self, src: HardwareComponent, dst: HardwareComponent,
        edge_data: Dict,
    ) -> float:
        """Return the routing cost for one directed traversal."""
        bandwidth = max(
            fabric.src_to_dst_bandwidth_gbs
            if fabric.src is src else fabric.dst_to_src_bandwidth_gbs
            for fabric in edge_data["fabrics"]
        )
        return 1.0 / bandwidth

    @lru_cache(maxsize=None)
    def find_fabric(self, src: HardwareComponent, dst: HardwareComponent) -> FabricEdge:
        """Find the effective fabric between src and dst.

        Single-hop: returns the highest-bandwidth direct FabricEdge.
        Multi-hop: returns a synthetic FabricEdge with bottleneck bandwidth
        (min along path) and cumulative alpha (sum along path).
        Raises ValueError if no path exists.
        """
        if not self._graph.has_node(src) or not self._graph.has_node(dst):
            raise ValueError(f"No path between {src.name} and {dst.name}")
        if src is dst:
            raise ValueError(f"src and dst are the same node: {src.name}")
        path, hops = self._find_route(src, dst)

        if len(hops) == 1:
            return hops[0]

        fwd_bws: List[float] = []
        rev_bws: List[float] = []
        total_alpha = 0.0
        all_full_duplex = True
        for i, fab in enumerate(hops):
            a = path[i]
            if fab.src is a:
                fwd_bws.append(fab.src_to_dst_bandwidth_gbs)
                rev_bws.append(fab.dst_to_src_bandwidth_gbs)
            else:
                fwd_bws.append(fab.dst_to_src_bandwidth_gbs)
                rev_bws.append(fab.src_to_dst_bandwidth_gbs)
            total_alpha += fab.alpha_us
            if not fab.is_full_duplex:
                all_full_duplex = False

        return FabricEdge(
            name=f"path({src.name}->{dst.name})",
            src=src, dst=dst,
            src_to_dst_bandwidth_gbs=min(fwd_bws),
            dst_to_src_bandwidth_gbs=min(rev_bws),
            is_full_duplex=all_full_duplex,
            alpha_us=total_alpha,
        )

    def find_fabric_path(
        self, src: HardwareComponent, dst: HardwareComponent,
    ) -> List[FabricEdge]:
        """Return the list of actual FabricEdge objects on the shortest path."""
        return list(self._find_fabric_path(src, dst))

    @lru_cache(maxsize=None)
    def _find_fabric_path(
        self, src: HardwareComponent, dst: HardwareComponent,
    ) -> Tuple[FabricEdge, ...]:
        """Cache the immutable fabric path for one ordered endpoint pair."""
        if src is dst:
            return ()
        _, hops = self._find_route(src, dst)
        return hops

    def find_fabric_path_directed(
        self, src: HardwareComponent, dst: HardwareComponent,
    ) -> List[Tuple[FabricEdge, str]]:
        """Return directed path: list of (FabricEdge, direction).

        direction is 'fwd' if data flows src→dst on that edge,
        'rev' if data flows dst→src.
        """
        return list(self._find_fabric_path_directed(src, dst))

    @lru_cache(maxsize=None)
    def _find_fabric_path_directed(
        self, src: HardwareComponent, dst: HardwareComponent,
    ) -> Tuple[Tuple[FabricEdge, str], ...]:
        """Cache an immutable directed path for one ordered endpoint pair."""
        if src is dst:
            return ()
        path, hops = self._find_route(src, dst)
        result: List[Tuple[FabricEdge, str]] = []
        for i, edge in enumerate(hops):
            direction = 'fwd' if edge.src is path[i] else 'rev'
            result.append((edge, direction))
        return tuple(result)

    def _clear_lookup_caches(self) -> None:
        """Invalidate topology-dependent lookups after graph mutation."""
        self._find_route.cache_clear()
        self.find_fabric.cache_clear()
        self._find_fabric_path.cache_clear()
        self._find_fabric_path_directed.cache_clear()
        self.find_local_memory.cache_clear()
        self.find_local_device.cache_clear()

    @lru_cache(maxsize=None)
    def find_local_memory(self, device: Compute) -> Memory:
        """Find the Memory node connected to device with highest bandwidth."""
        best_mem: Optional[Memory] = None
        best_bw = 0.0
        for neighbor in self._graph.neighbors(device):
            if not isinstance(neighbor, Memory):
                continue
            for fab in self._graph.edges[device, neighbor]["fabrics"]:
                bw = (fab.src_to_dst_bandwidth_gbs if fab.src is device
                      else fab.dst_to_src_bandwidth_gbs)
                if bw > best_bw:
                    best_bw = bw
                    best_mem = neighbor
        if best_mem is None:
            raise ValueError(f"No memory attached to device: {device.name}")
        return best_mem

    @lru_cache(maxsize=None)
    def find_local_device(self, memory: Memory) -> Compute:
        """Find the nearest execution endpoint that owns ``memory``.

        Switches and NICs are fabric transit components, not devices on which
        tensor kernels execute.  A memory may nevertheless be attached to a
        switch directly (for example, an NVMe SSD below an HGX PCIe switch),
        so walk outward through transit components until the nearest GPU/CPU
        endpoint is reached.  Among equally near endpoints, prefer the path
        with the highest device-to-memory bandwidth; an equally connected GPU
        wins the final tie because per-GPU SSDs share a PCIe switch with both
        their GPU and a socket CPU.
        """
        transit_kinds = {"switch", "nic"}
        visited = {memory}
        frontier = [memory]

        while frontier:
            endpoints = []
            next_frontier = []
            for component in frontier:
                for neighbor in self._graph.neighbors(component):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if not isinstance(neighbor, Compute):
                        continue
                    if neighbor.kind in transit_kinds:
                        next_frontier.append(neighbor)
                    else:
                        endpoints.append(neighbor)

            if endpoints:
                def endpoint_score(device):
                    fabric = self.find_fabric(device, memory)
                    bandwidth = (
                        fabric.src_to_dst_bandwidth_gbs
                        if fabric.src is device
                        else fabric.dst_to_src_bandwidth_gbs
                    )
                    return bandwidth, device.kind == "gpu"

                return max(endpoints, key=endpoint_score)
            frontier = next_frontier

        raise ValueError(
            f"No device attached to memory: {memory.name}")

    def find_aggregate_bandwidth(self, devices: List[Compute]) -> float:
        """Aggregate bandwidth for ring/tree collectives among devices.

        Default: returns min pair BW (conservative, no multi-rail benefit).
        Override in subclass for topologies with multiple parallel inter-node
        links (e.g. return sum of per-NIC BWs for multi-rail IB).
        """
        if len(devices) < 2:
            return float("inf")
        min_bw = float("inf")
        for i, d1 in enumerate(devices):
            for d2 in devices[i + 1:]:
                fab = self.find_fabric(d1, d2)
                bw = min(fab.src_to_dst_bandwidth_gbs, fab.dst_to_src_bandwidth_gbs)
                if bw < min_bw:
                    min_bw = bw
        return min_bw

    def find_aggregate_latency(self, devices: List[Compute]) -> float:
        """Max path latency (diameter) among all pairs in the device group."""
        if len(devices) < 2:
            return 0.0
        max_alpha = 0.0
        for i, d1 in enumerate(devices):
            for d2 in devices[i + 1:]:
                fab = self.find_fabric(d1, d2)
                if fab.alpha_us > max_alpha:
                    max_alpha = fab.alpha_us
        return max_alpha

    def _best_fabric(self, a: HardwareComponent, b: HardwareComponent) -> FabricEdge:
        """Pick the highest-bandwidth fabric between two adjacent nodes."""
        fabs = self._graph.edges[a, b]["fabrics"]
        best = fabs[0]
        best_bw = (best.src_to_dst_bandwidth_gbs if best.src is a
                   else best.dst_to_src_bandwidth_gbs)
        for fab in fabs[1:]:
            bw = (fab.src_to_dst_bandwidth_gbs if fab.src is a
                  else fab.dst_to_src_bandwidth_gbs)
            if bw > best_bw:
                best_bw = bw
                best = fab
        return best
