"""ComputeGraph — DAG of kernels with typed edges.

Two edge types (distinguished by whether mapping is empty):
  - Data edge (mapping non-empty): kernel B consumes kernel A's output.
    Carries a mapping {src_output_name: dst_input_name} with shape+dtype
    validation.
  - Control edge (mapping empty): kernel B must execute after kernel A
    (ordering only, no data flow).
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet, List, NamedTuple, Set

import networkx as nx

from rooflang.language.kernels.comm import Broadcast
from rooflang.language.kernels.kernel import Kernel


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
        for k_in, k_out in zip(k_ins, k_outs):
            src_output = next(k for k, v in src_edge.mapping.items() if v == k_in)
            dst_input = dst_edge.mapping[k_out]
            reconnect[src_output] = dst_input
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
        """Split a kernel into n copies with communication kernels.

        split_class: callable that takes (kernel, n) and returns
        (prev_comm, kernels, next_comm) where:
          - prev_comm: Kernel — communication before the split copies
            (e.g., Scatter/Broadcast). Its outputs are ordered by copy:
            first len(copy.inputs) entries feed copy_0, next feed copy_1, etc.
          - kernels: List[Kernel] of length n — the split copies.
          - next_comm: Kernel — communication after the split copies
            (e.g., Gather/Reduce). Its inputs are ordered by copy:
            first len(copy.outputs) entries receive from copy_0, etc.

        Returns (prev_comm, kernels, next_comm) — all already in the graph.
        """
        self._check_in_graph(kernel)
        prev_comm, copies, next_comm = split_class(kernel, n)
        if len(copies) != n:
            raise ValueError(
                f"split_class returned {len(copies)} kernels, expected {n}")
        assert prev_comm is not None, "split_class must return prev_comm"
        assert next_comm is not None, "split_class must return next_comm"

        in_edges = self._in_edges(kernel)
        out_edges = self._out_edges(kernel)

        self.add_kernel(prev_comm)
        for edge in in_edges:
            self.add_data_edge(edge.src, prev_comm, edge.mapping)

        prev_out_keys = list(prev_comm.outputs)
        n_inputs_per_copy = len(copies[0].inputs)
        for i, c in enumerate(copies):
            self.add_kernel(c)
            chunk = prev_out_keys[i * n_inputs_per_copy:(i + 1) * n_inputs_per_copy]
            self.add_data_edge(prev_comm, c, dict(zip(chunk, c.inputs)))

        self.add_kernel(next_comm)
        next_in_keys = list(next_comm.inputs)
        n_outputs_per_copy = len(copies[0].outputs)
        for i, c in enumerate(copies):
            chunk = next_in_keys[i * n_outputs_per_copy:(i + 1) * n_outputs_per_copy]
            self.add_data_edge(c, next_comm, dict(zip(c.outputs, chunk)))

        for edge in out_edges:
            self.add_data_edge(next_comm, edge.dst, edge.mapping)

        self.remove_kernel(kernel)
        return prev_comm, copies, next_comm

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

    # ── Internals ─────────────────────────────────────────────────────

    def _in_edges(self, kernel: Kernel) -> List[DataEdge]:
        self._check_in_graph(kernel)
        edges = []
        for src, _, attr in self._dag.in_edges(kernel, data=True):
            if attr["mapping"]:
                edges.append(DataEdge(src=src, dst=kernel,
                                      mapping=attr["mapping"]))
        return edges

    def _out_edges(self, kernel: Kernel) -> List[DataEdge]:
        self._check_in_graph(kernel)
        edges = []
        for _, dst, attr in self._dag.out_edges(kernel, data=True):
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
