"""ComputeGraph — DAG of kernels with typed edges.

Two edge types (distinguished by whether mapping is empty):
  - Data edge (mapping non-empty): kernel B consumes kernel A's output.
    Carries a mapping {src_output_name: dst_input_name} with shape+dtype
    validation.
  - Control edge (mapping empty): kernel B must execute after kernel A
    (ordering only, no data flow).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, NamedTuple, Set

import networkx as nx

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
