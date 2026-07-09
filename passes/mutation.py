"""Graph Mutation Pass — structural transformations on ComputeGraph."""

from rooflang.language.graph import ComputeGraph
from rooflang.language.kernels.kernel import Kernel


def add_control_dep(graph: ComputeGraph, src: Kernel, dst: Kernel) -> None:
    """Add a control-dependency edge: dst must execute after src."""
    graph.add_control_edge(src, dst)


def remove_control_dep(graph: ComputeGraph, src: Kernel, dst: Kernel) -> None:
    """Remove a control-dependency edge between src and dst."""
    graph.remove_control_edge(src, dst)
