"""rooflang.language.hardware — hardware component graph.

Hardware is a graph: Compute/Memory nodes connected by Fabric edges.
Cluster holds the full topology.
"""

from rooflang.language.hardware.component import (
    HardwareComponent,
    Compute,
    Memory,
    Fabric,
    Cluster,
)
