"""rooflang.language.hardware — hardware component graph.

Hardware is a graph: Compute/Memory nodes connected by Fabric edges.
Cluster holds the full topology. spec.py is retained for backward
compatibility with render.py until the runtime is migrated.
"""

from rooflang.language.hardware.component import (
    HardwareComponent,
    Compute,
    Memory,
    Fabric,
    Cluster,
)

from rooflang.language.hardware.spec import (
    HardwareSpec,
    LinkSpec,
    InterNodeSpec,
    HW_B300,
    hardware_spec,
    collective_bytes,
    collective_time,
)
