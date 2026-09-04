# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""rooflang.language.hardware — hardware component nodes.

Compute/Memory nodes live here. The hardware graph (HardwareGraph) and
its edges (FabricEdge) live in language/graph.py alongside ComputeGraph.
"""

from rooflang.language.hardware.component import (
    HardwareComponent,
    Compute,
    Memory,
)
