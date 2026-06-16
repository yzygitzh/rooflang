"""Hardware component graph: Compute, Memory, Fabric, and Cluster.

Hardware is modeled as a graph:
  - Nodes: Compute (GPU, CPU, NIC, switch) and Memory (HBM, DRAM, SSD).
  - Edges: Fabric (NVLink, PCIe, IB, etc.) with directional bandwidth.
  - Cluster: a complete hardware topology (nodes + edges).
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HardwareComponent:
    """Base class for all hardware graph nodes."""
    name: str


@dataclass
class Compute(HardwareComponent):
    """A compute node: GPU, CPU, NIC, switch, etc."""
    tflops: Dict[str, float] = field(default_factory=dict)


@dataclass
class Memory(HardwareComponent):
    """A memory node: HBM, DRAM, SSD, etc."""
    capacity_gb: float = 0.0


@dataclass
class Fabric:
    """A directed edge connecting two hardware components.

    Bandwidth model:
      - is_full_duplex=True: both directions can sustain peak simultaneously.
        time = alpha + max(src_bytes/src_bw, dst_bytes/dst_bw)
      - is_full_duplex=False: shared medium, directions compete.
        Constraint: reached_src/peak_src + reached_dst/peak_dst = 1.0
        time = alpha + src_bytes/src_bw + dst_bytes/dst_bw
    """
    name: str
    src: HardwareComponent
    dst: HardwareComponent
    src_to_dst_bandwidth_gbs: float
    dst_to_src_bandwidth_gbs: float
    is_full_duplex: bool
    alpha_us: float = 0.0

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


@dataclass
class Cluster:
    """A complete hardware topology: compute/memory nodes + fabric edges."""
    computes: List[Compute] = field(default_factory=list)
    memories: List[Memory] = field(default_factory=list)
    fabrics: List[Fabric] = field(default_factory=list)
