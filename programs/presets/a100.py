# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""A100 cluster and aggregated-node presets."""

from collections import Counter
from typing import List

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


def _validate_nic_gbps(nic_gbps: int) -> None:
    if nic_gbps <= 0:
        raise ValueError(f"nic_gbps must be positive; got {nic_gbps}")


class A100Cluster(HardwareGraph):
    """A100 cluster.

    Per-node topology:
      - 8 NVIDIA A100 SXM 80GB GPUs, each with 80 GB HBM2e.
      - 1 logical NVSwitch connecting all 8 GPUs (NVLink 3).
      - 8 HGX PCIe switches, each connecting one GPU, NIC, and SSD.
      - 2 Intel Xeon 6767P CPUs connected via QPI, each with 1.5 TB DDR5.
      - Each CPU connects directly to four HGX PCIe switches.
      - 8 Mellanox ConnectX-6 NICs (``nic_gbps`` Gb/s InfiniBand each,
        default 200 Gb/s).
      - 8 SSDs (256 TB NVMe each), one per HGX PCIe switch.
    Inter-node: all NICs connect to a shared IB switch.

    The host platform (CPU, DRAM, SSD, PCIe) matches the H200 preset so that
    accelerator comparisons are not confounded by host differences.  A100 has
    no native FP4/FP8 tensor cores; those peaks are modeled at the BF16 rate.
    """

    def __init__(self, n_nodes: int = 1, nic_gbps: int = 200):
        _validate_nic_gbps(nic_gbps)
        super().__init__()
        self.nic_gbps = nic_gbps
        nic_gbs = nic_gbps / 8.0

        ib_switch = Compute(name="ib-switch", kind="switch")
        self.add_node(ib_switch)

        for node in range(n_nodes):
            p = f"n{node}-"

            gpus = [Compute(name=f"{p}nvidia-a100-sxm-80gb-{i}", tflops={
                "fp4": 312.0, "fp8": 312.0,
                "bf16": 312.0, "fp16": 312.0, "fp32": 19.5,
            }, kind="gpu") for i in range(8)]

            nvswitch = Compute(name=f"{p}nvswitch", kind="switch")
            hgx_pcie_switches = [Compute(
                name=f"{p}hgx-pcie-switch-{i}", kind="switch")
                for i in range(8)]

            cpus = [Compute(name=f"{p}intel-xeon-6767p-{i}", tflops={
                "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
            }, kind="cpu") for i in range(2)]

            nics = [Compute(name=f"{p}mellanox-cx6-{i}", kind="nic")
                    for i in range(8)]

            hbms = [Memory(name=f"{p}hbm2e-{i}", capacity_gb=80.0,
                           kind="hbm") for i in range(8)]
            drams = [Memory(name=f"{p}ddr5-{i}", capacity_gb=1536.0,
                            kind="dram") for i in range(2)]
            ssds = [Memory(name=f"{p}ssd-{i}", capacity_gb=256000.0,
                           kind="ssd") for i in range(8)]

            for comp in (gpus + [nvswitch] + hgx_pcie_switches
                         + cpus + nics):
                self.add_node(comp)
            for mem in (hbms + drams + ssds):
                self.add_node(mem)

            for i in range(8):
                self.add_edge(FabricEdge(
                    name="hbm", src=gpus[i], dst=hbms[i],
                    src_to_dst_bandwidth_gbs=2039.0,
                    dst_to_src_bandwidth_gbs=2039.0,
                    is_full_duplex=False, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="nvlink", src=gpus[i], dst=nvswitch,
                    src_to_dst_bandwidth_gbs=300.0,
                    dst_to_src_bandwidth_gbs=300.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=gpus[i], dst=hgx_pcie_switches[i],
                    src_to_dst_bandwidth_gbs=64.0,
                    dst_to_src_bandwidth_gbs=64.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=nics[i], dst=hgx_pcie_switches[i],
                    src_to_dst_bandwidth_gbs=64.0,
                    dst_to_src_bandwidth_gbs=64.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="infiniband", src=nics[i], dst=ib_switch,
                    src_to_dst_bandwidth_gbs=nic_gbs,
                    dst_to_src_bandwidth_gbs=nic_gbs,
                    is_full_duplex=True, alpha_us=1.0,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=ssds[i], dst=hgx_pcie_switches[i],
                    src_to_dst_bandwidth_gbs=14.0,
                    dst_to_src_bandwidth_gbs=7.0,
                    is_full_duplex=True, alpha_us=50.0,
                ))

            for s in range(2):
                self.add_edge(FabricEdge(
                    name="dram", src=cpus[s], dst=drams[s],
                    src_to_dst_bandwidth_gbs=332.8,
                    dst_to_src_bandwidth_gbs=332.8,
                    is_full_duplex=False, alpha_us=0.1,
                ))
                for i in range(s * 4, (s + 1) * 4):
                    self.add_edge(FabricEdge(
                        name="pcie", src=cpus[s],
                        dst=hgx_pcie_switches[i],
                        src_to_dst_bandwidth_gbs=64.0,
                        dst_to_src_bandwidth_gbs=64.0,
                        is_full_duplex=True, alpha_us=0.5,
                    ))

            self.add_edge(FabricEdge(
                name="qpi", src=cpus[0], dst=cpus[1],
                src_to_dst_bandwidth_gbs=192.0,
                dst_to_src_bandwidth_gbs=192.0,
                is_full_duplex=True, alpha_us=0.05,
            ))

    def find_aggregate_bandwidth(self, devices: List[Compute]) -> float:
        """A100 aggregate BW: NVLink intra-node, multi-rail IB inter-node."""
        if len(devices) < 2:
            return float("inf")
        nodes = Counter(d.name.split("-")[0] for d in devices)
        if len(nodes) == 1:
            return 300.0
        return min(nodes.values()) * self.nic_gbps / 8.0


class A100SuperChip(HardwareGraph):
    """One A100 node represented as aggregated components.

    GPU/HBM, CPU/DRAM, NIC, and SSD components each aggregate the matching
    components from one eight-GPU A100 node. There is no NVSwitch because
    the eight GPUs are represented by one compute component.  ``nic_gbps``
    is the per-NIC speed aggregated by the single NIC component (default
    200 Gb/s).
    """

    def __init__(self, nic_gbps: int = 200):
        _validate_nic_gbps(nic_gbps)
        super().__init__()
        self.nic_gbps = nic_gbps

        gpu = Compute(name="n0-nvidia-a100-sxm-80gb-0", tflops={
            "fp4": 2496.0, "fp8": 2496.0,
            "bf16": 2496.0, "fp16": 2496.0, "fp32": 156.0,
        }, kind="gpu")
        hgx_pcie_switch = Compute(
            name="n0-hgx-pcie-switch", kind="switch")
        cpu_pcie_switch = Compute(
            name="n0-cpu-pcie-switch", kind="switch")
        cpu = Compute(name="n0-intel-xeon-6767p-0", tflops={
            "bf16": 510.58, "fp16": 510.58, "int8": 1022.36,
        }, kind="cpu")
        nic = Compute(name="n0-mellanox-cx6-0", kind="nic")
        ib_switch = Compute(name="ib-switch", kind="switch")

        hbm = Memory(
            name="n0-hbm2e-0", capacity_gb=640.0, kind="hbm")
        dram = Memory(
            name="n0-ddr5-0", capacity_gb=3072.0, kind="dram")
        ssd = Memory(name="n0-ssd", capacity_gb=2048000.0, kind="ssd")

        for comp in [gpu, hgx_pcie_switch, cpu_pcie_switch,
                     cpu, nic, ib_switch]:
            self.add_node(comp)
        for mem in [hbm, dram, ssd]:
            self.add_node(mem)

        self.add_edge(FabricEdge(
            name="hbm", src=gpu, dst=hbm,
            src_to_dst_bandwidth_gbs=16312.0,
            dst_to_src_bandwidth_gbs=16312.0,
            is_full_duplex=False, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=gpu, dst=hgx_pcie_switch,
            src_to_dst_bandwidth_gbs=512.0,
            dst_to_src_bandwidth_gbs=512.0,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=nic, dst=hgx_pcie_switch,
            src_to_dst_bandwidth_gbs=512.0,
            dst_to_src_bandwidth_gbs=512.0,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="infiniband", src=nic, dst=ib_switch,
            src_to_dst_bandwidth_gbs=nic_gbps,
            dst_to_src_bandwidth_gbs=nic_gbps,
            is_full_duplex=True, alpha_us=1.0,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=cpu_pcie_switch, dst=hgx_pcie_switch,
            src_to_dst_bandwidth_gbs=512.0,
            dst_to_src_bandwidth_gbs=512.0,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=cpu, dst=cpu_pcie_switch,
            src_to_dst_bandwidth_gbs=512.0,
            dst_to_src_bandwidth_gbs=512.0,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="dram", src=cpu, dst=dram,
            src_to_dst_bandwidth_gbs=665.6,
            dst_to_src_bandwidth_gbs=665.6,
            is_full_duplex=False, alpha_us=0.1,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=ssd, dst=hgx_pcie_switch,
            src_to_dst_bandwidth_gbs=112.0,
            dst_to_src_bandwidth_gbs=56.0,
            is_full_duplex=True, alpha_us=50.0,
        ))
