"""H200 cluster and aggregated-node presets."""

from collections import Counter
from typing import List

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


class H200Cluster(HardwareGraph):
    """H200 cluster.

    Per-node topology:
      - 8 NVIDIA H200 SXM GPUs, each with 141 GB HBM3e.
      - 1 logical NVSwitch connecting all 8 GPUs.
      - 1 HGX PCIe switch connecting GPUs, NICs, and SSDs.
      - 2 Intel Xeon 6767P CPUs connected via QPI, each with 1.5 TB DDR5.
      - 1 CPU PCIe switch connecting both CPUs to the HGX PCIe switch.
      - 8 Mellanox ConnectX-7 NICs (400 Gb/s InfiniBand each).
      - 8 SSDs (30.72 TB NVMe each) attached to the HGX PCIe switch.
    Inter-node: all NICs connect to a shared IB switch.
    """

    def __init__(self, n_nodes: int = 1):
        super().__init__()

        ib_switch = Compute(name="ib-switch", kind="switch")
        self.add_node(ib_switch)

        for node in range(n_nodes):
            p = f"n{node}-"

            gpus = [Compute(name=f"{p}nvidia-h200-sxm-{i}", tflops={
                "fp4": 1979.0, "fp8": 1979.0,
                "bf16": 989.5, "fp16": 989.5, "fp32": 494.5,
            }, kind="gpu") for i in range(8)]

            nvswitch = Compute(name=f"{p}nvswitch", kind="switch")
            hgx_pcie_switch = Compute(
                name=f"{p}hgx-pcie-switch", kind="switch")
            cpu_pcie_switch = Compute(
                name=f"{p}cpu-pcie-switch", kind="switch")

            cpus = [Compute(name=f"{p}intel-xeon-6767p-{i}", tflops={
                "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
            }, kind="cpu") for i in range(2)]

            nics = [Compute(name=f"{p}mellanox-cx7-{i}", kind="nic")
                    for i in range(8)]

            hbms = [Memory(name=f"{p}hbm3e-{i}", capacity_gb=141.0,
                           kind="hbm")
                    for i in range(8)]
            drams = [Memory(name=f"{p}ddr5-{i}", capacity_gb=1536.0,
                            kind="dram")
                     for i in range(2)]
            ssds = [Memory(name=f"{p}ssd-{i}", capacity_gb=30720.0,
                           kind="ssd")
                    for i in range(8)]

            for comp in (gpus + [nvswitch, hgx_pcie_switch,
                                 cpu_pcie_switch] + cpus + nics):
                self.add_node(comp)
            for mem in (hbms + drams + ssds):
                self.add_node(mem)

            for i in range(8):
                self.add_edge(FabricEdge(
                    name="hbm", src=gpus[i], dst=hbms[i],
                    src_to_dst_bandwidth_gbs=4800.0,
                    dst_to_src_bandwidth_gbs=4800.0,
                    is_full_duplex=False, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="nvlink", src=gpus[i], dst=nvswitch,
                    src_to_dst_bandwidth_gbs=450.0,
                    dst_to_src_bandwidth_gbs=450.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=gpus[i], dst=hgx_pcie_switch,
                    src_to_dst_bandwidth_gbs=64.0,
                    dst_to_src_bandwidth_gbs=64.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=nics[i], dst=hgx_pcie_switch,
                    src_to_dst_bandwidth_gbs=64.0,
                    dst_to_src_bandwidth_gbs=64.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="infiniband", src=nics[i], dst=ib_switch,
                    src_to_dst_bandwidth_gbs=50.0,
                    dst_to_src_bandwidth_gbs=50.0,
                    is_full_duplex=True, alpha_us=1.0,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=ssds[i], dst=hgx_pcie_switch,
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
                self.add_edge(FabricEdge(
                    name="pcie", src=cpus[s], dst=cpu_pcie_switch,
                    src_to_dst_bandwidth_gbs=256.0,
                    dst_to_src_bandwidth_gbs=256.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))

            self.add_edge(FabricEdge(
                name="qpi", src=cpus[0], dst=cpus[1],
                src_to_dst_bandwidth_gbs=192.0,
                dst_to_src_bandwidth_gbs=192.0,
                is_full_duplex=True, alpha_us=0.05,
            ))
            self.add_edge(FabricEdge(
                name="pcie", src=cpu_pcie_switch, dst=hgx_pcie_switch,
                src_to_dst_bandwidth_gbs=512.0,
                dst_to_src_bandwidth_gbs=512.0,
                is_full_duplex=True, alpha_us=0.5,
            ))

    def find_aggregate_bandwidth(self, devices: List[Compute]) -> float:
        """H200 aggregate BW: NVLink intra-node, multi-rail IB inter-node."""
        if len(devices) < 2:
            return float("inf")
        nodes = Counter(d.name.split("-")[0] for d in devices)
        if len(nodes) == 1:
            return 450.0
        return min(nodes.values()) * 50.0


class H200SuperChip(HardwareGraph):
    """One H200 node represented as aggregated components.

    GPU/HBM, CPU/DRAM, NIC, and SSD components each aggregate the matching
    components from one eight-GPU H200 node. There is no NVSwitch because
    the eight GPUs are represented by one compute component.
    """

    def __init__(self):
        super().__init__()

        gpu = Compute(name="n0-nvidia-h200-sxm-0", tflops={
            "fp4": 15832.0, "fp8": 15832.0,
            "bf16": 7916.0, "fp16": 7916.0, "fp32": 3956.0,
        }, kind="gpu")
        hgx_pcie_switch = Compute(
            name="n0-hgx-pcie-switch", kind="switch")
        cpu_pcie_switch = Compute(
            name="n0-cpu-pcie-switch", kind="switch")
        cpu = Compute(name="n0-intel-xeon-6767p-0", tflops={
            "bf16": 510.58, "fp16": 510.58, "int8": 1022.36,
        }, kind="cpu")
        nic = Compute(name="n0-mellanox-cx7-0", kind="nic")
        ib_switch = Compute(name="ib-switch", kind="switch")

        hbm = Memory(
            name="n0-hbm3e-0", capacity_gb=1128.0, kind="hbm")
        dram = Memory(
            name="n0-ddr5-0", capacity_gb=3072.0, kind="dram")
        ssd = Memory(name="n0-ssd", capacity_gb=245760.0, kind="ssd")

        for comp in [gpu, hgx_pcie_switch, cpu_pcie_switch,
                     cpu, nic, ib_switch]:
            self.add_node(comp)
        for mem in [hbm, dram, ssd]:
            self.add_node(mem)

        self.add_edge(FabricEdge(
            name="hbm", src=gpu, dst=hbm,
            src_to_dst_bandwidth_gbs=38400.0,
            dst_to_src_bandwidth_gbs=38400.0,
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
            src_to_dst_bandwidth_gbs=400.0,
            dst_to_src_bandwidth_gbs=400.0,
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
