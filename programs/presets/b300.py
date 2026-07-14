"""B300 Cluster A preset."""

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


class B300ClusterA(HardwareGraph):
    """B300 Cluster A.

    Per-node topology:
      - 8 NVIDIA B300 SXM GPUs, each with 288 GB HBM3e.
      - 1 NVSwitch connecting all 8 GPUs (NVLink full duplex).
      - 1 HGX PCIe switch connecting GPUs and NICs.
      - 2 Intel Xeon 6767P CPUs connected via QPI, each with 1.5 TB DDR5.
      - 1 CPU PCIe switch connecting both CPUs to HGX PCIe switch.
      - 8 Mellanox CX8 NICs (800G NDR IB).
      - 1 SSD (3.84 TB NVMe) under CPU-0.
    Inter-node: all NICs connect to a shared IB switch.
    """

    def __init__(self, n_nodes: int = 1):
        super().__init__()

        ib_switch = Compute(name="ib-switch")
        self.add_node(ib_switch)

        for node in range(n_nodes):
            p = f"n{node}-"

            gpus = [Compute(name=f"{p}nvidia-b300-sxm-{i}", tflops={
                "fp4": 13500.0, "fp8": 4500.0,
                "bf16": 2250.0, "fp16": 2250.0, "fp32": 1125.0,
            }) for i in range(8)]

            nvswitch = Compute(name=f"{p}nvswitch")
            hgx_pcie_switch = Compute(name=f"{p}hgx-pcie-switch")
            cpu_pcie_switch = Compute(name=f"{p}cpu-pcie-switch")

            cpus = [Compute(name=f"{p}intel-xeon-6767p-{i}", tflops={
                "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
            }) for i in range(2)]

            nics = [Compute(name=f"{p}mellanox-cx8-{i}") for i in range(8)]

            hbms = [Memory(name=f"{p}hbm3e-{i}", capacity_gb=288.0) for i in range(8)]
            drams = [Memory(name=f"{p}ddr5-{i}", capacity_gb=1536.0) for i in range(2)]
            ssd = Memory(name=f"{p}ssd", capacity_gb=3840.0)

            nvme_pcie_switch = Compute(name=f"{p}nvme-pcie-switch")

            for comp in (gpus + [nvswitch, hgx_pcie_switch, cpu_pcie_switch,
                                 nvme_pcie_switch] + cpus + nics):
                self.add_node(comp)
            for mem in (hbms + drams + [ssd]):
                self.add_node(mem)

            for i in range(8):
                self.add_edge(FabricEdge(
                    name="hbm", src=gpus[i], dst=hbms[i],
                    src_to_dst_bandwidth_gbs=7750.0,
                    dst_to_src_bandwidth_gbs=7750.0,
                    is_full_duplex=False, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="nvlink", src=gpus[i], dst=nvswitch,
                    src_to_dst_bandwidth_gbs=900.0,
                    dst_to_src_bandwidth_gbs=900.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=gpus[i], dst=nics[i],
                    src_to_dst_bandwidth_gbs=128.0,
                    dst_to_src_bandwidth_gbs=128.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=gpus[i], dst=hgx_pcie_switch,
                    src_to_dst_bandwidth_gbs=128.0,
                    dst_to_src_bandwidth_gbs=128.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=nics[i], dst=hgx_pcie_switch,
                    src_to_dst_bandwidth_gbs=128.0,
                    dst_to_src_bandwidth_gbs=128.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="infiniband", src=nics[i], dst=ib_switch,
                    src_to_dst_bandwidth_gbs=100.0,
                    dst_to_src_bandwidth_gbs=100.0,
                    is_full_duplex=True, alpha_us=1.0,
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
            self.add_edge(FabricEdge(
                name="pcie", src=cpus[0], dst=nvme_pcie_switch,
                src_to_dst_bandwidth_gbs=64.0,
                dst_to_src_bandwidth_gbs=64.0,
                is_full_duplex=True, alpha_us=0.5,
            ))
            self.add_edge(FabricEdge(
                name="pcie", src=ssd, dst=nvme_pcie_switch,
                src_to_dst_bandwidth_gbs=14.0,
                dst_to_src_bandwidth_gbs=7.0,
                is_full_duplex=True, alpha_us=50.0,
            ))
