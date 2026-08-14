"""GB300 NVL cluster and aggregated-scope presets."""

from collections import Counter
from typing import List

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


def _validate_nvl_scope(nvl_scope: int, max_scope: int) -> None:
    if nvl_scope < 4 or nvl_scope > max_scope or nvl_scope % 4 != 0:
        raise ValueError(
            f"nvl_scope must be divisible by 4 and between 4 and {max_scope}; "
            f"got {nvl_scope}")


class GB300Cluster(HardwareGraph):
    """GB300 cluster with one NVLink domain represented as one node.

    ``nvl_scope`` controls the number of GPUs in each node. Every four GPUs
    form one compute-tray unit with two Grace CPUs, four ConnectX-8 NICs, and
    four SSDs. One Grace CPU is paired with two GPUs over NVLink-C2C, and the
    two Grace CPUs in each compute-tray unit are also connected by NVLink-C2C.
    All GPUs in the node share one logical fifth-generation NVSwitch fabric.
    """

    max_scope = 72

    def __init__(self, nvl_scope: int, n_nodes: int = 1):
        _validate_nvl_scope(nvl_scope, self.max_scope)
        super().__init__()
        self.nvl_scope = nvl_scope

        ib_switch = Compute(name="ib-switch", kind="switch")
        self.add_node(ib_switch)

        for node in range(n_nodes):
            p = f"n{node}-"
            n_cpus = nvl_scope // 2

            gpus = [Compute(name=f"{p}nvidia-gb300-{i}", tflops={
                "fp4": 15000.0, "fp8": 5000.0,
                "bf16": 2500.0, "fp16": 2500.0, "fp32": 1250.0,
            }, kind="gpu") for i in range(nvl_scope)]

            nvswitch = Compute(name=f"{p}nvswitch", kind="switch")

            cpus = [Compute(name=f"{p}nvidia-grace-{i}", kind="cpu")
                    for i in range(n_cpus)]

            nics = [Compute(name=f"{p}mellanox-cx8-{i}", kind="nic")
                    for i in range(nvl_scope)]

            hbms = [Memory(name=f"{p}hbm3e-{i}", capacity_gb=288.0,
                           kind="hbm") for i in range(nvl_scope)]
            drams = [Memory(name=f"{p}dram-{i}", capacity_gb=480.0,
                            kind="dram") for i in range(n_cpus)]
            ssds = [Memory(name=f"{p}ssd-{i}", capacity_gb=30720.0,
                           kind="ssd") for i in range(nvl_scope)]

            for comp in gpus + [nvswitch] + cpus + nics:
                self.add_node(comp)
            for mem in hbms + drams + ssds:
                self.add_node(mem)

            for i in range(nvl_scope):
                cpu = cpus[i // 2]
                self.add_edge(FabricEdge(
                    name="hbm", src=gpus[i], dst=hbms[i],
                    src_to_dst_bandwidth_gbs=8000.0,
                    dst_to_src_bandwidth_gbs=8000.0,
                    is_full_duplex=False, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="nvlink", src=gpus[i], dst=nvswitch,
                    src_to_dst_bandwidth_gbs=900.0,
                    dst_to_src_bandwidth_gbs=900.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="nvlink-c2c", src=cpu, dst=gpus[i],
                    src_to_dst_bandwidth_gbs=450.0,
                    dst_to_src_bandwidth_gbs=450.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=gpus[i], dst=nics[i],
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
                self.add_edge(FabricEdge(
                    name="pcie", src=ssds[i], dst=cpu,
                    src_to_dst_bandwidth_gbs=14.0,
                    dst_to_src_bandwidth_gbs=7.0,
                    is_full_duplex=True, alpha_us=50.0,
                ))

            for s in range(n_cpus):
                self.add_edge(FabricEdge(
                    name="dram", src=cpus[s], dst=drams[s],
                    src_to_dst_bandwidth_gbs=384.0,
                    dst_to_src_bandwidth_gbs=384.0,
                    is_full_duplex=False, alpha_us=0.1,
                ))

            for s in range(0, n_cpus, 2):
                self.add_edge(FabricEdge(
                    name="nvlink-c2c", src=cpus[s], dst=cpus[s + 1],
                    src_to_dst_bandwidth_gbs=450.0,
                    dst_to_src_bandwidth_gbs=450.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))

    def find_aggregate_bandwidth(self, devices: List[Compute]) -> float:
        """Use NVLink within a node and capped multi-rail IB across nodes."""
        if len(devices) < 2:
            return float("inf")
        nodes = Counter(d.name.split("-")[0] for d in devices)
        if len(nodes) == 1:
            return 900.0
        return min(min(nodes.values()) * 100.0, 800.0)


class GB300SuperChip(HardwareGraph):
    """Aggregate one GB300 NVLink scope into single logical components."""

    max_scope = GB300Cluster.max_scope

    def __init__(self, nvl_scope: int):
        _validate_nvl_scope(nvl_scope, self.max_scope)
        super().__init__()
        self.nvl_scope = nvl_scope

        n_cpus = nvl_scope // 2
        gpu = Compute(name="n0-nvidia-gb300-0", tflops={
            "fp4": 15000.0 * nvl_scope,
            "fp8": 5000.0 * nvl_scope,
            "bf16": 2500.0 * nvl_scope,
            "fp16": 2500.0 * nvl_scope,
            "fp32": 1250.0 * nvl_scope,
        }, kind="gpu")
        cpu = Compute(name="n0-nvidia-grace-0", kind="cpu")
        nic = Compute(name="n0-mellanox-cx8-0", kind="nic")
        ib_switch = Compute(name="ib-switch", kind="switch")

        hbm = Memory(
            name="n0-hbm3e-0", capacity_gb=288.0 * nvl_scope,
            kind="hbm")
        dram = Memory(
            name="n0-dram-0", capacity_gb=480.0 * n_cpus,
            kind="dram")
        ssd = Memory(
            name="n0-ssd", capacity_gb=30720.0 * nvl_scope,
            kind="ssd")

        for comp in [gpu, cpu, nic, ib_switch]:
            self.add_node(comp)
        for mem in [hbm, dram, ssd]:
            self.add_node(mem)

        self.add_edge(FabricEdge(
            name="hbm", src=gpu, dst=hbm,
            src_to_dst_bandwidth_gbs=8000.0 * nvl_scope,
            dst_to_src_bandwidth_gbs=8000.0 * nvl_scope,
            is_full_duplex=False, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="nvlink-c2c", src=cpu, dst=gpu,
            src_to_dst_bandwidth_gbs=900.0 * n_cpus,
            dst_to_src_bandwidth_gbs=900.0 * n_cpus,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="dram", src=cpu, dst=dram,
            src_to_dst_bandwidth_gbs=384.0 * n_cpus,
            dst_to_src_bandwidth_gbs=384.0 * n_cpus,
            is_full_duplex=False, alpha_us=0.1,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=gpu, dst=nic,
            src_to_dst_bandwidth_gbs=128.0 * nvl_scope,
            dst_to_src_bandwidth_gbs=128.0 * nvl_scope,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="infiniband", src=nic, dst=ib_switch,
            src_to_dst_bandwidth_gbs=100.0 * nvl_scope,
            dst_to_src_bandwidth_gbs=100.0 * nvl_scope,
            is_full_duplex=True, alpha_us=1.0,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=ssd, dst=cpu,
            src_to_dst_bandwidth_gbs=14.0 * nvl_scope,
            dst_to_src_bandwidth_gbs=7.0 * nvl_scope,
            is_full_duplex=True, alpha_us=50.0,
        ))
