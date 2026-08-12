"""Ascend 950DT Unified Bus cluster and aggregated-scope presets."""

from collections import Counter
from typing import List

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


def _validate_ub_scope(ub_scope: int) -> None:
    if ub_scope < 8 or ub_scope > 64 or ub_scope % 8 != 0:
        raise ValueError(
            "ub_scope must be divisible by 8 and between 8 and 64; "
            f"got {ub_scope}")


class Ascend950DTCluster(HardwareGraph):
    """Ascend 950DT cluster with one UB scope represented as one node.

    ``ub_scope`` controls the number of NPUs in each node. Every eight NPUs
    share two B300-equivalent host CPUs, while each NPU has one local memory
    and one SSD. All NPUs in the node share one logical UB switch.
    """

    def __init__(self, ub_scope: int, n_nodes: int = 1):
        _validate_ub_scope(ub_scope)
        super().__init__()
        self.ub_scope = ub_scope

        eth_switch = Compute(name="eth-switch", kind="switch")
        self.add_node(eth_switch)

        for node in range(n_nodes):
            p = f"n{node}-"
            n_cpus = ub_scope // 4

            gpus = [Compute(name=f"{p}huawei-ascend-950dt-{i}", tflops={
                "fp4": 1946.0, "fp8": 973.0,
                "bf16": 486.0, "fp16": 486.0, "fp32": 243.0,
            }, kind="gpu") for i in range(ub_scope)]

            ub_switch = Compute(name=f"{p}ub-switch", kind="switch")

            cpus = [Compute(name=f"{p}intel-xeon-6767p-{i}", tflops={
                "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
            }, kind="cpu") for i in range(n_cpus)]

            hbms = [Memory(name=f"{p}hbm-{i}", capacity_gb=144.0,
                           kind="hbm") for i in range(ub_scope)]
            drams = [Memory(name=f"{p}ddr5-{i}", capacity_gb=1536.0,
                            kind="dram") for i in range(n_cpus)]
            ssds = [Memory(name=f"{p}ssd-{i}", capacity_gb=3840.0,
                           kind="ssd") for i in range(ub_scope)]

            for comp in gpus + [ub_switch] + cpus:
                self.add_node(comp)
            for mem in hbms + drams + ssds:
                self.add_node(mem)

            for i in range(ub_scope):
                cpu = cpus[i // 4]
                self.add_edge(FabricEdge(
                    name="hbm", src=gpus[i], dst=hbms[i],
                    src_to_dst_bandwidth_gbs=4000.0,
                    dst_to_src_bandwidth_gbs=4000.0,
                    is_full_duplex=False, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="ub", src=gpus[i], dst=ub_switch,
                    src_to_dst_bandwidth_gbs=896.0,
                    dst_to_src_bandwidth_gbs=896.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=cpu, dst=gpus[i],
                    src_to_dst_bandwidth_gbs=64.0,
                    dst_to_src_bandwidth_gbs=64.0,
                    is_full_duplex=True, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="uboe", src=gpus[i], dst=eth_switch,
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
                    src_to_dst_bandwidth_gbs=332.8,
                    dst_to_src_bandwidth_gbs=332.8,
                    is_full_duplex=False, alpha_us=0.1,
                ))

            for s in range(0, n_cpus, 2):
                self.add_edge(FabricEdge(
                    name="qpi", src=cpus[s], dst=cpus[s + 1],
                    src_to_dst_bandwidth_gbs=192.0,
                    dst_to_src_bandwidth_gbs=192.0,
                    is_full_duplex=True, alpha_us=0.05,
                ))

    def find_aggregate_bandwidth(self, devices: List[Compute]) -> float:
        """Use UB within a node and multi-rail UBoE across nodes."""
        if len(devices) < 2:
            return float("inf")
        nodes = Counter(d.name.split("-")[0] for d in devices)
        if len(nodes) == 1:
            return 896.0
        return min(nodes.values()) * 100.0


class Ascend950DTSuperChip(HardwareGraph):
    """Aggregate one Ascend 950DT UB scope into logical components."""

    def __init__(self, ub_scope: int):
        _validate_ub_scope(ub_scope)
        super().__init__()
        self.ub_scope = ub_scope

        n_cpus = ub_scope // 4
        gpu = Compute(name="n0-huawei-ascend-950dt-0", tflops={
            "fp4": 1946.0 * ub_scope,
            "fp8": 973.0 * ub_scope,
            "bf16": 486.0 * ub_scope,
            "fp16": 486.0 * ub_scope,
            "fp32": 243.0 * ub_scope,
        }, kind="gpu")
        cpu = Compute(name="n0-intel-xeon-6767p-0", tflops={
            "bf16": 255.29 * n_cpus,
            "fp16": 255.29 * n_cpus,
            "int8": 511.18 * n_cpus,
        }, kind="cpu")
        eth_switch = Compute(name="eth-switch", kind="switch")

        hbm = Memory(
            name="n0-hbm-0", capacity_gb=144.0 * ub_scope,
            kind="hbm")
        dram = Memory(
            name="n0-ddr5-0", capacity_gb=1536.0 * n_cpus,
            kind="dram")
        ssd = Memory(
            name="n0-ssd", capacity_gb=3840.0 * ub_scope,
            kind="ssd")

        for comp in [gpu, cpu, eth_switch]:
            self.add_node(comp)
        for mem in [hbm, dram, ssd]:
            self.add_node(mem)

        self.add_edge(FabricEdge(
            name="hbm", src=gpu, dst=hbm,
            src_to_dst_bandwidth_gbs=4000.0 * ub_scope,
            dst_to_src_bandwidth_gbs=4000.0 * ub_scope,
            is_full_duplex=False, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=cpu, dst=gpu,
            src_to_dst_bandwidth_gbs=64.0 * ub_scope,
            dst_to_src_bandwidth_gbs=64.0 * ub_scope,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="dram", src=cpu, dst=dram,
            src_to_dst_bandwidth_gbs=332.8 * n_cpus,
            dst_to_src_bandwidth_gbs=332.8 * n_cpus,
            is_full_duplex=False, alpha_us=0.1,
        ))
        self.add_edge(FabricEdge(
            name="uboe", src=gpu, dst=eth_switch,
            src_to_dst_bandwidth_gbs=100.0 * ub_scope,
            dst_to_src_bandwidth_gbs=100.0 * ub_scope,
            is_full_duplex=True, alpha_us=1.0,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=ssd, dst=cpu,
            src_to_dst_bandwidth_gbs=14.0 * ub_scope,
            dst_to_src_bandwidth_gbs=7.0 * ub_scope,
            is_full_duplex=True, alpha_us=50.0,
        ))
