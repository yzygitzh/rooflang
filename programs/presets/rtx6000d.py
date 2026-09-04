# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""RTX 6000D Ethernet supernode and aggregated-scope presets."""

from collections import Counter
from typing import List

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


def _validate_eth_scope(eth_scope: int) -> None:
    if eth_scope < 8 or eth_scope % 8 != 0:
        raise ValueError(
            f"eth_scope must be divisible by 8 and at least 8; "
            f"got {eth_scope}")


class RTX6000DCluster(HardwareGraph):
    """RTX 6000D cluster with one Ethernet scope represented as one node.

    ``eth_scope`` controls the number of GPUs in each node. Every four GPUs
    share one B300-equivalent host CPU, while each GPU has one local GDDR7
    memory and one SSD. GPUs use an 800 Gb/s logical Ethernet fabric inside
    the scope.
    """

    def __init__(self, eth_scope: int, n_nodes: int = 1):
        _validate_eth_scope(eth_scope)
        super().__init__()
        self.eth_scope = eth_scope

        for node in range(n_nodes):
            p = f"n{node}-"
            n_cpus = eth_scope // 4

            gpus = [Compute(name=f"{p}nvidia-rtx-6000d-{i}", tflops={
                "fp4": 593.0, "fp8": 296.0,
                "bf16": 148.0, "fp16": 148.0, "fp32": 74.0,
            }, kind="gpu") for i in range(eth_scope)]

            eth_switch = Compute(
                name=f"{p}eth-supernode-switch", kind="switch")

            cpus = [Compute(name=f"{p}intel-xeon-6767p-{i}", tflops={
                "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
            }, kind="cpu") for i in range(n_cpus)]

            # Keep accelerator-local memory classified as HBM so existing
            # placement and memory-feasibility logic also applies to GDDR7.
            gddrs = [Memory(name=f"{p}gddr7-{i}", capacity_gb=84.0,
                            kind="hbm") for i in range(eth_scope)]
            drams = [Memory(name=f"{p}ddr5-{i}", capacity_gb=1536.0,
                            kind="dram") for i in range(n_cpus)]
            ssds = [Memory(name=f"{p}ssd-{i}", capacity_gb=256000.0,
                           kind="ssd") for i in range(eth_scope)]

            for comp in gpus + [eth_switch] + cpus:
                self.add_node(comp)
            for mem in gddrs + drams + ssds:
                self.add_node(mem)

            for i in range(eth_scope):
                cpu = cpus[i // 4]
                self.add_edge(FabricEdge(
                    name="gddr7", src=gpus[i], dst=gddrs[i],
                    src_to_dst_bandwidth_gbs=1398.0,
                    dst_to_src_bandwidth_gbs=1398.0,
                    is_full_duplex=False, alpha_us=0.5,
                ))
                self.add_edge(FabricEdge(
                    name="eth", src=gpus[i], dst=eth_switch,
                    src_to_dst_bandwidth_gbs=100.0,
                    dst_to_src_bandwidth_gbs=100.0,
                    is_full_duplex=True, alpha_us=1.0,
                ))
                self.add_edge(FabricEdge(
                    name="pcie", src=cpu, dst=gpus[i],
                    src_to_dst_bandwidth_gbs=64.0,
                    dst_to_src_bandwidth_gbs=64.0,
                    is_full_duplex=True, alpha_us=0.5,
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
        """Use 800 Gb/s aggregate Ethernet bandwidth."""
        if len(devices) < 2:
            return float("inf")
        nodes = Counter(d.name.split("-")[0] for d in devices)
        if len(nodes) == 1:
            return 100.0
        raise ValueError("RTX 6000D has no scale-out Ethernet fabric")


class RTX6000DSuperChip(HardwareGraph):
    """Aggregate one RTX 6000D Ethernet scope into logical components."""

    def __init__(self, eth_scope: int):
        _validate_eth_scope(eth_scope)
        super().__init__()
        self.eth_scope = eth_scope

        n_cpus = eth_scope // 4
        gpu = Compute(name="n0-nvidia-rtx-6000d-0", tflops={
            "fp4": 593.0 * eth_scope,
            "fp8": 296.0 * eth_scope,
            "bf16": 148.0 * eth_scope,
            "fp16": 148.0 * eth_scope,
            "fp32": 74.0 * eth_scope,
        }, kind="gpu")
        cpu = Compute(name="n0-intel-xeon-6767p-0", tflops={
            "bf16": 255.29 * n_cpus,
            "fp16": 255.29 * n_cpus,
            "int8": 511.18 * n_cpus,
        }, kind="cpu")
        gddr = Memory(
            name="n0-gddr7-0", capacity_gb=84.0 * eth_scope,
            kind="hbm")
        dram = Memory(
            name="n0-ddr5-0", capacity_gb=1536.0 * n_cpus,
            kind="dram")
        ssd = Memory(
            name="n0-ssd", capacity_gb=256000.0 * eth_scope,
            kind="ssd")

        for comp in [gpu, cpu]:
            self.add_node(comp)
        for mem in [gddr, dram, ssd]:
            self.add_node(mem)

        self.add_edge(FabricEdge(
            name="gddr7", src=gpu, dst=gddr,
            src_to_dst_bandwidth_gbs=1398.0 * eth_scope,
            dst_to_src_bandwidth_gbs=1398.0 * eth_scope,
            is_full_duplex=False, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=cpu, dst=gpu,
            src_to_dst_bandwidth_gbs=64.0 * eth_scope,
            dst_to_src_bandwidth_gbs=64.0 * eth_scope,
            is_full_duplex=True, alpha_us=0.5,
        ))
        self.add_edge(FabricEdge(
            name="dram", src=cpu, dst=dram,
            src_to_dst_bandwidth_gbs=332.8 * n_cpus,
            dst_to_src_bandwidth_gbs=332.8 * n_cpus,
            is_full_duplex=False, alpha_us=0.1,
        ))
        self.add_edge(FabricEdge(
            name="pcie", src=ssd, dst=cpu,
            src_to_dst_bandwidth_gbs=14.0 * eth_scope,
            dst_to_src_bandwidth_gbs=7.0 * eth_scope,
            is_full_duplex=True, alpha_us=50.0,
        ))
