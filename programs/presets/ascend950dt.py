# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

"""Ascend 950DT Unified Bus cluster and aggregated-scope presets."""

from collections import Counter
from typing import List

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.graph import FabricEdge, HardwareGraph


def _validate_ub_scope(ub_scope: int, max_scope: int) -> None:
    if ub_scope < 8 or ub_scope > max_scope or ub_scope % 8 != 0:
        raise ValueError(
            f"ub_scope must be divisible by 8 and between 8 and {max_scope}; "
            f"got {ub_scope}")


class Ascend950DTCluster(HardwareGraph):
    """Ascend 950DT cluster with one UB scope represented as one node.

    ``ub_scope`` controls the number of NPUs in each node. Every eight NPUs
    share two B300-equivalent host CPUs, while each NPU has one local memory
    and one SSD. Each node has an L1 UB switch; groups of up to sixteen L1
    switches share an L2 switch, and multiple L2 switches share one L3.
    """

    max_scope = 64

    def __init__(self, ub_scope: int, n_nodes: int = 1):
        _validate_ub_scope(ub_scope, self.max_scope)
        super().__init__()
        self.ub_scope = ub_scope

        ub_switches_l1 = []
        machine_by_gpu = {}
        l1_by_gpu = {}
        l2_by_l1 = {}

        for node in range(n_nodes):
            p = f"n{node}-"
            n_cpus = ub_scope // 4

            gpus = [Compute(name=f"{p}huawei-ascend-950dt-{i}", tflops={
                "fp4": 1946.0, "fp8": 973.0,
                "bf16": 486.0, "fp16": 486.0, "fp32": 243.0,
            }, kind="gpu") for i in range(ub_scope)]

            ub_switch_l1 = Compute(
                name=f"{p}ub-switch-l1", kind="switch")
            ub_switches_l1.append(ub_switch_l1)
            for rank, gpu in enumerate(gpus):
                machine_by_gpu[gpu] = (node, rank // 8)
                l1_by_gpu[gpu] = ub_switch_l1

            cpus = [Compute(name=f"{p}intel-xeon-6767p-{i}", tflops={
                "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
            }, kind="cpu") for i in range(n_cpus)]

            hbms = [Memory(name=f"{p}hbm-{i}", capacity_gb=144.0,
                           kind="hbm") for i in range(ub_scope)]
            drams = [Memory(name=f"{p}ddr5-{i}", capacity_gb=1536.0,
                            kind="dram") for i in range(n_cpus)]
            ssds = [Memory(name=f"{p}ssd-{i}", capacity_gb=256000.0,
                           kind="ssd") for i in range(ub_scope)]

            for comp in gpus + [ub_switch_l1] + cpus:
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
                    name="ub-l1", src=gpus[i], dst=ub_switch_l1,
                    src_to_dst_bandwidth_gbs=448.0,
                    dst_to_src_bandwidth_gbs=448.0,
                    is_full_duplex=True, alpha_us=0.5,
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

            for group_start in range(0, ub_scope, 8):
                group = gpus[group_start:group_start + 8]
                for i, src in enumerate(group):
                    for dst in group[i + 1:]:
                        self.add_edge(FabricEdge(
                            name="ub-mesh", src=src, dst=dst,
                            src_to_dst_bandwidth_gbs=56.0,
                            dst_to_src_bandwidth_gbs=56.0,
                            is_full_duplex=True, alpha_us=0.5,
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

        ub_switches_l2 = []
        if len(ub_switches_l1) >= 2:
            n_l2 = (len(ub_switches_l1) + 15) // 16
            quotient, remainder = divmod(len(ub_switches_l1), n_l2)
            offset = 0
            for index in range(n_l2):
                group_size = quotient + (index < remainder)
                ub_switch_l2 = Compute(
                    name=f"ub-switch-l2-{index}", kind="switch")
                self.add_node(ub_switch_l2)
                ub_switches_l2.append(ub_switch_l2)
                for ub_switch_l1 in ub_switches_l1[
                        offset:offset + group_size]:
                    l2_by_l1[ub_switch_l1] = ub_switch_l2
                    self.add_edge(FabricEdge(
                        name="ub-l2", src=ub_switch_l1, dst=ub_switch_l2,
                        src_to_dst_bandwidth_gbs=12800.0,
                        dst_to_src_bandwidth_gbs=12800.0,
                        is_full_duplex=True, alpha_us=1.0,
                    ))
                offset += group_size

        if len(ub_switches_l2) >= 2:
            ub_switch_l3 = Compute(name="ub-switch-l3", kind="switch")
            self.add_node(ub_switch_l3)
            for ub_switch_l2 in ub_switches_l2:
                self.add_edge(FabricEdge(
                    name="ub-l3", src=ub_switch_l2, dst=ub_switch_l3,
                    src_to_dst_bandwidth_gbs=51200.0,
                    dst_to_src_bandwidth_gbs=51200.0,
                    is_full_duplex=True, alpha_us=1.0,
                ))

        self._machine_by_gpu = machine_by_gpu
        self._l1_by_gpu = l1_by_gpu
        self._l2_by_l1 = l2_by_l1

    def find_aggregate_bandwidth(self, devices: List[Compute]) -> float:
        """Return aggregate collective bandwidth across the UB hierarchy."""
        if len(devices) < 2:
            return float("inf")
        if any(device not in self._machine_by_gpu for device in devices):
            return super().find_aggregate_bandwidth(devices)

        machine_counts = Counter(
            self._machine_by_gpu[device] for device in devices)
        if len(machine_counts) == 1:
            return min(120.0 * (len(devices) - 1), 840.0)

        min_machine_devices = min(machine_counts.values())
        l1_counts = Counter(self._l1_by_gpu[device] for device in devices)
        l1_switches = set(l1_counts)
        if len(l1_switches) == 1:
            per_machine_bandwidth = min(
                448.0,
                float("inf") if min_machine_devices == 1
                else 392.0 / (min_machine_devices - 1),
            )
            return (
                448.0 - per_machine_bandwidth
                + per_machine_bandwidth * min_machine_devices
            )

        min_l1_devices = min(l1_counts.values())
        l2_counts = Counter(
            self._l2_by_l1[self._l1_by_gpu[device]]
            for device in devices
        )
        l2_switches = set(l2_counts)
        if len(l2_switches) == 1:
            per_machine_bandwidth = min(
                200.0,
                float("inf") if min_machine_devices == 1
                else 392.0 / (min_machine_devices - 1),
            )
            per_l1_bandwidth = min(
                200.0,
                float("inf") if min_l1_devices == 1
                else 448.0 / (min_l1_devices - 1),
            )
            return max(
                200.0 - per_machine_bandwidth
                + per_machine_bandwidth * min_machine_devices,
                200.0 - per_l1_bandwidth
                + per_l1_bandwidth * min_l1_devices,
            )

        min_l2_devices = min(l2_counts.values())
        per_machine_bandwidth = min(
            50.0,
            float("inf") if min_machine_devices == 1
            else 392.0 / (min_machine_devices - 1),
        )
        per_l1_bandwidth = min(
            50.0,
            float("inf") if min_l1_devices == 1
            else 448.0 / (min_l1_devices - 1),
        )
        per_l2_bandwidth = min(
            50.0,
            float("inf") if min_l2_devices == 1
            else 200.0 / (min_l2_devices - 1),
        )
        return max(
            50.0 - per_machine_bandwidth
            + per_machine_bandwidth * min_machine_devices,
            50.0 - per_l1_bandwidth
            + per_l1_bandwidth * min_l1_devices,
            50.0 - per_l2_bandwidth
            + per_l2_bandwidth * min_l2_devices,
        )


class Ascend950DTSuperChip(HardwareGraph):
    """Aggregate one Ascend 950DT UB scope into logical components."""

    max_scope = Ascend950DTCluster.max_scope

    def __init__(self, ub_scope: int):
        _validate_ub_scope(ub_scope, self.max_scope)
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
            name="n0-ssd", capacity_gb=256000.0 * ub_scope,
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
