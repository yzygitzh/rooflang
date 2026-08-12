"""Unit tests for hardware preset classes."""

import pytest

from rooflang.language.hardware.component import Compute, Memory
from rooflang.programs.presets.ascend950dt import (
    Ascend950DTCluster,
    Ascend950DTSuperChip,
)
from rooflang.programs.presets.b300 import B300Cluster, B300SuperChip
from rooflang.programs.presets.gb300 import GB300Cluster, GB300SuperChip
from rooflang.programs.presets.gh200 import GH200Cluster, GH200SuperChip
from rooflang.programs.presets.h200 import H200Cluster, H200SuperChip


class TestB300AggregateOverride:
    def test_component_kinds(self):
        hw = B300Cluster(n_nodes=1)
        components = {component.name: component for component in hw.nodes}

        assert components["n0-nvidia-b300-sxm-0"].kind == "gpu"
        assert components["n0-intel-xeon-6767p-0"].kind == "cpu"
        assert components["n0-mellanox-cx8-0"].kind == "nic"
        assert components["n0-hbm3e-0"].kind == "hbm"
        assert components["n0-ddr5-0"].kind == "dram"
        assert components["n0-ssd-0"].kind == "ssd"
        assert components["n0-nvswitch"].kind == "switch"

    def test_intra_node(self):
        hw = B300Cluster(n_nodes=1)
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert hw.find_aggregate_bandwidth(gpus) == 900.0

    def test_inter_node(self):
        hw = B300Cluster(n_nodes=2)
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert hw.find_aggregate_bandwidth(gpus) == 800.0

    def test_single_device_returns_inf(self):
        hw = B300Cluster(n_nodes=1)
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert hw.find_aggregate_bandwidth(gpus[:1]) == float("inf")

    def test_eight_ssds_attach_directly_to_hgx(self):
        hw = B300Cluster(n_nodes=1)
        components = {component.name: component for component in hw.nodes}
        hgx = components["n0-hgx-pcie-switch"]
        ssds = [components[f"n0-ssd-{index}"] for index in range(8)]

        assert "n0-nvme-pcie-switch" not in components
        for ssd in ssds:
            assert ssd.capacity_gb == 3840.0
            path = hw.find_fabric_path(ssd, hgx)
            assert len(path) == 1
            assert path[0].src_to_dst_bandwidth_gbs == 14.0
            assert path[0].dst_to_src_bandwidth_gbs == 7.0

    def test_gpu_reaches_nic_through_hgx(self):
        hw = B300Cluster(n_nodes=1)
        components = {component.name: component for component in hw.nodes}
        gpu = components["n0-nvidia-b300-sxm-0"]
        nic = components["n0-mellanox-cx8-0"]
        hgx = components["n0-hgx-pcie-switch"]

        path = hw.find_fabric_path(gpu, nic)
        assert len(path) == 2
        assert all(fabric.name == "pcie" for fabric in path)
        assert any(hgx in (fabric.src, fabric.dst) for fabric in path)


class TestB300SuperChip:
    def test_single_gpu(self):
        hw = B300SuperChip()
        gpus = [n for n in hw.nodes
                if isinstance(n, Compute) and "nvidia-b300" in n.name]
        assert len(gpus) == 1

    def test_hbm_capacity(self):
        hw = B300SuperChip()
        hbm = [n for n in hw.nodes
                if isinstance(n, Memory) and "hbm3e" in n.name]
        assert len(hbm) == 1
        assert hbm[0].capacity_gb == 2304.0

    def test_topology_has_nodes(self):
        hw = B300SuperChip()
        assert len(hw.nodes) > 0

    def test_find_local_memory(self):
        hw = B300SuperChip()
        gpu = [n for n in hw.nodes
               if isinstance(n, Compute) and "nvidia-b300" in n.name][0]
        mem = hw.find_local_memory(gpu)
        assert isinstance(mem, Memory)
        assert "hbm3e" in mem.name

    def test_aggregates_one_node(self):
        hw = B300SuperChip()
        components = {component.name: component for component in hw.nodes}

        gpu = components["n0-nvidia-b300-sxm-0"]
        cpu = components["n0-intel-xeon-6767p-0"]
        assert gpu.tflops == {
            "fp4": 108000.0, "fp8": 36000.0,
            "bf16": 18000.0, "fp16": 18000.0, "fp32": 9000.0,
        }
        assert cpu.tflops == {
            "bf16": 510.58, "fp16": 510.58, "int8": 1022.36,
        }
        assert components["n0-ddr5-0"].capacity_gb == 3072.0
        assert components["n0-ssd"].capacity_gb == 30720.0
        assert "n0-mellanox-cx8-0" in components
        assert not any("nvswitch" in name for name in components)
        assert not any("nvme-pcie-switch" in name for name in components)

    def test_aggregates_node_bandwidths(self):
        hw = B300SuperChip()
        components = {component.name: component for component in hw.nodes}

        hbm = hw.find_fabric(
            components["n0-nvidia-b300-sxm-0"],
            components["n0-hbm3e-0"])
        gpu_pcie = hw.find_fabric(
            components["n0-nvidia-b300-sxm-0"],
            components["n0-hgx-pcie-switch"])
        dram = hw.find_fabric(
            components["n0-intel-xeon-6767p-0"],
            components["n0-ddr5-0"])
        infiniband = hw.find_fabric(
            components["n0-mellanox-cx8-0"],
            components["ib-switch"])
        ssd_pcie = hw.find_fabric(
            components["n0-ssd"], components["n0-hgx-pcie-switch"])
        gpu_nic_path = hw.find_fabric_path(
            components["n0-nvidia-b300-sxm-0"],
            components["n0-mellanox-cx8-0"])

        assert hbm.src_to_dst_bandwidth_gbs == 62000.0
        assert gpu_pcie.src_to_dst_bandwidth_gbs == 1024.0
        assert dram.src_to_dst_bandwidth_gbs == 665.6
        assert infiniband.src_to_dst_bandwidth_gbs == 800.0
        assert ssd_pcie.src_to_dst_bandwidth_gbs == 112.0
        assert ssd_pcie.dst_to_src_bandwidth_gbs == 56.0
        assert len(gpu_nic_path) == 2


class TestH200Cluster:
    def test_component_kinds(self):
        hw = H200Cluster(n_nodes=1)
        components = {component.name: component for component in hw.nodes}

        assert components["n0-nvidia-h200-sxm-0"].kind == "gpu"
        assert components["n0-intel-xeon-6767p-0"].kind == "cpu"
        assert components["n0-mellanox-cx7-0"].kind == "nic"
        assert components["n0-hbm3e-0"].kind == "hbm"
        assert components["n0-ddr5-0"].kind == "dram"
        assert components["n0-ssd-0"].kind == "ssd"
        assert components["n0-nvswitch"].kind == "switch"

    def test_gpu_and_hbm_specs(self):
        hw = H200Cluster(n_nodes=1)
        components = {component.name: component for component in hw.nodes}

        gpu = components["n0-nvidia-h200-sxm-0"]
        hbm = components["n0-hbm3e-0"]
        hbm_fabric = hw.find_fabric(gpu, hbm)

        assert gpu.tflops == {
            "fp4": 1979.0, "fp8": 1979.0,
            "bf16": 989.5, "fp16": 989.5, "fp32": 494.5,
        }
        assert hbm.capacity_gb == 141.0
        assert hbm_fabric.src_to_dst_bandwidth_gbs == 4800.0

    def test_aggregate_bandwidth(self):
        hw = H200Cluster(n_nodes=2)
        gpus = [component for component in hw.nodes
                if isinstance(component, Compute)
                and "nvidia-h200" in component.name]
        node0_gpus = [gpu for gpu in gpus if gpu.name.startswith("n0-")]

        assert hw.find_aggregate_bandwidth(node0_gpus) == 450.0
        assert hw.find_aggregate_bandwidth(gpus) == 400.0

    def test_nvlink_and_pcie_bandwidths(self):
        hw = H200Cluster(n_nodes=1)
        components = {component.name: component for component in hw.nodes}
        gpu = components["n0-nvidia-h200-sxm-0"]
        nic = components["n0-mellanox-cx7-0"]
        nvswitch = components["n0-nvswitch"]
        hgx = components["n0-hgx-pcie-switch"]

        nvlink = hw.find_fabric(gpu, nvswitch)
        gpu_pcie = hw.find_fabric(gpu, hgx)
        nic_pcie = hw.find_fabric(nic, hgx)
        gpu_nic_path = hw.find_fabric_path(gpu, nic)

        assert nvlink.src_to_dst_bandwidth_gbs == 450.0
        assert nvlink.dst_to_src_bandwidth_gbs == 450.0
        assert gpu_pcie.src_to_dst_bandwidth_gbs == 64.0
        assert gpu_pcie.dst_to_src_bandwidth_gbs == 64.0
        assert nic_pcie.src_to_dst_bandwidth_gbs == 64.0
        assert nic_pcie.dst_to_src_bandwidth_gbs == 64.0
        assert len(gpu_nic_path) == 2
        assert all(fabric.name == "pcie" for fabric in gpu_nic_path)

    def test_reuses_b300_cpu_and_ssd_specs(self):
        hw = H200Cluster(n_nodes=1)
        components = {component.name: component for component in hw.nodes}

        assert components["n0-intel-xeon-6767p-0"].tflops == {
            "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
        }
        assert components["n0-ddr5-0"].capacity_gb == 1536.0
        for index in range(8):
            ssd = components[f"n0-ssd-{index}"]
            assert ssd.capacity_gb == 3840.0
            path = hw.find_fabric_path(
                ssd, components["n0-hgx-pcie-switch"])
            assert len(path) == 1
            assert path[0].src_to_dst_bandwidth_gbs == 14.0
            assert path[0].dst_to_src_bandwidth_gbs == 7.0


class TestH200SuperChip:
    def test_aggregates_one_node(self):
        hw = H200SuperChip()
        components = {component.name: component for component in hw.nodes}

        gpu = components["n0-nvidia-h200-sxm-0"]
        assert gpu.tflops == {
            "fp4": 15832.0, "fp8": 15832.0,
            "bf16": 7916.0, "fp16": 7916.0, "fp32": 3956.0,
        }
        assert components["n0-hbm3e-0"].capacity_gb == 1128.0
        assert components["n0-ddr5-0"].capacity_gb == 3072.0
        assert components["n0-ssd"].capacity_gb == 30720.0
        assert "n0-mellanox-cx7-0" in components
        assert not any("nvswitch" in name for name in components)

    def test_aggregates_node_bandwidths(self):
        hw = H200SuperChip()
        components = {component.name: component for component in hw.nodes}

        hbm = hw.find_fabric(
            components["n0-nvidia-h200-sxm-0"],
            components["n0-hbm3e-0"])
        gpu_pcie = hw.find_fabric(
            components["n0-nvidia-h200-sxm-0"],
            components["n0-hgx-pcie-switch"])
        nic_pcie = hw.find_fabric(
            components["n0-mellanox-cx7-0"],
            components["n0-hgx-pcie-switch"])
        infiniband = hw.find_fabric(
            components["n0-mellanox-cx7-0"],
            components["ib-switch"])
        ssd_pcie = hw.find_fabric(
            components["n0-ssd"], components["n0-hgx-pcie-switch"])

        assert hbm.src_to_dst_bandwidth_gbs == 38400.0
        assert gpu_pcie.src_to_dst_bandwidth_gbs == 512.0
        assert nic_pcie.src_to_dst_bandwidth_gbs == 512.0
        assert infiniband.src_to_dst_bandwidth_gbs == 400.0
        assert ssd_pcie.src_to_dst_bandwidth_gbs == 112.0
        assert ssd_pcie.dst_to_src_bandwidth_gbs == 56.0
        assert len(hw.find_fabric_path(
            components["n0-nvidia-h200-sxm-0"],
            components["n0-mellanox-cx7-0"])) == 2


@pytest.mark.parametrize("preset", [GB300Cluster, GB300SuperChip])
@pytest.mark.parametrize("nvl_scope", [0, 3, 6, 76])
def test_gb300_rejects_invalid_nvl_scope(preset, nvl_scope):
    with pytest.raises(ValueError, match="nvl_scope"):
        preset(nvl_scope=nvl_scope)


@pytest.mark.parametrize("preset", [GB300Cluster, GB300SuperChip])
@pytest.mark.parametrize("nvl_scope", [4, 72])
def test_gb300_accepts_nvl_scope_boundaries(preset, nvl_scope):
    assert preset(nvl_scope=nvl_scope).nvl_scope == nvl_scope


class TestGB300Cluster:
    def test_nvl_scope_controls_component_counts(self):
        hw = GB300Cluster(nvl_scope=8)

        assert hw.nvl_scope == 8
        assert sum(component.kind == "gpu" for component in hw.nodes) == 8
        assert sum(component.kind == "cpu" for component in hw.nodes) == 4
        assert sum(component.kind == "nic" for component in hw.nodes) == 8
        assert sum(component.kind == "hbm" for component in hw.nodes) == 8
        assert sum(component.kind == "dram" for component in hw.nodes) == 4
        assert sum(component.kind == "ssd" for component in hw.nodes) == 8

    def test_gpu_specs_and_nvl_bandwidth(self):
        hw = GB300Cluster(nvl_scope=8)
        components = {component.name: component for component in hw.nodes}
        gpu = components["n0-nvidia-gb300-0"]

        assert gpu.tflops == {
            "fp4": 15000.0, "fp8": 5000.0,
            "bf16": 2500.0, "fp16": 2500.0, "fp32": 1250.0,
        }
        assert components["n0-hbm3e-0"].capacity_gb == 288.0
        hbm = hw.find_fabric(gpu, components["n0-hbm3e-0"])
        nvlink = hw.find_fabric(gpu, components["n0-nvswitch"])
        assert hbm.src_to_dst_bandwidth_gbs == 8000.0
        assert nvlink.src_to_dst_bandwidth_gbs == 900.0
        assert nvlink.dst_to_src_bandwidth_gbs == 900.0

        gpus = [component for component in hw.nodes
                if component.kind == "gpu"]
        assert hw.find_aggregate_bandwidth(gpus) == 900.0

    @pytest.mark.parametrize("nvl_scope, expected", [(4, 400.0),
                                                       (12, 800.0)])
    def test_inter_node_bandwidth_is_capped(self, nvl_scope, expected):
        hw = GB300Cluster(nvl_scope=nvl_scope, n_nodes=2)
        gpus = [component for component in hw.nodes
                if component.kind == "gpu"]

        assert hw.find_aggregate_bandwidth(gpus) == expected

    def test_grace_gpu_c2c_and_local_dram(self):
        hw = GB300Cluster(nvl_scope=8)
        components = {component.name: component for component in hw.nodes}
        cpu = components["n0-nvidia-grace-0"]

        for gpu_index in (0, 1):
            c2c = hw.find_fabric(
                cpu, components[f"n0-nvidia-gb300-{gpu_index}"])
            assert c2c.name == "nvlink-c2c"
            assert c2c.src_to_dst_bandwidth_gbs == 450.0
            assert c2c.dst_to_src_bandwidth_gbs == 450.0

        assert hw.find_local_memory(cpu) is components["n0-dram-0"]
        assert components["n0-dram-0"].capacity_gb == 480.0

        cpu_c2c = hw.find_fabric(
            components["n0-nvidia-grace-0"],
            components["n0-nvidia-grace-1"])
        assert cpu_c2c.name == "nvlink-c2c"
        assert cpu_c2c.src_to_dst_bandwidth_gbs == 450.0
        assert cpu_c2c.dst_to_src_bandwidth_gbs == 450.0

        second_tray_c2c = hw.find_fabric(
            components["n0-nvidia-grace-2"],
            components["n0-nvidia-grace-3"])
        assert second_tray_c2c.name == "nvlink-c2c"

    def test_uses_ib_switch(self):
        hw = GB300Cluster(nvl_scope=4)
        components = {component.name: component for component in hw.nodes}

        fabric = hw.find_fabric(
            components["n0-mellanox-cx8-0"], components["ib-switch"])
        assert fabric.name == "infiniband"

    def test_ssd_specs_match_existing_presets(self):
        hw = GB300Cluster(nvl_scope=4)
        components = {component.name: component for component in hw.nodes}

        for index in range(4):
            ssd = components[f"n0-ssd-{index}"]
            cpu = components[f"n0-nvidia-grace-{index // 2}"]
            fabric = hw.find_fabric(ssd, cpu)
            assert ssd.capacity_gb == 3840.0
            assert fabric.src_to_dst_bandwidth_gbs == 14.0
            assert fabric.dst_to_src_bandwidth_gbs == 7.0


class TestGB300SuperChip:
    def test_aggregates_requested_nvl_scope(self):
        hw = GB300SuperChip(nvl_scope=8)
        components = {component.name: component for component in hw.nodes}
        gpu = components["n0-nvidia-gb300-0"]

        assert hw.nvl_scope == 8
        assert gpu.tflops == {
            "fp4": 120000.0, "fp8": 40000.0,
            "bf16": 20000.0, "fp16": 20000.0, "fp32": 10000.0,
        }
        assert components["n0-hbm3e-0"].capacity_gb == 2304.0
        assert components["n0-dram-0"].capacity_gb == 1920.0
        assert components["n0-ssd"].capacity_gb == 30720.0
        assert not any("nvswitch" in component.name
                       for component in hw.nodes)

    def test_aggregates_bandwidths(self):
        hw = GB300SuperChip(nvl_scope=8)
        components = {component.name: component for component in hw.nodes}

        hbm = hw.find_fabric(
            components["n0-nvidia-gb300-0"], components["n0-hbm3e-0"])
        c2c = hw.find_fabric(
            components["n0-nvidia-grace-0"],
            components["n0-nvidia-gb300-0"])
        dram = hw.find_fabric(
            components["n0-nvidia-grace-0"],
            components["n0-dram-0"])
        ssd = hw.find_fabric(
            components["n0-ssd"], components["n0-nvidia-grace-0"])

        assert hbm.src_to_dst_bandwidth_gbs == 64000.0
        assert c2c.src_to_dst_bandwidth_gbs == 3600.0
        assert dram.src_to_dst_bandwidth_gbs == 1536.0
        assert ssd.src_to_dst_bandwidth_gbs == 112.0
        assert ssd.dst_to_src_bandwidth_gbs == 56.0


@pytest.mark.parametrize("preset", [GH200Cluster, GH200SuperChip])
@pytest.mark.parametrize("nvl_scope", [0, 257, 258])
def test_gh200_rejects_invalid_nvl_scope(preset, nvl_scope):
    with pytest.raises(ValueError, match="nvl_scope"):
        preset(nvl_scope=nvl_scope)


@pytest.mark.parametrize("preset", [GH200Cluster, GH200SuperChip])
@pytest.mark.parametrize("nvl_scope", [1, 2, 3, 255, 256])
def test_gh200_accepts_nvl_scope_boundaries(preset, nvl_scope):
    assert preset(nvl_scope=nvl_scope).nvl_scope == nvl_scope


class TestGH200Cluster:
    def test_nvl_scope_controls_component_counts(self):
        hw = GH200Cluster(nvl_scope=4)

        assert hw.nvl_scope == 4
        assert sum(component.kind == "gpu" for component in hw.nodes) == 4
        assert sum(component.kind == "cpu" for component in hw.nodes) == 4
        assert sum(component.kind == "nic" for component in hw.nodes) == 4
        assert sum(component.kind == "hbm" for component in hw.nodes) == 4
        assert sum(component.kind == "dram" for component in hw.nodes) == 4
        assert sum(component.kind == "ssd" for component in hw.nodes) == 4

    def test_superchip_specs_and_links(self):
        hw = GH200Cluster(nvl_scope=2)
        components = {component.name: component for component in hw.nodes}
        gpu = components["n0-nvidia-gh200-0"]
        cpu = components["n0-nvidia-grace-0"]
        nic = components["n0-mellanox-cx7-0"]

        assert gpu.tflops == {
            "fp4": 1979.0, "fp8": 1979.0,
            "bf16": 989.5, "fp16": 989.5, "fp32": 494.5,
        }
        assert cpu.tflops == {"fp64": 3.55}
        assert components["n0-hbm3e-0"].capacity_gb == 144.0
        assert components["n0-dram-0"].capacity_gb == 480.0

        hbm = hw.find_fabric(gpu, components["n0-hbm3e-0"])
        nvlink = hw.find_fabric(gpu, components["n0-nvswitch"])
        c2c = hw.find_fabric(cpu, gpu)
        gpu_nic_path = hw.find_fabric_path(gpu, nic)

        assert hbm.src_to_dst_bandwidth_gbs == 4900.0
        assert nvlink.src_to_dst_bandwidth_gbs == 450.0
        assert nvlink.dst_to_src_bandwidth_gbs == 450.0
        assert c2c.name == "nvlink-c2c"
        assert c2c.src_to_dst_bandwidth_gbs == 450.0
        assert c2c.dst_to_src_bandwidth_gbs == 450.0
        dram = hw.find_fabric(cpu, components["n0-dram-0"])
        assert dram.src_to_dst_bandwidth_gbs == 512.0
        assert [fabric.name for fabric in gpu_nic_path] == [
            "nvlink-c2c", "pcie"]

    def test_aggregate_bandwidth(self):
        hw = GH200Cluster(nvl_scope=4, n_nodes=2)
        gpus = [component for component in hw.nodes
                if component.kind == "gpu"]
        node0_gpus = [gpu for gpu in gpus if gpu.name.startswith("n0-")]

        assert hw.find_aggregate_bandwidth(node0_gpus) == 450.0
        assert hw.find_aggregate_bandwidth(gpus) == 200.0

    def test_ssd_specs_match_existing_presets(self):
        hw = GH200Cluster(nvl_scope=2)
        components = {component.name: component for component in hw.nodes}

        for index in range(2):
            ssd = components[f"n0-ssd-{index}"]
            cpu = components[f"n0-nvidia-grace-{index}"]
            fabric = hw.find_fabric(ssd, cpu)
            assert ssd.capacity_gb == 3840.0
            assert fabric.src_to_dst_bandwidth_gbs == 14.0
            assert fabric.dst_to_src_bandwidth_gbs == 7.0


class TestGH200SuperChip:
    def test_aggregates_requested_nvl_scope(self):
        hw = GH200SuperChip(nvl_scope=4)
        components = {component.name: component for component in hw.nodes}

        assert hw.nvl_scope == 4
        assert components["n0-nvidia-gh200-0"].tflops == {
            "fp4": 7916.0, "fp8": 7916.0,
            "bf16": 3958.0, "fp16": 3958.0, "fp32": 1978.0,
        }
        assert components["n0-nvidia-grace-0"].tflops == {"fp64": 14.2}
        assert components["n0-hbm3e-0"].capacity_gb == 576.0
        assert components["n0-dram-0"].capacity_gb == 1920.0
        assert components["n0-ssd"].capacity_gb == 15360.0
        assert not any("nvswitch" in component.name
                       for component in hw.nodes)

    def test_aggregates_bandwidths(self):
        hw = GH200SuperChip(nvl_scope=4)
        components = {component.name: component for component in hw.nodes}
        gpu = components["n0-nvidia-gh200-0"]
        cpu = components["n0-nvidia-grace-0"]
        nic = components["n0-mellanox-cx7-0"]

        hbm = hw.find_fabric(gpu, components["n0-hbm3e-0"])
        c2c = hw.find_fabric(cpu, gpu)
        nic_pcie = hw.find_fabric(cpu, nic)
        infiniband = hw.find_fabric(nic, components["ib-switch"])
        dram = hw.find_fabric(cpu, components["n0-dram-0"])
        ssd = hw.find_fabric(components["n0-ssd"], cpu)

        assert hbm.src_to_dst_bandwidth_gbs == 19600.0
        assert c2c.src_to_dst_bandwidth_gbs == 1800.0
        assert nic_pcie.src_to_dst_bandwidth_gbs == 256.0
        assert infiniband.src_to_dst_bandwidth_gbs == 200.0
        assert dram.src_to_dst_bandwidth_gbs == 2048.0
        assert ssd.src_to_dst_bandwidth_gbs == 56.0
        assert ssd.dst_to_src_bandwidth_gbs == 28.0


@pytest.mark.parametrize(
    "preset", [Ascend950DTCluster, Ascend950DTSuperChip])
@pytest.mark.parametrize("ub_scope", [0, 7, 9, 65])
def test_ascend950dt_rejects_invalid_ub_scope(preset, ub_scope):
    with pytest.raises(ValueError, match="ub_scope"):
        preset(ub_scope=ub_scope)


@pytest.mark.parametrize(
    "preset", [Ascend950DTCluster, Ascend950DTSuperChip])
@pytest.mark.parametrize("ub_scope", [8, 64])
def test_ascend950dt_accepts_ub_scope_boundaries(preset, ub_scope):
    assert preset(ub_scope=ub_scope).ub_scope == ub_scope


class TestAscend950DTCluster:
    def test_ub_scope_controls_component_counts(self):
        hw = Ascend950DTCluster(ub_scope=8)

        assert hw.ub_scope == 8
        assert sum(component.kind == "gpu" for component in hw.nodes) == 8
        assert sum(component.kind == "cpu" for component in hw.nodes) == 2
        assert sum(component.kind == "hbm" for component in hw.nodes) == 8
        assert sum(component.kind == "dram" for component in hw.nodes) == 2
        assert sum(component.kind == "ssd" for component in hw.nodes) == 8

    def test_npu_specs_and_links(self):
        hw = Ascend950DTCluster(ub_scope=8)
        components = {component.name: component for component in hw.nodes}
        npu = components["n0-huawei-ascend-950dt-0"]
        cpu = components["n0-intel-xeon-6767p-0"]

        assert npu.tflops == {
            "fp4": 1946.0, "fp8": 973.0,
            "bf16": 486.0, "fp16": 486.0, "fp32": 243.0,
        }
        assert components["n0-hbm-0"].capacity_gb == 144.0

        hbm = hw.find_fabric(npu, components["n0-hbm-0"])
        ub = hw.find_fabric(npu, components["n0-ub-switch"])
        pcie = hw.find_fabric(cpu, npu)
        uboe = hw.find_fabric(npu, components["eth-switch"])

        assert hbm.src_to_dst_bandwidth_gbs == 4000.0
        assert ub.src_to_dst_bandwidth_gbs == 896.0
        assert ub.dst_to_src_bandwidth_gbs == 896.0
        assert pcie.src_to_dst_bandwidth_gbs == 64.0
        assert pcie.dst_to_src_bandwidth_gbs == 64.0
        assert uboe.src_to_dst_bandwidth_gbs == 100.0
        assert uboe.dst_to_src_bandwidth_gbs == 100.0

    def test_reuses_b300_cpu_dram_and_ssd_specs(self):
        hw = Ascend950DTCluster(ub_scope=8)
        components = {component.name: component for component in hw.nodes}

        for cpu_index in range(2):
            cpu = components[f"n0-intel-xeon-6767p-{cpu_index}"]
            dram = components[f"n0-ddr5-{cpu_index}"]
            fabric = hw.find_fabric(cpu, dram)
            assert cpu.tflops == {
                "bf16": 255.29, "fp16": 255.29, "int8": 511.18,
            }
            assert dram.capacity_gb == 1536.0
            assert fabric.src_to_dst_bandwidth_gbs == 332.8

        qpi = hw.find_fabric(
            components["n0-intel-xeon-6767p-0"],
            components["n0-intel-xeon-6767p-1"])
        assert qpi.src_to_dst_bandwidth_gbs == 192.0

        for index in range(8):
            ssd = components[f"n0-ssd-{index}"]
            cpu = components[f"n0-intel-xeon-6767p-{index // 4}"]
            fabric = hw.find_fabric(ssd, cpu)
            assert ssd.capacity_gb == 3840.0
            assert fabric.src_to_dst_bandwidth_gbs == 14.0
            assert fabric.dst_to_src_bandwidth_gbs == 7.0

    def test_aggregate_bandwidth(self):
        hw = Ascend950DTCluster(ub_scope=8, n_nodes=2)
        npus = [component for component in hw.nodes
                if component.kind == "gpu"]
        node0_npus = [npu for npu in npus if npu.name.startswith("n0-")]

        assert hw.find_aggregate_bandwidth(node0_npus) == 896.0
        assert hw.find_aggregate_bandwidth(npus) == 800.0


class TestAscend950DTSuperChip:
    def test_aggregates_requested_ub_scope(self):
        hw = Ascend950DTSuperChip(ub_scope=8)
        components = {component.name: component for component in hw.nodes}

        assert hw.ub_scope == 8
        assert components["n0-huawei-ascend-950dt-0"].tflops == {
            "fp4": 15568.0, "fp8": 7784.0,
            "bf16": 3888.0, "fp16": 3888.0, "fp32": 1944.0,
        }
        assert components["n0-intel-xeon-6767p-0"].tflops == {
            "bf16": 510.58, "fp16": 510.58, "int8": 1022.36,
        }
        assert components["n0-hbm-0"].capacity_gb == 1152.0
        assert components["n0-ddr5-0"].capacity_gb == 3072.0
        assert components["n0-ssd"].capacity_gb == 30720.0
        assert not any("ub-switch" in component.name
                       for component in hw.nodes)

    def test_aggregates_bandwidths(self):
        hw = Ascend950DTSuperChip(ub_scope=8)
        components = {component.name: component for component in hw.nodes}
        npu = components["n0-huawei-ascend-950dt-0"]
        cpu = components["n0-intel-xeon-6767p-0"]

        hbm = hw.find_fabric(npu, components["n0-hbm-0"])
        pcie = hw.find_fabric(cpu, npu)
        dram = hw.find_fabric(cpu, components["n0-ddr5-0"])
        uboe = hw.find_fabric(npu, components["eth-switch"])
        ssd = hw.find_fabric(components["n0-ssd"], cpu)

        assert hbm.src_to_dst_bandwidth_gbs == 32000.0
        assert pcie.src_to_dst_bandwidth_gbs == 512.0
        assert dram.src_to_dst_bandwidth_gbs == 665.6
        assert uboe.src_to_dst_bandwidth_gbs == 800.0
        assert ssd.src_to_dst_bandwidth_gbs == 112.0
        assert ssd.dst_to_src_bandwidth_gbs == 56.0
