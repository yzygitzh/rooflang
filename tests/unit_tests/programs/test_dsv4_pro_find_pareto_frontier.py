"""Tests for the DSV4 Pro Pareto-frontier search driver."""

from types import SimpleNamespace

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.forward import Nop, Sampling
from rooflang.programs.dsv4_pro.find_pareto_frontier import (
    Case,
    ParallelConfig,
    _gpu_timing_metrics,
    _output_path_kernels,
    _parser,
    batch_quantum,
    build_hardware,
    enumerate_cases,
    enumerate_parallel_configs,
    pareto_frontier,
    write_outputs,
)


def test_default_worker_count_is_eight():
    assert _parser().parse_args([]).workers == 8


def test_parallel_configs_obey_optimizer_equalities():
    configs = enumerate_parallel_configs(48, 8192)

    assert configs
    assert all(config.cp * config.dp == config.ep for config in configs)
    assert all(config.ep * config.pp == 48 for config in configs)
    assert all(sum(config.pp_partition) == 61 for config in configs)
    assert all(max(config.pp_partition) - min(config.pp_partition) <= 1
               for config in configs)


def test_batch_quantum_matches_prefill_and_decode_routing_constraints():
    config = ParallelConfig(cp=4, dp=16, ep=64, pp_partition=(61,))

    assert batch_quantum("prefill", 8192, config) == 16
    assert batch_quantum("decode", 8192, config) == 4096


def test_case_ids_include_the_full_search_configuration():
    case = Case(
        workload="prefill-8k",
        hardware="gb300",
        n_gpus=64,
        batch_size=16,
        cp=4,
        dp=16,
        ep=64,
        pp_partition=(31, 30),
    )

    assert case.case_id == (
        "prefill-8k:gb300:g64:b16:cp4:dp16:ep64:pp31-30")


def test_case_enumeration_applies_batch_multipliers_and_limit():
    cases = list(enumerate_cases(
        workloads=["decode-8k"],
        hardware_names=["gb300"],
        gpu_counts=[64],
        batch_multipliers=[1, 2, 4],
        pp_degrees=[1],
        max_batch_size=8192,
    ))

    assert {case.batch_size for case in cases} == {4096, 8192}
    assert all(case.pp_partition == (61,) for case in cases)


def test_single_node_scopes_and_eight_gpu_nodes():
    for name in ("gh200", "gb300", "ascend950dt"):
        hardware = build_hardware(name, 48)
        gpus = [component for component in hardware.nodes
                if isinstance(component, Compute)
                and component.kind == "gpu"]
        assert len(gpus) == 48
        assert {gpu.name.split("-", 1)[0] for gpu in gpus} == {"n0"}

    for name in ("h200", "b300"):
        hardware = build_hardware(name, 48)
        gpus = [component for component in hardware.nodes
                if isinstance(component, Compute)
                and component.kind == "gpu"]
        assert len(gpus) == 48
        assert len({gpu.name.split("-", 1)[0] for gpu in gpus}) == 6


def test_pareto_frontier_removes_dominated_and_duplicate_points():
    records = [
        {"case_id": "a", "status": "ok",
         "tokens_per_s_user": 10.0, "tokens_per_s_gpu": 10.0},
        {"case_id": "b", "status": "ok",
         "tokens_per_s_user": 8.0, "tokens_per_s_gpu": 12.0},
        {"case_id": "c", "status": "ok",
         "tokens_per_s_user": 7.0, "tokens_per_s_gpu": 9.0},
        {"case_id": "d", "status": "ok",
         "tokens_per_s_user": 10.0, "tokens_per_s_gpu": 10.0},
        {"case_id": "oom", "status": "oom"},
    ]

    assert [point["case_id"] for point in pareto_frontier(records)] == [
        "b", "a",
    ]


def test_gpu_activity_includes_bubbles_until_each_gpus_final_kernel():
    gpu0 = Compute(name="gpu0", kind="gpu")
    gpu1 = Compute(name="gpu1", kind="gpu")
    cpu = Compute(name="cpu0", kind="cpu")
    compute0, compute1 = Nop(), Nop()
    communication0, communication1 = Nop(), Nop()
    barrier = Nop()
    result = SimpleNamespace(
        measurement_start_us=100.0,
        measured_time_us=200.0,
        trace=[
            SimpleNamespace(kernel=compute0, device=gpu0,
                            start_us=50.0, end_us=150.0,
                            compute_time_us=100.0, memory_time_us=80.0,
                            network_time_us=0.0),
            SimpleNamespace(kernel=compute1, device=gpu1,
                            start_us=120.0, end_us=160.0,
                            compute_time_us=40.0, memory_time_us=20.0,
                            network_time_us=0.0),
            SimpleNamespace(kernel=communication0, device=gpu0,
                            start_us=150.0, end_us=180.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=30.0),
            SimpleNamespace(kernel=compute0, device=gpu0,
                            start_us=180.0, end_us=200.0,
                            compute_time_us=10.0, memory_time_us=20.0,
                            network_time_us=15.0),
            SimpleNamespace(kernel=communication1, device=gpu1,
                            start_us=180.0, end_us=220.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=40.0),
            SimpleNamespace(kernel=barrier, device=gpu0,
                            start_us=250.0, end_us=300.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=0.0),
            SimpleNamespace(kernel=barrier, device=cpu,
                            start_us=100.0, end_us=300.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=0.0),
        ],
    )

    metrics = _gpu_timing_metrics(
        result, total_tokens=100, n_gpus=2,
        included_kernels={
            compute0, compute1, communication0, communication1,
        },
        duration_us=120.0)

    assert metrics["total_gpu_elapsed_ms"] == 0.2
    assert metrics["tokens_per_s_gpu_elapsed"] == 100 / 0.0002
    assert metrics["compute_ratio"] == 110 / 200
    assert metrics["communication_ratio"] == 70 / 200
    assert metrics["tokens_per_s_gpu_overlapped"] == 100 / 0.00013
    assert metrics["gpu_completion_fraction"] == 200 / (2 * 120)


def test_output_path_excludes_kv_persistence_barrier():
    graph = ComputeGraph()
    producer, sampling, barrier = Nop(), Sampling(1, 1), Nop()
    for kernel in (producer, sampling, barrier):
        graph.add_kernel(kernel)
    graph.add_control_edge(producer, sampling)
    graph.add_control_edge(sampling, barrier)

    outputs, output_path = _output_path_kernels(graph)

    assert outputs == {sampling}
    assert output_path == {producer, sampling}


def test_write_outputs_honors_filtered_workloads(tmp_path):
    record = {
        "case_id": "one",
        "status": "ok",
        "workload": "decode-8k",
        "hardware": "gb300",
        "n_gpus": 8,
        "pp_partition": [61],
        "tokens_per_s_user": 10.0,
        "tokens_per_s_gpu": 20.0,
    }

    write_outputs(tmp_path, [record])

    assert (tmp_path / "all_points.csv").is_file()
    assert (tmp_path / "pareto_frontier.csv").is_file()
    assert (tmp_path / "pareto_decode-8k.png").is_file()
    assert not (tmp_path / "pareto_prefill-8k.png").exists()
