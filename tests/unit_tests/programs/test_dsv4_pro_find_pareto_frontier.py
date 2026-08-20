"""Tests for the DSV4 Pro Pareto-frontier search driver."""

import csv
import json
from types import SimpleNamespace

import pytest

from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.forward import Nop, Sampling
from rooflang.language.tensor import Tensor
from rooflang.programs.experiments import find_pareto_frontier as finder
from rooflang.programs.experiments.find_pareto_frontier import (
    Case,
    ParallelConfig,
    _gpu_timing_metrics,
    _memory_feasible,
    _output_path_kernels,
    _peak_memory_gb,
    _point_label,
    _project_memory,
    _read_jsonl,
    _run_parallel,
    _parser,
    batch_quantum,
    build_hardware,
    enumerate_cases,
    enumerate_parallel_configs,
    grouped_frontiers,
    pareto_frontier,
    run_case,
    write_outputs,
)
from rooflang.runtime.simulator import OOMError


@pytest.fixture(autouse=True)
def select_dsv4_pro_model():
    finder._select_model("dsv4_pro")


def test_default_worker_count_is_eight():
    args = _parser().parse_args([])

    assert args.model == "dsv4_pro"
    assert args.workers == 8
    assert args.gpu_counts == [8, 16, 32, 64, 128, 256, 512]
    assert args.axis_scale == "linear"
    assert _parser().parse_args(["--axis-scale", "log"]).axis_scale == "log"
    assert not args.point_labels
    assert _parser().parse_args(["--point-labels"]).point_labels


def test_workloads_include_intermediate_context_lengths():
    assert finder.WORKLOADS == {
        "prefill-8k": ("prefill", 8192),
        "decode-8k": ("decode", 8192),
        "prefill-64k": ("prefill", 65536),
        "decode-64k": ("decode", 65536),
        "prefill-256k": ("prefill", 262144),
        "decode-256k": ("decode", 262144),
        "prefill-1m": ("prefill", 1048576),
        "decode-1m": ("decode", 1048576),
    }


def test_parallel_configs_obey_optimizer_equalities():
    configs = enumerate_parallel_configs(48, 8192)

    assert configs
    assert all(config.cp * config.dp == config.ep for config in configs)
    assert all(config.ep * config.pp == 48 for config in configs)
    assert all(sum(config.pp_partition) == 61 for config in configs)
    assert all(max(config.pp_partition) - min(config.pp_partition) <= 1
               for config in configs)


def test_batch_quantum_matches_prefill_and_decode_split_constraints():
    config = ParallelConfig(cp=4, dp=16, ep=64, pp_partition=(61,))

    assert batch_quantum("prefill", 8192, config) == 16
    assert batch_quantum("decode", 8192, config) == 64


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


def test_point_label_contains_only_ep_degree():
    point = {"batch_size": 16, "cp": 4, "dp": 2, "ep": 8, "pp": 2}

    assert _point_label(point) == "EP=8"


def test_case_enumeration_applies_batch_multipliers_and_limit():
    cases = list(enumerate_cases(
        workloads=["decode-8k"],
        hardware_names=["gb300"],
        gpu_counts=[64],
        batch_multipliers=[1, 2, 4],
        pp_degrees=[1],
        max_batch_size=8192,
    ))

    assert {case.batch_size for case in cases} == {64, 128, 256}
    assert all(case.pp_partition == (61,) for case in cases)


def test_default_batch_range_contains_only_each_sweeps_minimum():
    cases = list(enumerate_cases(
        workloads=["decode-8k"],
        hardware_names=["gb300"],
        gpu_counts=[8],
        batch_multipliers=None,
    ))

    assert cases
    for case in cases:
        config = ParallelConfig(
            case.cp, case.dp, case.ep, case.pp_partition)
        assert case.batch_size == batch_quantum("decode", 8192, config)


def test_explicit_max_batch_size_bounds_default_finite_sweep():
    cases = list(enumerate_cases(
        workloads=["decode-8k"],
        hardware_names=["gb300"],
        gpu_counts=[48],
        batch_multipliers=None,
        pp_degrees=[3],
        max_batch_size=12000,
    ))

    batches = {case.batch_size for case in cases}
    assert max(batches) == 12000
    assert 4096 in batches


def test_parallel_config_enumeration_rejects_expert_and_compression_cases(
        monkeypatch):
    monkeypatch.setattr(finder, "N_EXPERTS", 10)
    assert enumerate_parallel_configs(8, 8192)

    monkeypatch.setattr(finder, "N_EXPERTS", 384)
    monkeypatch.setattr(finder, "COMPRESS_RATIOS", (3,))
    assert enumerate_parallel_configs(8, 8192) == []


def test_single_node_scopes_and_eight_gpu_nodes():
    for name in ("gh200", "gb300", "ascend950dt", "rtx6000d"):
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


@pytest.mark.parametrize(
    ("name", "n_gpus", "expected_nodes"),
    [
        ("gb300", 128, 2),
        ("gh200", 512, 2),
        ("ascend950dt", 128, 2),
        ("rtx6000d", 512, 1),
    ],
)
def test_large_fabric_hardware_uses_multiple_maximal_nodes(
        name, n_gpus, expected_nodes):
    hardware = build_hardware(name, n_gpus)
    gpus = [component for component in hardware.nodes
            if isinstance(component, Compute)
            and component.kind == "gpu"]

    assert len(gpus) == n_gpus
    assert len({gpu.name.split("-", 1)[0] for gpu in gpus}) == expected_nodes


@pytest.mark.parametrize("name", ["h200", "b300", "rtx6000d"])
def test_eight_gpu_node_hardware_rejects_partial_nodes(name):
    with pytest.raises(ValueError, match="divisible by 8"):
        build_hardware(name, 7)


def test_build_hardware_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown hardware"):
        build_hardware("unknown", 8)


def test_ordered_gpus_ignores_hardware_node_set_order():
    gpus = [
        Compute(name=f"n{node}-gpu-{rank}", kind="gpu")
        for node in range(2)
        for rank in range(8)
    ]
    hardware = SimpleNamespace(nodes=frozenset(reversed(gpus)))

    ordered = finder._ordered_gpus(hardware)

    assert [device.name for device in ordered] == [
        f"n{node}-gpu-{rank}"
        for node in range(2)
        for rank in range(8)
    ]


def test_peak_memory_filters_component_kind():
    hbm = Memory("hbm", kind="hbm")
    dram = Memory("dram", kind="dram")
    result = SimpleNamespace(peak_memory={hbm: 2e9, dram: 3e9, object(): 9e9})

    assert _peak_memory_gb(result, "hbm") == 2.0
    assert _peak_memory_gb(result, "ssd") == 0.0


def test_project_memory_replicates_only_kv_cache():
    hbm = Memory("hbm", capacity_gb=10.0, kind="hbm")
    result = SimpleNamespace(
        peak_memory={hbm: 6e9},
        memory_footprints=[
            SimpleNamespace(
                memory=hbm, size_bytes=2e9, role="kv_cache"),
            SimpleNamespace(
                memory=hbm, size_bytes=1e9, role="other"),
        ],
    )

    projected = _project_memory(result, concurrent_batches=3)

    assert projected[hbm] == 10e9
    assert _memory_feasible(projected)
    assert not _memory_feasible(_project_memory(result, 4))


@pytest.mark.parametrize(
    ("stage", "pp_degree", "expected"),
    [
        ("prefill", 4, 1),
        ("decode", 4, 4),
    ],
)
def test_concurrent_kv_batches_excludes_prefill_pipeline_copies(
        stage, pp_degree, expected):
    assert finder._concurrent_kv_batches(stage, pp_degree) == expected


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


def test_pareto_frontier_uses_selected_timing_metrics():
    records = [
        {"case_id": "original", "status": "ok",
         "tokens_per_s_user": 12.0, "tokens_per_s_gpu": 12.0,
         "tokens_per_s_user_elapsed": 8.0,
         "tokens_per_s_gpu_elapsed": 8.0},
        {"case_id": "elapsed", "status": "ok",
         "tokens_per_s_user": 8.0, "tokens_per_s_gpu": 8.0,
         "tokens_per_s_user_elapsed": 12.0,
         "tokens_per_s_gpu_elapsed": 12.0},
    ]

    original = pareto_frontier(records)
    elapsed = pareto_frontier(
        records, "tokens_per_s_user_elapsed", "tokens_per_s_gpu_elapsed")

    assert [point["case_id"] for point in original] == ["original"]
    assert [point["case_id"] for point in elapsed] == ["elapsed"]


def test_pareto_frontier_uses_timing_specific_memory_feasibility():
    records = [
        {"case_id": "infeasible", "status": "ok",
         "tokens_per_s_user_elapsed": 12.0,
         "tokens_per_s_gpu_elapsed": 12.0,
         "memory_feasible_elapsed": False},
        {"case_id": "feasible", "status": "ok",
         "tokens_per_s_user_elapsed": 8.0,
         "tokens_per_s_gpu_elapsed": 8.0,
         "memory_feasible_elapsed": True},
    ]

    frontier = pareto_frontier(
        records,
        "tokens_per_s_user_elapsed",
        "tokens_per_s_gpu_elapsed",
        "memory_feasible_elapsed",
    )

    assert [point["case_id"] for point in frontier] == ["feasible"]


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
                            network_time_us=0.0,
                            local_elapsed_time_us=100.0,
                            network_elapsed_time_us=0.0),
            SimpleNamespace(kernel=compute1, device=gpu1,
                            start_us=120.0, end_us=160.0,
                            compute_time_us=40.0, memory_time_us=20.0,
                            network_time_us=0.0,
                            local_elapsed_time_us=40.0,
                            network_elapsed_time_us=0.0),
            SimpleNamespace(kernel=communication0, device=gpu0,
                            start_us=150.0, end_us=180.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=30.0,
                            local_elapsed_time_us=0.0,
                            network_elapsed_time_us=30.0),
            SimpleNamespace(kernel=compute0, device=gpu0,
                            start_us=180.0, end_us=200.0,
                            compute_time_us=10.0, memory_time_us=20.0,
                            network_time_us=15.0,
                            local_elapsed_time_us=20.0,
                            network_elapsed_time_us=15.0),
            SimpleNamespace(kernel=communication1, device=gpu1,
                            start_us=180.0, end_us=220.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=40.0,
                            local_elapsed_time_us=0.0,
                            network_elapsed_time_us=40.0),
            SimpleNamespace(kernel=barrier, device=gpu0,
                            start_us=250.0, end_us=300.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=0.0,
                            local_elapsed_time_us=0.0,
                            network_elapsed_time_us=0.0),
            SimpleNamespace(kernel=barrier, device=cpu,
                            start_us=100.0, end_us=300.0,
                            compute_time_us=0.0, memory_time_us=0.0,
                            network_time_us=0.0,
                            local_elapsed_time_us=0.0,
                            network_elapsed_time_us=0.0),
        ],
    )

    metrics = _gpu_timing_metrics(
        result, total_tokens=100, tokens_per_user=10, n_gpus=2,
        included_kernels={
            compute0, compute1, communication0, communication1,
        },
        duration_us=120.0)

    assert metrics["total_gpu_elapsed_ms"] == 0.24
    assert metrics["tokens_per_s_user_elapsed"] == 10 / 0.00012
    assert metrics["tokens_per_s_user_overlapped"] == 10 / 0.00008
    assert metrics["tokens_per_s_gpu_elapsed"] == 100 / 0.00024
    assert metrics["compute_ratio"] == 110 / 240
    assert metrics["communication_ratio"] == 70 / 240
    assert metrics["tokens_per_s_gpu_overlapped"] == 100 / 0.00016
    assert metrics["gpu_completion_fraction"] == 200 / (2 * 120)


@pytest.mark.parametrize(
    ("local_elapsed_us", "network_elapsed_us", "compute_ratio",
     "communication_ratio"),
    [
        (0.0, 100.0, 0.0, 1.0),
        (100.0, 20.0, 1.0, 0.0),
    ],
)
def test_gpu_activity_attributes_contention_to_its_resource(
    local_elapsed_us, network_elapsed_us, compute_ratio,
    communication_ratio,
):
    gpu = Compute(name="gpu0", kind="gpu")
    communication = Nop()
    result = SimpleNamespace(
        measurement_start_us=0.0,
        trace=[
            SimpleNamespace(
                kernel=communication, device=gpu,
                start_us=-20.0, end_us=80.0,
                compute_time_us=0.0, memory_time_us=0.0,
                network_time_us=0.0,
                local_elapsed_time_us=local_elapsed_us,
                network_elapsed_time_us=network_elapsed_us),
        ],
    )

    metrics = _gpu_timing_metrics(
        result, total_tokens=1, tokens_per_user=1, n_gpus=1,
        included_kernels={communication}, duration_us=80.0)

    assert metrics["compute_ratio"] == compute_ratio
    assert metrics["communication_ratio"] == communication_ratio
    assert metrics["tokens_per_s_gpu_overlapped"] == 1 / 0.00008


def test_gpu_activity_uses_slowest_pipeline_stage():
    gpus = [Compute(name=f"gpu{i}", kind="gpu") for i in range(6)]
    kernels = [Nop() for _ in gpus]
    durations = [200.0, 200.0, 200.0, 100.0, 100.0, 100.0]
    start = 0.0
    trace = []
    for gpu, kernel, duration in zip(gpus, kernels, durations):
        trace.append(SimpleNamespace(
            kernel=kernel, device=gpu, start_us=start,
            end_us=start + duration,
            compute_time_us=duration, memory_time_us=0.0,
            network_time_us=0.0, local_elapsed_time_us=duration,
            network_elapsed_time_us=0.0,
        ))
        start += duration
    result = SimpleNamespace(measurement_start_us=0.0, trace=trace)

    metrics = _gpu_timing_metrics(
        result, total_tokens=120, tokens_per_user=1, n_gpus=6,
        included_kernels=set(kernels), duration_us=start,
        stage_devices=tuple((gpu,) for gpu in gpus),
    )

    assert metrics["total_gpu_elapsed_ms"] == 1.2
    assert metrics["tokens_per_s_user_elapsed"] == 1 / 0.0012
    assert metrics["tokens_per_s_user_overlapped"] == 1 / 0.0012
    assert metrics["tokens_per_s_gpu_elapsed"] == 120 / 0.0012
    assert metrics["tokens_per_s_gpu_overlapped"] == 120 / 0.0012
    assert metrics["compute_ratio"] == 0.75
    assert metrics["communication_ratio"] == 0.0
    assert metrics["gpu_completion_fraction"] == 1 / 6


def test_output_path_excludes_kv_persistence_barrier():
    graph = ComputeGraph()
    producer, sampling, barrier = Nop(), Sampling(1, 1), Nop()
    barrier.inputs = {"stage_output": Tensor("int32", (1,))}
    barrier.outputs = {"done": Tensor("int32", (1,))}
    for kernel in (producer, sampling, barrier):
        graph.add_kernel(kernel)
    graph.add_control_edge(producer, sampling)
    graph.add_control_edge(sampling, barrier)

    outputs, output_path = _output_path_kernels(graph)

    assert outputs == {sampling}
    assert output_path == {producer, sampling}


def test_output_path_includes_terminal_kv_sink():
    graph = ComputeGraph()
    token_producer = Nop()
    sampling = Sampling(1, 1)
    kv_producer = Nop(outputs={"kv": Tensor("bf16", (1, 1, 64))})
    kv_sink = Nop(inputs={"kv": Tensor("bf16", (1, 1, 64))})
    for kernel in (token_producer, sampling, kv_producer, kv_sink):
        graph.add_kernel(kernel)
    graph.add_control_edge(token_producer, sampling)
    graph.add_data_edge(kv_producer, kv_sink, {"kv": "kv"})

    outputs, output_path = _output_path_kernels(graph)

    assert outputs == {sampling, kv_sink}
    assert output_path == {
        token_producer, sampling, kv_producer, kv_sink,
    }


def _case(stage="prefill"):
    return Case(
        workload=f"{stage}-8k", hardware="b300", n_gpus=8,
        batch_size=8, cp=1, dp=8, ep=8, pp_partition=(61,),
    )


def test_run_case_success_for_prefill_and_decode(monkeypatch):
    sampling = Sampling(1, 1)
    graph = SimpleNamespace(kernels={sampling})
    declared = (graph, ["layer"], "emb", "read", ["kv"], "head")
    hardware = SimpleNamespace(nodes=[
        Compute(name=f"n0-gpu-{i}", kind="gpu") for i in range(8)
    ])
    monkeypatch.setattr(finder, "build_hardware", lambda *args: hardware)
    monkeypatch.setattr(finder, "declare_model", lambda **kwargs: declared)
    optimizer_calls = []
    monkeypatch.setattr(
        finder, "optimize_model_cluster_prefill",
        lambda *args, **kwargs: optimizer_calls.append(("prefill", kwargs))
        or (graph, "placement"),
    )
    monkeypatch.setattr(
        finder, "optimize_model_cluster_decode",
        lambda *args, **kwargs: optimizer_calls.append(("decode", kwargs))
        or (graph, "placement"),
    )
    result = SimpleNamespace(
        trace=[SimpleNamespace(kernel=sampling, end_us=110.0)],
        measurement_start_us=10.0,
        total_time_us=120.0,
        peak_memory={},
        memory_footprints=(),
    )

    class FakeSimulator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def run(self):
            return result

    monkeypatch.setattr(finder, "Simulator", FakeSimulator)
    monkeypatch.setattr(
        finder, "_output_path_kernels",
        lambda _: ({sampling}, {sampling}),
    )
    monkeypatch.setattr(
        finder, "_gpu_timing_metrics",
        lambda *args: {"compute_ratio": 0.5},
    )
    times = iter((1.0, 2.0, 3.0, 5.0))
    monkeypatch.setattr(finder.time, "perf_counter", lambda: next(times))

    prefill = run_case(_case("prefill"))
    decode = run_case(_case("decode"))

    assert prefill["status"] == decode["status"] == "ok"
    assert prefill["tokens_per_s_user"] == 8192 / 0.0001
    assert decode["tokens_per_s_user"] == 1 / 0.0001
    assert prefill["post_output_ms"] == 0.01
    assert prefill["compute_ratio"] == 0.5
    assert prefill["concurrent_batches_elapsed"] == 1
    assert prefill["concurrent_batches_overlapped"] == 1
    assert prefill["memory_feasible_elapsed"]
    assert prefill["memory_feasible_overlapped"]
    assert [name for name, _ in optimizer_calls] == ["prefill", "decode"]
    assert "seq_prefill" in optimizer_calls[1][1]


def test_run_case_reports_oom(monkeypatch):
    memory = Memory("hbm0", capacity_gb=1.0, kind="hbm")
    error = OOMError(memory, 2e9, 1e9, [], Nop())
    monkeypatch.setattr(
        finder, "build_hardware",
        lambda *args: (_ for _ in ()).throw(error),
    )
    times = iter((1.0, 2.0))
    monkeypatch.setattr(finder.time, "perf_counter", lambda: next(times))

    record = run_case(_case())

    assert record["status"] == "oom"
    assert record["oom_memory"] == "hbm0"
    assert record["oom_memory_kind"] == "hbm"
    assert record["oom_used_gb"] == 2.0
    assert record["oom_capacity_gb"] == 1.0


def test_run_case_keeps_sweep_running_after_error(monkeypatch):
    monkeypatch.setattr(
        finder, "build_hardware",
        lambda *args: (_ for _ in ()).throw(RuntimeError("broken")),
    )
    times = iter((1.0, 2.0))
    monkeypatch.setattr(finder.time, "perf_counter", lambda: next(times))

    record = run_case(_case())

    assert record["status"] == "error"
    assert record["error"] == "RuntimeError: broken"


def test_grouped_frontiers_skips_failures():
    records = [
        {"case_id": "ok", "status": "ok", "workload": "prefill-8k",
         "hardware": "b300", "n_gpus": 8,
         "tokens_per_s_user": 1.0, "tokens_per_s_gpu": 1.0},
        {"case_id": "bad", "status": "oom"},
    ]

    assert list(grouped_frontiers(records)) == [("prefill-8k", "b300", 8)]


def test_combined_plot_frontiers_merge_feasible_timing_projections():
    record = {
        "case_id": "one", "status": "ok", "workload": "decode-8k",
        "hardware": "b300", "n_gpus": 8,
        "tokens_per_s_user": 10.0, "tokens_per_s_gpu": 10.0,
        "tokens_per_s_user_elapsed": 8.0,
        "tokens_per_s_gpu_elapsed": 20.0,
        "tokens_per_s_user_overlapped": 20.0,
        "tokens_per_s_gpu_overlapped": 8.0,
        "memory_feasible_elapsed": True,
        "memory_feasible_overlapped": True,
    }

    frontier = finder._combined_plot_frontiers([record])[
        ("decode-8k", "b300", 8)]

    assert [point["_plot_timing"] for point in frontier] == [
        "elapsed", "original", "overlapped"]
    assert [point[finder._PLOT_USER_METRIC] for point in frontier] == [
        8.0, 10.0, 20.0]


def test_combined_plot_frontiers_exclude_infeasible_projection():
    record = {
        "case_id": "one", "status": "ok", "workload": "decode-8k",
        "hardware": "b300", "n_gpus": 8,
        "tokens_per_s_user": 10.0, "tokens_per_s_gpu": 10.0,
        "tokens_per_s_user_elapsed": 8.0,
        "tokens_per_s_gpu_elapsed": 20.0,
        "tokens_per_s_user_overlapped": 20.0,
        "tokens_per_s_gpu_overlapped": 8.0,
        "memory_feasible_elapsed": False,
        "memory_feasible_overlapped": True,
    }

    frontier = finder._combined_plot_frontiers([record])[
        ("decode-8k", "b300", 8)]

    assert [point["_plot_timing"] for point in frontier] == [
        "original", "overlapped"]


def test_read_jsonl_handles_missing_blank_valid_and_invalid(tmp_path):
    path = tmp_path / "records.jsonl"
    assert _read_jsonl(path) == []
    path.write_text('\n{"case_id": "one"}\n', encoding="utf-8")
    assert _read_jsonl(path) == [{"case_id": "one"}]
    path.write_text('{"case_id":\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"records.jsonl:1"):
        _read_jsonl(path)


def test_write_outputs_handles_no_records(tmp_path):
    (tmp_path / "pareto_frontier_elapsed.csv").write_text("stale")
    (tmp_path / "pareto_frontier_overlapped.csv").write_text("stale")

    write_outputs(tmp_path, [], "dsv4_pro")

    assert (tmp_path / "all_points.csv").read_text() == ""
    assert (tmp_path / "pareto_frontier.csv").read_text() == ""
    assert not (tmp_path / "pareto_frontier_elapsed.csv").exists()
    assert not (tmp_path / "pareto_frontier_overlapped.csv").exists()
    assert not list(tmp_path.glob("*.png"))


def test_shared_axis_limits_cover_all_subplots_with_same_scale():
    frontiers = {
        ("decode-8k", "h200", 8): [
            {"tokens_per_s_user": 10.0, "tokens_per_s_gpu": 20.0},
        ],
        ("decode-8k", "gb300", 16): [
            {"tokens_per_s_user": 100.0, "tokens_per_s_gpu": 200.0},
        ],
        ("prefill-8k", "gb300", 16): [
            {"tokens_per_s_user": 1000.0, "tokens_per_s_gpu": 2000.0},
        ],
    }

    x_limits, y_limits = finder._shared_axis_limits(
        frontiers, "decode-8k", [8, 16], ["h200", "gb300"],
        "tokens_per_s_user", "tokens_per_s_gpu",
    )
    assert x_limits == pytest.approx((0.0, 105.0))
    assert y_limits == pytest.approx((0.0, 210.0))


def test_shared_axis_limits_start_at_zero_for_empty_and_single_point_axes():
    assert finder._shared_axis_limits(
        {}, "decode-8k", [8], ["h200"],
        "tokens_per_s_user", "tokens_per_s_gpu",
    ) == ((0.0, 1.0), (0.0, 1.0))

    frontiers = {
        ("decode-8k", "h200", 8): [
            {"tokens_per_s_user": 16.0, "tokens_per_s_gpu": 32.0},
        ],
    }
    assert finder._shared_axis_limits(
        frontiers, "decode-8k", [8], ["h200"],
        "tokens_per_s_user", "tokens_per_s_gpu",
    ) == ((0.0, 16.8), (0.0, 33.6))


def test_shared_axis_limits_restore_positive_padding_for_log_axes():
    frontiers = {
        ("decode-8k", "h200", 8): [
            {"tokens_per_s_user": 16.0, "tokens_per_s_gpu": 32.0},
        ],
    }
    assert finder._shared_axis_limits(
        frontiers, "decode-8k", [8], ["h200"],
        "tokens_per_s_user", "tokens_per_s_gpu", "log",
    ) == ((8.0, 32.0), (16.0, 64.0))
    assert finder._shared_axis_limits(
        {}, "decode-8k", [8], ["h200"],
        "tokens_per_s_user", "tokens_per_s_gpu", "log",
    ) == ((1.0, 2.0), (1.0, 2.0))


@pytest.mark.parametrize(("value", "label"), [
    (1000.0, "1K"),
    (2000.0, "2K"),
    (1000.0 * 1000, "1M"),
    (1.5 * 1000 * 1000, "1.5M"),
    (1000.0 ** 3, "1G"),
    (1.5 * 1000 ** 3, "1.5G"),
    (1000.0 ** 4, "1T"),
    (1.5 * 1000 ** 4, "1.5T"),
    (1.0, "1"),
    (0.5, "0.5"),
    (0.0, "0"),
])
def test_plain_tick_label_uses_decimal_numbers(value, label):
    assert finder._plain_tick_label(value) == label


def test_plain_tick_label_supports_binary_log_axis_units():
    assert finder._plain_tick_label(1024.0, unit_base=1024) == "1K"
    assert finder._plain_tick_label(1024.0 ** 2, unit_base=1024) == "1M"


def test_write_outputs_honors_filtered_workloads(tmp_path):
    record = {
        "case_id": "one",
        "status": "ok",
        "workload": "decode-8k",
        "hardware": "gb300",
        "n_gpus": 8,
        "batch_size": 8,
        "cp": 1,
        "dp": 8,
        "ep": 8,
        "pp": 1,
        "pp_partition": [61],
        "tokens_per_s_user": 10.0,
        "tokens_per_s_gpu": 20.0,
        "tokens_per_s_user_elapsed": 12.0,
        "tokens_per_s_gpu_elapsed": 24.0,
        "tokens_per_s_user_overlapped": 15.0,
        "tokens_per_s_gpu_overlapped": 30.0,
        "memory_feasible_elapsed": True,
        "memory_feasible_overlapped": True,
    }

    (tmp_path / "pareto_frontier_elapsed.csv").write_text("stale")
    (tmp_path / "pareto_frontier_overlapped.csv").write_text("stale")

    write_outputs(tmp_path, [record], "dsv4_pro")

    assert (tmp_path / "all_points.csv").is_file()
    assert (tmp_path / "pareto_frontier.csv").is_file()
    assert not (tmp_path / "pareto_frontier_elapsed.csv").exists()
    assert not (tmp_path / "pareto_frontier_overlapped.csv").exists()
    with (tmp_path / "pareto_frontier.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert [(row["_plot_timing"], row["_plot_tokens_per_s_user"],
             row["_plot_tokens_per_s_gpu"]) for row in rows] == [
        ("overlapped", "15.0", "30.0"),
    ]
    assert (tmp_path / "pareto_decode-8k.png").is_file()
    assert not (tmp_path / "pareto_decode-8k_elapsed.png").exists()
    assert not (tmp_path / "pareto_decode-8k_overlapped.png").exists()
    assert not (tmp_path / "pareto_prefill-8k.png").exists()


def test_write_outputs_point_labels_are_opt_in(tmp_path, monkeypatch):
    record = {
        "case_id": "one", "status": "ok", "workload": "prefill-8k",
        "hardware": "b300", "n_gpus": 8, "batch_size": 8,
        "cp": 1, "dp": 8, "ep": 8, "pp": 1, "pp_partition": [61],
        "tokens_per_s_user": 10.0, "tokens_per_s_gpu": 20.0,
        "tokens_per_s_user_elapsed": 10.0,
        "tokens_per_s_gpu_elapsed": 20.0,
        "tokens_per_s_user_overlapped": 10.0,
        "tokens_per_s_gpu_overlapped": 20.0,
        "memory_feasible_elapsed": True,
        "memory_feasible_overlapped": True,
    }
    labels = []
    monkeypatch.setattr(
        finder, "_point_label",
        lambda point: labels.append(point["case_id"]) or "label",
    )

    write_outputs(tmp_path, [record], "dsv4_pro")
    assert not labels

    write_outputs(tmp_path, [record], "dsv4_pro", point_labels=True)
    assert labels == ["one"]


def test_write_outputs_shows_plain_ticks_on_every_subplot(
        tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    records = []
    for n_gpus in (8, 16, 32):
        value = float(n_gpus * 128)
        records.append({
            "case_id": str(n_gpus), "status": "ok",
            "workload": "decode-8k", "hardware": "h200",
            "n_gpus": n_gpus, "batch_size": n_gpus,
            "cp": 1, "dp": n_gpus, "ep": n_gpus,
            "pp": 1, "pp_partition": [61],
            "tokens_per_s_user": value,
            "tokens_per_s_gpu": value * 2,
            "tokens_per_s_user_elapsed": value,
            "tokens_per_s_gpu_elapsed": value * 2,
            "tokens_per_s_user_overlapped": value,
            "tokens_per_s_gpu_overlapped": value * 2,
            "memory_feasible_elapsed": True,
            "memory_feasible_overlapped": True,
        })

    write_outputs(tmp_path, records, "dsv4_flash")

    assert len(figures) == 1
    assert figures[0][0]._suptitle.get_text() == (
        "dsv4_flash Pareto Frontier: decode-8k")
    for figure, axes in figures:
        figure.canvas.draw()
        for axis in list(axes.flat)[:3]:
            assert axis.get_xscale() == "linear"
            assert axis.get_yscale() == "linear"
            assert axis.get_xlim()[0] == 0
            assert axis.get_ylim()[0] == 0
            xlabels = axis.get_xticklabels()
            ylabels = axis.get_yticklabels()
            assert xlabels and all(label.get_visible() for label in xlabels)
            assert ylabels and all(label.get_visible() for label in ylabels)
            assert all("$" not in label.get_text()
                       for label in (*xlabels, *ylabels))

    write_outputs(tmp_path, records, "dsv4_flash", axis_scale="log")

    assert len(figures) == 2
    for axis in list(figures[-1][1].flat)[:3]:
        assert axis.get_xscale() == "log"
        assert axis.get_yscale() == "log"
        assert axis.get_xlim()[0] > 0
        assert axis.get_ylim()[0] > 0
        assert axis.xaxis.get_major_formatter()(1024, 0) == "1K"


def test_write_outputs_merges_legends_and_keeps_hardware_colors(
        tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    figures = []
    original_subplots = plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    records = []
    for n_gpus, hardware in (
        (8, "rtx6000d"),
        (16, "h200"),
        (16, "rtx6000d"),
    ):
        records.append({
            "case_id": f"{hardware}-{n_gpus}", "status": "ok",
            "workload": "prefill-8k", "hardware": hardware,
            "n_gpus": n_gpus, "batch_size": n_gpus,
            "cp": 1, "dp": n_gpus, "ep": n_gpus,
            "pp": 1, "pp_partition": [61],
            "tokens_per_s_user": float(n_gpus),
            "tokens_per_s_gpu": float(n_gpus),
            "tokens_per_s_user_elapsed": float(n_gpus),
            "tokens_per_s_gpu_elapsed": float(n_gpus),
            "tokens_per_s_user_overlapped": float(n_gpus),
            "tokens_per_s_gpu_overlapped": float(n_gpus),
            "memory_feasible_elapsed": True,
            "memory_feasible_overlapped": True,
        })

    write_outputs(tmp_path, records, "dsv4_pro")

    figure, axes = figures[0]
    assert [text.get_text() for text in figure.legends[0].get_texts()] == [
        "h200", "rtx6000d",
    ]
    colors = [
        {line.get_label(): line.get_color() for line in axis.lines}
        for axis in axes.flat
    ]
    assert colors[0]["rtx6000d"] == colors[1]["rtx6000d"]


def test_write_outputs_handles_empty_frontiers_and_extra_axes(tmp_path):
    records = [
        {"case_id": str(n_gpus), "status": "oom",
         "workload": "prefill-8k", "hardware": "b300",
         "n_gpus": n_gpus, "pp_partition": [61]}
        for n_gpus in (8, 16, 32, 48)
    ]

    write_outputs(tmp_path, records, "dsv4_pro")

    assert (tmp_path / "pareto_prefill-8k.png").is_file()


class _Future:
    def __init__(self, case):
        self.case = case

    def result(self):
        if self.case.hardware == "h200":
            raise RuntimeError("worker failed")
        return {"case_id": self.case.case_id, "status": "ok"}


class _Executor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, case):
        return _Future(case)


def test_run_parallel_skips_completed_and_records_worker_errors(
        tmp_path, monkeypatch):
    cases = [
        _case(),
        Case("prefill-8k", "gb300", 8, 8, 1, 8, 8, (61,)),
        Case("prefill-8k", "h200", 8, 8, 1, 8, 8, (61,)),
    ]
    monkeypatch.setattr(finder, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(
        finder, "wait",
        lambda futures, return_when: ({next(iter(futures))}, set()),
    )
    raw_path = tmp_path / "raw.jsonl"

    records = _run_parallel(
        cases, workers=2, raw_path=raw_path,
        completed_ids={cases[0].case_id},
    )

    assert [record["status"] for record in records] == [
        "ok", "worker_error",
    ]
    written = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert [record["case_id"] for record in written] == [
        record["case_id"] for record in records
    ]
    assert [record["status"] for record in written] == [
        "ok", "worker_error",
    ]


def test_run_parallel_orders_each_sweep_and_stops_after_oom(
        tmp_path, monkeypatch):
    cases = [
        Case("prefill-8k", hardware, 8, 8, 1, 8, 8, (61,))
        for hardware in ("b300", "gb300")
    ]
    submitted = []

    class Future:
        def __init__(self, case):
            self.case = case

        def result(self):
            status = (
                "oom" if (
                    self.case.hardware == "b300"
                    and self.case.batch_size == 16
                    or self.case.hardware == "gb300"
                    and self.case.batch_size == 32
                )
                else "ok"
            )
            return {"case_id": self.case.case_id, "status": status}

    class Executor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, case):
            submitted.append(case)
            return Future(case)

    monkeypatch.setattr(finder, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        finder, "wait",
        lambda futures, return_when: ({next(iter(futures))}, set()),
    )

    records = _run_parallel(
        reversed(cases), workers=2, raw_path=tmp_path / "raw.jsonl",
        completed_ids=set(), grow_batches=True,
    )

    b300_batches = [
        case.batch_size for case in submitted if case.hardware == "b300"
    ]
    gb300_batches = [
        case.batch_size for case in submitted if case.hardware == "gb300"
    ]
    assert {case.hardware for case in submitted[:2]} == {"b300", "gb300"}
    assert b300_batches == [8, 16]
    assert gb300_batches == [8, 16, 32]
    assert len(records) == 5


def test_dynamic_sweep_resumes_after_completed_batches():
    first = Case("prefill-8k", "b300", 8, 8, 1, 8, 8, (61,))
    second = Case("prefill-8k", "b300", 8, 16, 1, 8, 8, (61,))

    sweeps = finder._pending_sweeps(
        [first], {first.case_id, second.case_id}, grow_batches=True)

    assert len(sweeps) == 1
    assert [case.batch_size for case in sweeps[0]] == [32]


def test_run_parallel_uses_existing_oom_as_resume_cutoff(
        tmp_path, monkeypatch):
    cases = [Case("prefill-8k", "b300", 8, 8, 1, 8, 8, (61,))]
    submitted = []

    class Executor(_Executor):
        def submit(self, function, case):
            submitted.append(case)
            return _Future(case)

    monkeypatch.setattr(finder, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(
        finder, "wait",
        lambda futures, return_when: ({next(iter(futures))}, set()),
    )

    records = _run_parallel(
        cases, workers=2, raw_path=tmp_path / "raw.jsonl",
        completed_ids=set(),
        baseline_oom_cutoffs={
            ("prefill-8k", "b300", 8, 1, 8, 8, (61,)): 16,
        },
        grow_batches=True,
    )

    assert [case.batch_size for case in submitted] == [8]
    assert len(records) == 1


def test_run_parallel_does_not_stop_after_worker_error(tmp_path, monkeypatch):
    cases = [
        Case("prefill-8k", "h200", 8, batch_size, 1, 8, 8, (61,))
        for batch_size in (8, 16)
    ]
    monkeypatch.setattr(finder, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr(
        finder, "wait",
        lambda futures, return_when: ({next(iter(futures))}, set()),
    )

    records = _run_parallel(
        cases, workers=2, raw_path=tmp_path / "raw.jsonl",
        completed_ids=set(),
    )

    assert [record["status"] for record in records] == [
        "worker_error", "worker_error",
    ]


def test_main_validates_workers_and_batch_multipliers(tmp_path):
    with pytest.raises(ValueError, match="workers"):
        finder.main(["--output-dir", str(tmp_path), "--workers", "0"])
    with pytest.raises(ValueError, match="batch-multipliers"):
        finder.main([
            "--output-dir", str(tmp_path),
            "--batch-multipliers", "0",
        ])


def test_main_plot_only_requires_and_writes_existing_records(
        tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="No records"):
        finder.main(["--output-dir", str(tmp_path), "--plot-only"])

    raw_path = tmp_path / "raw_results.jsonl"
    record = {"case_id": "one", "status": "oom"}
    raw_path.write_text(json.dumps(record) + "\n")
    writes = []
    monkeypatch.setattr(
        finder, "write_outputs",
        lambda output_dir, records, model, point_labels=False,
        axis_scale="linear":
        writes.append((output_dir, records, model, point_labels, axis_scale)),
    )

    assert finder.main([
        "--output-dir", str(tmp_path), "--plot-only", "--point-labels",
        "--axis-scale", "log", "--model", "dsv4_flash",
    ]) == 0
    assert writes == [(tmp_path, [record], "dsv4_flash", True, "log")]


def test_main_dry_run_and_overwrite(tmp_path, monkeypatch, capsys):
    raw_path = tmp_path / "raw_results.jsonl"
    tmp_path.mkdir(exist_ok=True)
    raw_path.write_text("old\n")
    monkeypatch.setattr(finder, "enumerate_cases", lambda **kwargs: [_case()])

    assert finder.main([
        "--output-dir", str(tmp_path), "--overwrite", "--dry-run",
    ]) == 0
    assert not raw_path.exists()
    assert "Generated 1 batch sweeps" in capsys.readouterr().out


def test_main_resumes_and_optionally_reruns_failures(tmp_path, monkeypatch):
    cases = [
        _case(),
        Case("prefill-8k", "h200", 8, 8, 1, 8, 8, (61,)),
        Case("prefill-8k", "gb300", 8, 8, 1, 8, 8, (61,)),
    ]
    existing = [
        {"case_id": cases[0].case_id, "status": "ok"},
        {"case_id": cases[1].case_id, "status": "oom"},
    ]
    monkeypatch.setattr(finder, "_read_jsonl", lambda path: existing)
    monkeypatch.setattr(finder, "enumerate_cases", lambda **kwargs: cases)
    parallel_calls = []
    monkeypatch.setattr(
        finder, "_run_parallel",
        lambda cases, workers, raw_path, completed_ids,
        baseline_oom_cutoffs, grow_batches:
        parallel_calls.append(
            (set(completed_ids), dict(baseline_oom_cutoffs), grow_batches))
        or [
            {"case_id": "new", "status": "ok"}
        ],
    )
    writes = []
    monkeypatch.setattr(
        finder, "write_outputs",
        lambda output_dir, records, model, point_labels=False,
        axis_scale="linear":
        writes.append((list(records), model, point_labels, axis_scale)),
    )

    assert finder.main(["--output-dir", str(tmp_path)]) == 0
    assert parallel_calls[-1] == (
        {cases[0].case_id, cases[1].case_id},
        {("prefill-8k", "h200", 8, 1, 8, 8, (61,)): 8},
        True,
    )
    assert writes[-1] == (
        existing + [{"case_id": "new", "status": "ok"}], "dsv4_pro",
        False, "linear")

    assert finder.main([
        "--output-dir", str(tmp_path), "--rerun-failures", "--no-plot",
    ]) == 0
    assert parallel_calls[-1] == ({cases[0].case_id}, {}, True)
    assert len(writes) == 1
