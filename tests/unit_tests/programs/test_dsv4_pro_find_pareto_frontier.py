"""Tests for the DSV4 Pro Pareto-frontier search driver."""

from rooflang.language.hardware.component import Compute
from rooflang.programs.dsv4_pro.find_pareto_frontier import (
    Case,
    ParallelConfig,
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
