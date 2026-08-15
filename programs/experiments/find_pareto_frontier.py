"""Search and plot DSV4 Pro throughput/latency Pareto frontiers.

The two maximized Pareto metrics are:

* tokens/s/GPU: aggregate token throughput divided by the physical GPU count;
* tokens/s/user: one user's token throughput (S / TTFT for prefill and
  1 / TPOT for one-step decode).

Each successful point also records tokens/s/user and tokens/s/GPU variants
using each GPU's elapsed time from its first measured kernel through its final
kernel. This includes internal dependency and resource-contention bubbles, but
excludes leading and trailing idle time. Compute time is each kernel's
elapsed local compute/memory path after resource contention; communication
time is only the elapsed network path exposed beyond that local path.
Ideal-overlap variants additionally assume that, on each GPU, the shorter of
total compute and exposed communication time can be completely hidden by the
longer one.

The simulator executes and accounts for one batch.  Memory feasibility is
then evaluated separately for each frontier: original uses one KV copy,
decode elapsed uses ``PP`` concurrent copies, and decode overlapped uses
``2 * PP`` copies. Prefill does not retain KV state for pipeline concurrency,
so its elapsed and overlapped projections use one and two copies respectively.
Only the tagged KV-cache footprint is replicated; weights and activations
keep their simulated peak usage.

Each simulation is an independent process.  Results are appended to JSONL as
they finish, so an interrupted search can resume without repeating completed
cases.

CSV files retain separate original, elapsed, and ideal-overlap frontiers.  A
plot combines the feasible points from all three timing/memory projections and
finds one frontier over the combined candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from rooflang.language.hardware.component import Compute, Memory
from rooflang.language.kernels.forward import Nop, Sampling
from rooflang.programs.models import MODEL_NAMES, load_model
from rooflang.programs.presets.ascend950dt import Ascend950DTCluster
from rooflang.programs.presets.b300 import B300Cluster
from rooflang.programs.presets.gb300 import GB300Cluster
from rooflang.programs.presets.gh200 import GH200Cluster
from rooflang.programs.presets.h200 import H200Cluster
from rooflang.runtime.simulator import OOMError, Simulator


_MODEL_NAME = None


def _select_model(name):
    """Load the model API used by this process."""
    global _MODEL_NAME
    global COMPRESS_RATIOS, N_EXPERTS, N_LAYERS, WINDOW
    global declare_model
    global optimize_model_cluster_decode, optimize_model_cluster_prefill

    model = load_model(name)
    _MODEL_NAME = name
    COMPRESS_RATIOS = model.COMPRESS_RATIOS
    N_EXPERTS = model.N_EXPERTS
    N_LAYERS = model.N_LAYERS
    WINDOW = model.WINDOW
    declare_model = model.declare_model
    optimize_model_cluster_decode = model.optimize_model_cluster_decode
    optimize_model_cluster_prefill = model.optimize_model_cluster_prefill

WORKLOADS = {
    "prefill-8k": ("prefill", 8192),
    "decode-8k": ("decode", 8192),
    "prefill-64k": ("prefill", 65536),
    "decode-64k": ("decode", 65536),
    "prefill-256k": ("prefill", 262144),
    "decode-256k": ("decode", 262144),
    "prefill-1m": ("prefill", 1048576),
    "decode-1m": ("decode", 1048576),
}
HARDWARE_NAMES = ("h200", "gh200", "b300", "gb300", "ascend950dt")
GPU_COUNTS = (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
MAX_BATCH_SIZE_PER_GPU = 64
THROUGHPUT_METRICS = {
    "original": ("tokens_per_s_user", "tokens_per_s_gpu"),
    "elapsed": (
        "tokens_per_s_user_elapsed", "tokens_per_s_gpu_elapsed"),
    "overlapped": (
        "tokens_per_s_user_overlapped", "tokens_per_s_gpu_overlapped"),
}
MEMORY_FEASIBILITY_FIELDS = {
    "original": None,
    "elapsed": "memory_feasible_elapsed",
    "overlapped": "memory_feasible_overlapped",
}
_PLOT_USER_METRIC = "_plot_tokens_per_s_user"
_PLOT_GPU_METRIC = "_plot_tokens_per_s_gpu"


@dataclass(frozen=True)
class ParallelConfig:
    cp: int
    dp: int
    ep: int
    pp_partition: tuple[int, ...]

    @property
    def pp(self) -> int:
        return len(self.pp_partition)


@dataclass(frozen=True)
class Case:
    workload: str
    hardware: str
    n_gpus: int
    batch_size: int
    cp: int
    dp: int
    ep: int
    pp_partition: tuple[int, ...]

    @property
    def case_id(self) -> str:
        partition = "-".join(map(str, self.pp_partition))
        return (
            f"{self.workload}:{self.hardware}:g{self.n_gpus}:"
            f"b{self.batch_size}:cp{self.cp}:dp{self.dp}:ep{self.ep}:"
            f"pp{partition}"
        )


def _divisors(value: int) -> list[int]:
    result = set()
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate == 0:
            result.add(candidate)
            result.add(value // candidate)
    return sorted(result)


def _balanced_partition(total: int, parts: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, parts)
    return tuple(
        quotient + (index < remainder)
        for index in range(parts)
    )


def enumerate_parallel_configs(
    n_gpus: int,
    seq_prefill: int,
    pp_degrees: Sequence[int] | None = None,
) -> list[ParallelConfig]:
    """Enumerate every balanced PP and legal CP×DP=EP configuration."""
    requested_pp = set(pp_degrees) if pp_degrees else None
    configs = []
    for pp in _divisors(n_gpus):
        if pp > N_LAYERS or requested_pp is not None and pp not in requested_pp:
            continue
        ep = n_gpus // pp
        if N_EXPERTS % ep != 0:
            continue
        for cp in _divisors(ep):
            if WINDOW % cp != 0 or seq_prefill % cp != 0:
                continue
            if any(
                seq_prefill % ratio != 0
                or (seq_prefill // ratio) % cp != 0
                for ratio in set(COMPRESS_RATIOS)
            ):
                continue
            configs.append(ParallelConfig(
                cp=cp,
                dp=ep // cp,
                ep=ep,
                pp_partition=_balanced_partition(N_LAYERS, pp),
            ))
    return configs


def batch_quantum(stage: str, seq_prefill: int, config: ParallelConfig) -> int:
    """Return the smallest batch satisfying optimizer divisibility checks."""
    return config.dp if stage == "prefill" else config.cp * config.dp


def _default_batch_multipliers(maximum: int) -> list[int]:
    """Double from one and include an exact, possibly non-power-of-two cap."""
    multipliers = []
    multiplier = 1
    while multiplier < maximum:
        multipliers.append(multiplier)
        multiplier *= 2
    multipliers.append(maximum)
    return multipliers


def enumerate_cases(
    workloads: Sequence[str],
    hardware_names: Sequence[str],
    gpu_counts: Sequence[int],
    batch_multipliers: Sequence[int] | None,
    pp_degrees: Sequence[int] | None = None,
    max_batch_size: int | None = None,
) -> Iterator[Case]:
    for workload in workloads:
        stage, seq_prefill = WORKLOADS[workload]
        for n_gpus in gpu_counts:
            configs = enumerate_parallel_configs(
                n_gpus, seq_prefill, pp_degrees)
            for config in configs:
                quantum = batch_quantum(stage, seq_prefill, config)
                batch_limit = n_gpus * MAX_BATCH_SIZE_PER_GPU
                if max_batch_size is not None:
                    batch_limit = min(batch_limit, max_batch_size)
                multipliers = batch_multipliers
                if multipliers is None:
                    multipliers = _default_batch_multipliers(
                        n_gpus * MAX_BATCH_SIZE_PER_GPU // quantum)
                batches = sorted({
                    quantum * multiplier
                    for multiplier in multipliers
                    if quantum * multiplier <= batch_limit
                })
                for hardware in hardware_names:
                    for batch_size in batches:
                        yield Case(
                            workload=workload,
                            hardware=hardware,
                            n_gpus=n_gpus,
                            batch_size=batch_size,
                            cp=config.cp,
                            dp=config.dp,
                            ep=config.ep,
                            pp_partition=config.pp_partition,
                        )


def _largest_node_scope(n_gpus: int, maximum: int, quantum: int = 1) -> int:
    """Return the largest legal per-node scope dividing ``n_gpus``."""
    for scope in range(min(n_gpus, maximum), quantum - 1, -1):
        if scope % quantum == 0 and n_gpus % scope == 0:
            return scope
    raise ValueError(
        f"GPU count {n_gpus} cannot be divided into scopes of {quantum}")


def build_hardware(name: str, n_gpus: int):
    """Construct the requested cluster with exactly ``n_gpus`` accelerators."""
    if name == "h200":
        if n_gpus % 8:
            raise ValueError("H200 GPU count must be divisible by 8")
        return H200Cluster(n_nodes=n_gpus // 8)
    if name == "b300":
        if n_gpus % 8:
            raise ValueError("B300 GPU count must be divisible by 8")
        return B300Cluster(n_nodes=n_gpus // 8)
    if name == "gh200":
        scope = _largest_node_scope(n_gpus, maximum=GH200Cluster.max_scope)
        return GH200Cluster(nvl_scope=scope, n_nodes=n_gpus // scope)
    if name == "gb300":
        scope = _largest_node_scope(
            n_gpus, maximum=GB300Cluster.max_scope, quantum=4)
        return GB300Cluster(nvl_scope=scope, n_nodes=n_gpus // scope)
    if name == "ascend950dt":
        scope = _largest_node_scope(
            n_gpus, maximum=Ascend950DTCluster.max_scope, quantum=8)
        return Ascend950DTCluster(ub_scope=scope, n_nodes=n_gpus // scope)
    raise ValueError(f"Unknown hardware: {name}")


def _max_memory_gb(memory_usage, kind: str) -> float:
    values = [
        value / 1e9
        for memory, value in memory_usage.items()
        if isinstance(memory, Memory) and memory.kind == kind
    ]
    return max(values, default=0.0)


def _peak_memory_gb(result, kind: str) -> float:
    return _max_memory_gb(result.peak_memory, kind)


def _kv_cache_memory(result) -> dict[Memory, float]:
    usage = {}
    for footprint in result.memory_footprints:
        if footprint.role != "kv_cache":
            continue
        usage[footprint.memory] = \
            usage.get(footprint.memory, 0.0) + footprint.size_bytes
    return usage


def _project_memory(result, concurrent_batches: int) -> dict[Memory, float]:
    """Project peaks while replicating only persistent KV cache state."""
    projected = dict(result.peak_memory)
    for memory, size_bytes in _kv_cache_memory(result).items():
        projected[memory] = projected.get(memory, 0.0) \
            + (concurrent_batches - 1) * size_bytes
    return projected


def _concurrent_kv_batches(stage: str, pp_degree: int) -> tuple[int, int]:
    """Return elapsed/overlapped KV copies for the selected stage."""
    if stage == "prefill":
        return 1, 2
    return pp_degree, pp_degree * 2


def _memory_feasible(memory_usage) -> bool:
    return all(
        size_bytes <= memory.capacity_gb * 1e9
        for memory, size_bytes in memory_usage.items()
    )


def _output_path_kernels(graph) -> tuple[set, set]:
    """Return stage-completion kernels and their dependency ancestors.

    Sampling produces the user-visible token.  A terminal, outputless Nop is
    an explicit dependency sink: DSV4 Pro uses one per layer to require that
    the newly computed decode KV has completed even though the simplified
    one-step model omits the cache-append data path.
    """
    outputs = {
        kernel for kernel in graph.kernels
        if isinstance(kernel, Sampling)
        or (isinstance(kernel, Nop) and kernel.inputs and not kernel.outputs)
    }
    ancestors = set(outputs)
    stack = list(outputs)
    while stack:
        kernel = stack.pop()
        for predecessor in graph._dag.predecessors(kernel):
            if predecessor not in ancestors:
                ancestors.add(predecessor)
                stack.append(predecessor)
    return outputs, ancestors


def _gpu_timing_totals_us(
    result, n_gpus: int, included_kernels: set,
) -> tuple[float, float, float, float]:
    """Return elapsed and roofline-decomposed GPU-times.

    The simulator records the elapsed local compute/memory path and network
    path separately after resource contention.  Their overlap belongs to the
    local path; only the exposed network tail belongs to communication.  Time
    not covered by either path remains a dependency/scheduling bubble.
    """
    measurement_start = result.measurement_start_us
    first_start_by_gpu = {}
    final_end_by_gpu = {}
    # Trace entries are appended as kernels complete, so end_us is
    # nondecreasing. In reverse, the first entry for a GPU is its final one.
    for entry in reversed(result.trace):
        if entry.end_us <= measurement_start:
            break
        if entry.kernel not in included_kernels:
            continue
        if not isinstance(entry.device, Compute) \
                or entry.device.kind != "gpu":
            continue
        final_end_by_gpu.setdefault(entry.device, entry.end_us)
        if len(final_end_by_gpu) == n_gpus:
            break

    compute_time_by_gpu = {device: 0.0 for device in final_end_by_gpu}
    communication_time_by_gpu = {
        device: 0.0 for device in final_end_by_gpu
    }
    for entry in result.trace:
        if entry.kernel not in included_kernels \
                or entry.device not in final_end_by_gpu \
                or entry.end_us <= measurement_start:
            continue
        start_us = max(entry.start_us, measurement_start)
        end_us = min(entry.end_us, final_end_by_gpu[entry.device])
        if end_us <= start_us:
            continue
        local_end_us = entry.start_us + entry.local_elapsed_time_us
        network_end_us = entry.start_us + entry.network_elapsed_time_us
        local_elapsed_us = max(0.0, min(end_us, local_end_us) - start_us)
        network_elapsed_us = max(
            0.0, min(end_us, network_end_us) - start_us)
        compute_time_by_gpu[entry.device] += local_elapsed_us
        communication_time_by_gpu[entry.device] += max(
            0.0, network_elapsed_us - local_elapsed_us)
        first_start_by_gpu[entry.device] = min(
            first_start_by_gpu.get(entry.device, start_us), start_us)

    elapsed_time_us = 0.0
    compute_time_us = 0.0
    communication_time_us = 0.0
    overlapped_time_us = 0.0
    for device, final_end_us in final_end_by_gpu.items():
        elapsed = final_end_us - first_start_by_gpu[device]
        compute = compute_time_by_gpu[device]
        communication = communication_time_by_gpu[device]
        elapsed_time_us += elapsed
        compute_time_us += compute
        communication_time_us += communication
        overlapped_time_us += elapsed - min(compute, communication)
    return (
        elapsed_time_us,
        compute_time_us,
        communication_time_us,
        overlapped_time_us,
    )


def _gpu_timing_metrics(
    result, total_tokens: int, tokens_per_user: int, n_gpus: int,
    included_kernels: set, duration_us: float,
) -> dict:
    """Calculate elapsed and ideal compute/comm-overlap throughput metrics."""
    elapsed_time_us, compute_time_us, communication_time_us, \
        overlapped_time_us = _gpu_timing_totals_us(
        result, n_gpus, included_kernels)
    return {
        "tokens_per_s_user_elapsed": (
            tokens_per_user / (elapsed_time_us / n_gpus / 1e6)),
        "tokens_per_s_user_overlapped": (
            tokens_per_user / (overlapped_time_us / n_gpus / 1e6)),
        "tokens_per_s_gpu_elapsed": total_tokens / (elapsed_time_us / 1e6),
        "tokens_per_s_gpu_overlapped": (
            total_tokens / (overlapped_time_us / 1e6)),
        "compute_ratio": compute_time_us / elapsed_time_us,
        "communication_ratio": communication_time_us / elapsed_time_us,
        "gpu_completion_fraction": (
            elapsed_time_us / (n_gpus * duration_us)),
        "total_gpu_elapsed_ms": elapsed_time_us / 1000,
    }


def run_case(case: Case) -> dict:
    """Build, optimize, and simulate one point.  Runs inside a worker."""
    started = time.perf_counter()
    stage, seq_prefill = WORKLOADS[case.workload]
    record = {
        "case_id": case.case_id,
        **asdict(case),
        "pp": len(case.pp_partition),
        "stage": stage,
        "seq_prefill": seq_prefill,
        "pid": os.getpid(),
    }
    try:
        hardware = build_hardware(case.hardware, case.n_gpus)
        graph, layers, emb, read_input, kv_reads, output_head = declare_model(
            batch_size=case.batch_size,
            seq_prefill=seq_prefill,
            decode=stage == "decode",
        )
        kwargs = dict(
            cp=case.cp,
            dp=case.dp,
            ep=case.ep,
            pp_partition=list(case.pp_partition),
            n_gpus=case.n_gpus,
        )
        if stage == "prefill":
            graph, placement = optimize_model_cluster_prefill(
                graph, layers, hardware, emb, read_input, output_head,
                **kwargs,
            )
        else:
            graph, placement = optimize_model_cluster_decode(
                graph, layers, hardware, emb, read_input, kv_reads,
                output_head, seq_prefill=seq_prefill, **kwargs,
            )

        result = Simulator(
            graph, placement, hardware, measurement_start=read_input,
        ).run()
        output_kernels, output_path = _output_path_kernels(graph)
        output_end_us = max(
            entry.end_us for entry in result.trace
            if entry.kernel in output_kernels
        )
        duration_us = output_end_us - result.measurement_start_us
        duration_s = duration_us / 1e6
        tokens_per_user = seq_prefill if stage == "prefill" else 1
        total_tokens = case.batch_size * tokens_per_user
        record.update({
            "status": "ok",
            "latency_ms": duration_us / 1000,
            "preload_ms": result.measurement_start_us / 1000,
            "post_output_ms": (result.total_time_us - output_end_us) / 1000,
            "tokens_per_s_user": tokens_per_user / duration_s,
            "tokens_per_s_gpu": (
                total_tokens / duration_s / case.n_gpus
            ),
            "peak_hbm_gb": _peak_memory_gb(result, "hbm"),
            "peak_dram_gb": _peak_memory_gb(result, "dram"),
            "peak_ssd_gb": _peak_memory_gb(result, "ssd"),
            "kernel_count": len(graph.kernels),
        })
        kv_cache_memory = _kv_cache_memory(result)
        pp_degree = len(case.pp_partition)
        elapsed_batches, overlapped_batches = _concurrent_kv_batches(
            stage, pp_degree)
        elapsed_memory = _project_memory(result, elapsed_batches)
        overlapped_memory = _project_memory(result, overlapped_batches)
        record.update({
            "concurrent_batches_elapsed": elapsed_batches,
            "concurrent_batches_overlapped": overlapped_batches,
            "kv_cache_hbm_gb": _max_memory_gb(kv_cache_memory, "hbm"),
            "kv_cache_ssd_gb": _max_memory_gb(kv_cache_memory, "ssd"),
            "memory_feasible_elapsed": _memory_feasible(elapsed_memory),
            "memory_feasible_overlapped": _memory_feasible(
                overlapped_memory),
            "peak_hbm_gb_elapsed": _max_memory_gb(
                elapsed_memory, "hbm"),
            "peak_dram_gb_elapsed": _max_memory_gb(
                elapsed_memory, "dram"),
            "peak_ssd_gb_elapsed": _max_memory_gb(
                elapsed_memory, "ssd"),
            "peak_hbm_gb_overlapped": _max_memory_gb(
                overlapped_memory, "hbm"),
            "peak_dram_gb_overlapped": _max_memory_gb(
                overlapped_memory, "dram"),
            "peak_ssd_gb_overlapped": _max_memory_gb(
                overlapped_memory, "ssd"),
        })
        record.update(_gpu_timing_metrics(
            result, total_tokens, tokens_per_user, case.n_gpus, output_path,
            duration_us))
    except OOMError as error:
        record.update({
            "status": "oom",
            "error": str(error),
            "oom_memory": error.memory.name,
            "oom_memory_kind": error.memory.kind,
            "oom_used_gb": error.used_bytes / 1e9,
            "oom_capacity_gb": error.capacity_bytes / 1e9,
        })
    except Exception as error:  # Keep a large sweep running after one failure.
        record.update({
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        })
    record["wall_time_s"] = time.perf_counter() - started
    return record


def pareto_frontier(
    records: Iterable[dict],
    user_metric: str = "tokens_per_s_user",
    gpu_metric: str = "tokens_per_s_gpu",
    memory_feasible_field: str | None = None,
) -> list[dict]:
    """Return points not dominated on both maximized throughput metrics."""
    points = [
        record for record in records
        if record.get("status") == "ok"
        and (memory_feasible_field is None
             or record.get(memory_feasible_field, False))
    ]
    points.sort(
        key=lambda record: (
            -record[user_metric],
            -record[gpu_metric],
            record["case_id"],
        )
    )
    frontier = []
    best_gpu_throughput = -math.inf
    for point in points:
        gpu_throughput = point[gpu_metric]
        if gpu_throughput > best_gpu_throughput:
            frontier.append(point)
            best_gpu_throughput = gpu_throughput
    return sorted(frontier, key=lambda record: record[user_metric])


def grouped_frontiers(
    records: Sequence[dict],
    user_metric: str = "tokens_per_s_user",
    gpu_metric: str = "tokens_per_s_gpu",
    memory_feasible_field: str | None = None,
) -> dict[tuple, list[dict]]:
    groups = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        key = (record["workload"], record["hardware"], record["n_gpus"])
        groups.setdefault(key, []).append(record)
    return {
        key: pareto_frontier(
            group, user_metric, gpu_metric, memory_feasible_field)
        for key, group in groups.items()
    }


def _combined_plot_frontiers(
    records: Sequence[dict],
) -> dict[tuple, list[dict]]:
    """Merge all feasible timing/memory projections before Pareto filtering."""
    groups = {}
    for timing, (user_metric, gpu_metric) in THROUGHPUT_METRICS.items():
        memory_feasible_field = MEMORY_FEASIBILITY_FIELDS[timing]
        for record in records:
            if record.get("status") != "ok":
                continue
            if (memory_feasible_field is not None
                    and not record.get(memory_feasible_field, False)):
                continue
            point = dict(record)
            point[_PLOT_USER_METRIC] = record[user_metric]
            point[_PLOT_GPU_METRIC] = record[gpu_metric]
            point["_plot_timing"] = timing
            key = (
                record["workload"], record["hardware"], record["n_gpus"])
            groups.setdefault(key, []).append(point)
    return {
        key: pareto_frontier(
            group, _PLOT_USER_METRIC, _PLOT_GPU_METRIC)
        for key, group in groups.items()
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {error}") from error
    return records


def _write_csv(path: Path, records: Sequence[dict]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["pp_partition"] = "-".join(map(str, row["pp_partition"]))
            writer.writerow(row)


def _point_label(point: dict) -> str:
    """Return the expert-parallel degree for a frontier point."""
    return f"EP={point['ep']}"


def _shared_axis_limits(
    frontiers: dict[tuple, list[dict]],
    workload: str,
    gpu_counts: Sequence[int],
    hardware_names: Sequence[str],
    user_metric: str,
    gpu_metric: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return common positive log-scale limits for one figure's subplots."""
    points = [
        point
        for n_gpus in gpu_counts
        for hardware in hardware_names
        for point in frontiers.get((workload, hardware, n_gpus), [])
    ]

    def padded_limits(metric: str) -> tuple[float, float]:
        values = [point[metric] for point in points if point[metric] > 0]
        if not values:
            return 1.0, 2.0
        lower, upper = min(values), max(values)
        if lower == upper:
            return lower / 2, upper * 2
        return lower / 1.05, upper * 1.05

    return padded_limits(user_metric), padded_limits(gpu_metric)


def _plain_tick_label(value: float, _position=None) -> str:
    """Format log ticks as plain numbers with binary K/M/G/T suffixes."""
    if value <= 0:
        return ""
    for scale, suffix in (
        (1024 ** 4, "T"),
        (1024 ** 3, "G"),
        (1024 ** 2, "M"),
        (1024, "K"),
    ):
        if value >= scale:
            scaled = value / scale
            number = f"{scaled:.3f}".rstrip("0").rstrip(".")
            return f"{number}{suffix}"
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_outputs(
    output_dir: Path,
    records: Sequence[dict],
    point_labels: bool = False,
) -> None:
    """Write timing-specific CSVs and combined-projection frontier plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "all_points.csv", records)
    for timing, (user_metric, gpu_metric) in THROUGHPUT_METRICS.items():
        frontiers = grouped_frontiers(
            records, user_metric, gpu_metric,
            MEMORY_FEASIBILITY_FIELDS[timing])
        frontier_records = [
            point for group in frontiers.values() for point in group
        ]
        suffix = "" if timing == "original" else f"_{timing}"
        _write_csv(
            output_dir / f"pareto_frontier{suffix}.csv",
            frontier_records,
        )

    for workload in WORKLOADS:
        for timing in ("elapsed", "overlapped"):
            (output_dir / f"pareto_{workload}_{timing}.png").unlink(
                missing_ok=True)

    if not records:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    workloads = [
        workload for workload in WORKLOADS
        if any(record.get("workload") == workload for record in records)
    ]
    gpu_counts = [
        n_gpus for n_gpus in GPU_COUNTS
        if any(record.get("n_gpus") == n_gpus for record in records)
    ]
    hardware_names = [
        hardware for hardware in HARDWARE_NAMES
        if any(record.get("hardware") == hardware for record in records)
    ]
    rows = 1 if len(gpu_counts) <= 2 else 2
    columns = math.ceil(len(gpu_counts) / rows)
    frontiers = _combined_plot_frontiers(records)
    for workload in workloads:
        x_limits, y_limits = _shared_axis_limits(
            frontiers, workload, gpu_counts, hardware_names,
            _PLOT_USER_METRIC, _PLOT_GPU_METRIC)
        figure, axes = plt.subplots(
            rows, columns, figsize=(6 * columns, 5 * rows), squeeze=False,
            sharex=True, sharey=True)
        flat_axes = list(axes.flat)
        for axis, n_gpus in zip(flat_axes, gpu_counts):
            for hardware in hardware_names:
                points = frontiers.get((workload, hardware, n_gpus), [])
                if not points:
                    continue
                user_values = [
                    point[_PLOT_USER_METRIC] for point in points]
                gpu_values = [
                    point[_PLOT_GPU_METRIC] for point in points]
                axis.plot(
                    user_values,
                    gpu_values,
                    marker="o",
                    label=hardware,
                )
                if point_labels:
                    for point, user_value, gpu_value in zip(
                            points, user_values, gpu_values):
                        axis.annotate(
                            _point_label(point),
                            (user_value, gpu_value),
                            xytext=(3, 3),
                            textcoords="offset points",
                            fontsize=6,
                            alpha=0.85,
                        )
            axis.set_title(f"{n_gpus} GPUs")
            axis.set_xscale("log", base=2)
            axis.set_yscale("log", base=2)
            axis.xaxis.set_major_formatter(FuncFormatter(_plain_tick_label))
            axis.yaxis.set_major_formatter(FuncFormatter(_plain_tick_label))
            axis.set_xlim(x_limits)
            axis.set_ylim(y_limits)
            axis.tick_params(
                axis="both", which="both", labelbottom=True, labelleft=True)
            axis.set_xlabel("tokens/s/user (combined, log2)")
            axis.set_ylabel("tokens/s/GPU (combined, log2)")
            axis.grid(True, which="both", alpha=0.25)
        for axis in flat_axes[len(gpu_counts):]:
            axis.axis("off")
        handles, labels = [], []
        for axis in flat_axes[:len(gpu_counts)]:
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                break
        if handles:
            figure.legend(handles, labels, loc="lower right")
        figure.suptitle(
            f"DSV4 Pro Pareto Frontier (combined): {workload}")
        figure.tight_layout(rect=(0, 0.04, 1, 0.96))
        figure.savefig(output_dir / f"pareto_{workload}.png", dpi=160)
        plt.close(figure)


def _run_parallel(
    cases: Iterable[Case],
    workers: int,
    raw_path: Path,
    completed_ids: set[str],
) -> list[dict]:
    pending_cases = (case for case in cases if case.case_id not in completed_ids)
    new_records = []
    submitted = completed = 0
    context = multiprocessing.get_context("spawn")
    with raw_path.open("a", encoding="utf-8") as output, ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=_select_model, initargs=(_MODEL_NAME,),
    ) as executor:
        futures = {}

        def submit_next() -> bool:
            nonlocal submitted
            try:
                case = next(pending_cases)
            except StopIteration:
                return False
            futures[executor.submit(run_case, case)] = case
            submitted += 1
            return True

        for _ in range(workers * 2):
            if not submit_next():
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                case = futures.pop(future)
                try:
                    record = future.result()
                except BaseException as error:
                    record = {
                        "case_id": case.case_id,
                        **asdict(case),
                        "status": "worker_error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                new_records.append(record)
                completed += 1
                print(
                    f"[{completed}/{submitted}] {record['status']:>12} "
                    f"{case.case_id} ({record.get('wall_time_s', 0):.1f}s)",
                    flush=True,
                )
                submit_next()
    return new_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=MODEL_NAMES, default="dsv4_pro",
        help="Model implementation (default: dsv4_pro)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("dsv4_pro_pareto"),
        help="Result directory (default: dsv4_pro_pareto)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel simulation processes (default: 8)",
    )
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS),
    )
    parser.add_argument(
        "--hardware", nargs="+", choices=HARDWARE_NAMES,
        default=list(HARDWARE_NAMES),
    )
    parser.add_argument(
        "--gpu-counts", nargs="+", type=int, choices=GPU_COUNTS,
        default=list(GPU_COUNTS),
    )
    parser.add_argument(
        "--batch-multipliers", nargs="+", type=int,
        help="Explicit multipliers applied to each configuration's legal "
             "batch quantum; by default doubles through 64 batches/GPU",
    )
    parser.add_argument(
        "--pp-degrees", nargs="+", type=int,
        help="Restrict PP degrees; default enumerates every legal divisor",
    )
    parser.add_argument(
        "--max-batch-size", type=int,
        help="Skip generated batch sizes above this value",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Discard an existing raw_results.jsonl instead of resuming",
    )
    parser.add_argument(
        "--rerun-failures", action="store_true",
        help="On resume, rerun non-successful cases",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Only regenerate CSV and plots from raw_results.jsonl",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the generated case count without running simulations",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Collect JSONL data without producing CSV and plots",
    )
    parser.add_argument(
        "--point-labels", action="store_true",
        help="Label Pareto points with batch size and parallelism",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _select_model(args.model)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.batch_multipliers is not None and any(
            multiplier <= 0 for multiplier in args.batch_multipliers):
        raise ValueError("--batch-multipliers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.jsonl"
    if args.overwrite and raw_path.exists():
        raw_path.unlink()
    existing = _read_jsonl(raw_path)

    if args.plot_only:
        if not existing:
            raise ValueError(f"No records found in {raw_path}")
        write_outputs(
            args.output_dir, existing, point_labels=args.point_labels)
        return 0

    cases = list(enumerate_cases(
        workloads=args.workloads,
        hardware_names=args.hardware,
        gpu_counts=args.gpu_counts,
        batch_multipliers=args.batch_multipliers,
        pp_degrees=args.pp_degrees,
        max_batch_size=args.max_batch_size,
    ))
    if args.dry_run:
        print(f"Generated {len(cases)} cases")
        return 0

    completed_ids = {
        record["case_id"] for record in existing
        if not args.rerun_failures or record.get("status") == "ok"
    }
    remaining = sum(case.case_id not in completed_ids for case in cases)
    print(
        f"Generated {len(cases)} cases; {remaining} remaining; "
        f"workers={args.workers}",
        flush=True,
    )
    new_records = _run_parallel(
        cases, args.workers, raw_path, completed_ids)
    records = existing + new_records
    if not args.no_plot:
        write_outputs(
            args.output_dir, records, point_labels=args.point_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
