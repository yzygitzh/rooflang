"""Search and plot DSV4 Pro throughput/latency Pareto frontiers.

The two maximized Pareto metrics are:

* tokens/s/GPU: aggregate token throughput divided by the physical GPU count;
* tokens/s/user: one user's token throughput (S / TTFT for prefill and
  1 / TPOT for one-step decode).

Each successful point also records pipeline-steady-state tokens/s/user and
tokens/s/GPU variants.  A GPU's active span runs from its first measured
kernel through its final kernel, excluding leading and trailing idle time.
Each pipeline stage runs from its earliest GPU start through its latest GPU
completion, and the pipeline cycle is set by the slowest stage.  Compute time
is each kernel's
elapsed local compute/memory path after resource contention; communication
time is only the elapsed network path exposed beyond that local path.  The
ratios use the full GPU pool over one slowest-stage cycle, so PP imbalance is
reported as bubble.  Ideal-overlap variants additionally hide, independently
on each GPU, the shorter of total compute and exposed communication time
before selecting the slowest stage.

The simulator executes and accounts for one batch.  Memory feasibility is
then evaluated separately for each frontier: original uses one KV copy,
decode elapsed uses ``PP`` concurrent copies, and decode overlapped uses
``2 * PP`` copies. Prefill does not retain KV state for pipeline concurrency,
so its elapsed and overlapped projections use one and two copies respectively.
Only the tagged KV-cache footprint is replicated; weights and activations
keep their simulated peak usage.

Each simulation is an independent process.  Results are appended to JSONL as
they finish, so an interrupted search can resume without repeating completed
cases.  Cases with the same workload, hardware, GPU count, and parallel
configuration form a sweep: batches run in ascending order, and the first
baseline OOM skips all larger batches in that sweep.  Independent sweeps
remain parallel up to the requested worker limit.

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
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
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
from rooflang.programs.presets.rtx6000d import RTX6000DCluster
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
HARDWARE_NAMES = (
    "h200", "gh200", "b300", "gb300", "ascend950dt", "rtx6000d",
)
GPU_COUNTS = (8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512)
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


SweepKey = tuple[str, str, int, int, int, int, tuple[int, ...]]


def _sweep_key(case: Case) -> SweepKey:
    return (
        case.workload,
        case.hardware,
        case.n_gpus,
        case.cp,
        case.dp,
        case.ep,
        case.pp_partition,
    )


def _record_sweep_key(record: dict) -> SweepKey:
    return (
        record["workload"],
        record["hardware"],
        record["n_gpus"],
        record["cp"],
        record["dp"],
        record["ep"],
        tuple(record["pp_partition"]),
    )


def _next_pending_batch(
    case: Case,
    completed_ids: set[str],
    oom_cutoff: int | None,
) -> Case | None:
    while oom_cutoff is None or case.batch_size < oom_cutoff:
        if case.case_id not in completed_ids:
            return case
        case = replace(case, batch_size=case.batch_size * 2)
    return None


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
    if maximum < 1:
        return []
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
                if batch_multipliers is not None:
                    multipliers = batch_multipliers
                elif max_batch_size is not None:
                    multipliers = _default_batch_multipliers(
                        max_batch_size // quantum)
                else:
                    multipliers = (1,)
                batches = sorted({
                    quantum * multiplier
                    for multiplier in multipliers
                    if max_batch_size is None
                    or quantum * multiplier <= max_batch_size
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
    if name == "rtx6000d":
        if n_gpus % 8:
            raise ValueError("RTX 6000D GPU count must be divisible by 8")
        return RTX6000DCluster(eth_scope=n_gpus)
    raise ValueError(f"Unknown hardware: {name}")


def _ordered_gpus(hardware) -> list[Compute]:
    """Return GPUs in the node/rank order used by model optimizers."""
    return sorted(
        (
            device for device in hardware.nodes
            if isinstance(device, Compute) and device.kind == "gpu"
        ),
        key=lambda device: (
            int(device.name.split("-", 1)[0][1:]),
            int(device.name.rsplit("-", 1)[1]),
        ),
    )


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


def _concurrent_kv_batches(stage: str, pp_degree: int) -> int:
    """Return the number of concurrent persistent KV cache copies."""
    return 1 if stage == "prefill" else pp_degree


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
    stage_devices: Sequence[Sequence[Compute]] | None = None,
) -> tuple[float, float, float, float, float]:
    """Return bottleneck-stage and roofline-decomposed GPU-times.

    The simulator records the elapsed local compute/memory path and network
    path separately after resource contention.  Their overlap belongs to the
    local path; only the exposed network tail belongs to communication.  Time
    not covered by either path remains a dependency/scheduling bubble.

    Stage duration covers its earliest GPU start through its latest GPU
    completion.  The returned elapsed and overlapped times are the
    slowest-stage duration multiplied by every physical GPU, so throughput
    includes both rank skew and idle capacity in faster PP stages.
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

    elapsed_by_gpu = {}
    overlapped_by_gpu = {}
    active_elapsed_time_us = 0.0
    compute_time_us = 0.0
    communication_time_us = 0.0
    for device, final_end_us in final_end_by_gpu.items():
        elapsed = final_end_us - first_start_by_gpu[device]
        compute = compute_time_by_gpu[device]
        communication = communication_time_by_gpu[device]
        elapsed_by_gpu[device] = elapsed
        overlapped_by_gpu[device] = elapsed - min(compute, communication)
        active_elapsed_time_us += elapsed
        compute_time_us += compute
        communication_time_us += communication

    if stage_devices is None:
        stage_devices = (tuple(final_end_by_gpu),)

    def stage_span_us(devices, end_by_gpu):
        active_devices = [
            device for device in devices if device in end_by_gpu
        ]
        if not active_devices:
            return 0.0
        return (
            max(end_by_gpu[device] for device in active_devices)
            - min(first_start_by_gpu[device] for device in active_devices)
        )

    slowest_stage_time_us = max(
        stage_span_us(devices, final_end_by_gpu)
        for devices in stage_devices
    )
    overlapped_end_by_gpu = {
        device: first_start_by_gpu[device] + overlapped
        for device, overlapped in overlapped_by_gpu.items()
    }
    slowest_overlapped_stage_time_us = max(
        stage_span_us(devices, overlapped_end_by_gpu)
        for devices in stage_devices
    )
    return (
        slowest_stage_time_us * n_gpus,
        compute_time_us,
        communication_time_us,
        slowest_overlapped_stage_time_us * n_gpus,
        active_elapsed_time_us,
    )


def _gpu_timing_metrics(
    result, total_tokens: int, tokens_per_user: int, n_gpus: int,
    included_kernels: set, duration_us: float,
    stage_devices: Sequence[Sequence[Compute]] | None = None,
) -> dict:
    """Calculate slowest-stage and ideal-overlap throughput metrics."""
    elapsed_time_us, compute_time_us, communication_time_us, \
        overlapped_time_us, active_elapsed_time_us = _gpu_timing_totals_us(
        result, n_gpus, included_kernels, stage_devices)
    pp_degree = len(stage_devices) if stage_devices is not None else 1
    user_elapsed_time_us = elapsed_time_us / n_gpus * pp_degree
    user_overlapped_time_us = overlapped_time_us / n_gpus * pp_degree
    return {
        "tokens_per_s_user_elapsed": (
            tokens_per_user / (user_elapsed_time_us / 1e6)),
        "tokens_per_s_user_overlapped": (
            tokens_per_user / (user_overlapped_time_us / 1e6)),
        "tokens_per_s_gpu_elapsed": total_tokens / (elapsed_time_us / 1e6),
        "tokens_per_s_gpu_overlapped": (
            total_tokens / (overlapped_time_us / 1e6)),
        "compute_ratio": compute_time_us / elapsed_time_us,
        "communication_ratio": communication_time_us / elapsed_time_us,
        "gpu_completion_fraction": (
            active_elapsed_time_us / (n_gpus * duration_us)),
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
        concurrent_batches = _concurrent_kv_batches(
            stage, pp_degree)
        projected_memory = _project_memory(result, concurrent_batches)
        record.update({
            "concurrent_batches_elapsed": concurrent_batches,
            "concurrent_batches_overlapped": concurrent_batches,
            "kv_cache_hbm_gb": _max_memory_gb(kv_cache_memory, "hbm"),
            "kv_cache_ssd_gb": _max_memory_gb(kv_cache_memory, "ssd"),
            "memory_feasible_elapsed": _memory_feasible(projected_memory),
            "memory_feasible_overlapped": _memory_feasible(
                projected_memory),
            "peak_hbm_gb_elapsed": _max_memory_gb(
                projected_memory, "hbm"),
            "peak_dram_gb_elapsed": _max_memory_gb(
                projected_memory, "dram"),
            "peak_ssd_gb_elapsed": _max_memory_gb(
                projected_memory, "ssd"),
            "peak_hbm_gb_overlapped": _max_memory_gb(
                projected_memory, "hbm"),
            "peak_dram_gb_overlapped": _max_memory_gb(
                projected_memory, "dram"),
            "peak_ssd_gb_overlapped": _max_memory_gb(
                projected_memory, "ssd"),
        })
        gpu_devices = _ordered_gpus(hardware)
        stage_devices = tuple(
            tuple(gpu_devices[stage * case.ep:(stage + 1) * case.ep])
            for stage in range(pp_degree)
        )
        record.update(_gpu_timing_metrics(
            result, total_tokens, tokens_per_user, case.n_gpus, output_path,
            duration_us, stage_devices))
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
    """Write all points, the final frontier CSV, and frontier plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "all_points.csv", records)
    frontiers = _combined_plot_frontiers(records)
    frontier_records = [
        point for group in frontiers.values() for point in group
    ]
    _write_csv(output_dir / "pareto_frontier.csv", frontier_records)
    for timing in ("elapsed", "overlapped"):
        (output_dir / f"pareto_frontier_{timing}.csv").unlink(
            missing_ok=True)

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
            axis.set_xlabel("tokens/s/user")
            axis.set_ylabel("tokens/s/GPU")
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
            f"DSV4 Pro Pareto Frontier: {workload}")
        figure.tight_layout(rect=(0, 0.04, 1, 0.96))
        figure.savefig(
            output_dir / f"pareto_{workload}.png", dpi=160)
        plt.close(figure)


def _pending_sweeps(
    cases: Iterable[Case],
    completed_ids: set[str],
    baseline_oom_cutoffs: dict[SweepKey, int] | None = None,
    grow_batches: bool = False,
) -> deque[deque[Case]]:
    """Group pending cases into batch-ordered, baseline-OOM-capped sweeps."""
    baseline_oom_cutoffs = baseline_oom_cutoffs or {}
    grouped_cases: dict[SweepKey, list[Case]] = {}
    for case in cases:
        grouped_cases.setdefault(_sweep_key(case), []).append(case)

    sweeps = deque()
    for key, sweep_cases in grouped_cases.items():
        sweep_cases.sort(key=lambda case: case.batch_size)
        oom_cutoff = baseline_oom_cutoffs.get(key)
        if grow_batches:
            candidate = _next_pending_batch(
                sweep_cases[0], completed_ids, oom_cutoff)
            pending = deque([candidate] if candidate is not None else [])
        else:
            pending = deque(
                case for case in sweep_cases
                if case.case_id not in completed_ids
                and (oom_cutoff is None or case.batch_size < oom_cutoff)
            )
        if pending:
            sweeps.append(pending)
    return sweeps


def _run_parallel(
    cases: Iterable[Case],
    workers: int,
    raw_path: Path,
    completed_ids: set[str],
    baseline_oom_cutoffs: dict[SweepKey, int] | None = None,
    grow_batches: bool = False,
) -> list[dict]:
    baseline_oom_cutoffs = baseline_oom_cutoffs or {}
    sweeps = _pending_sweeps(
        cases, completed_ids, baseline_oom_cutoffs, grow_batches)

    new_records = []
    submitted = completed = 0
    context = multiprocessing.get_context("spawn")
    with raw_path.open("a", encoding="utf-8") as output, ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=_select_model, initargs=(_MODEL_NAME,),
    ) as executor:
        futures = {}

        def submit_next(sweep) -> None:
            nonlocal submitted
            case = sweep.popleft()
            futures[executor.submit(run_case, case)] = (case, sweep)
            submitted += 1

        for _ in range(min(workers, len(sweeps))):
            submit_next(sweeps.popleft())
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                case, sweep = futures.pop(future)
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
                if record.get("status") == "oom":
                    sweep.clear()
                elif grow_batches and not sweep:
                    oom_cutoff = baseline_oom_cutoffs.get(_sweep_key(case))
                    candidate = _next_pending_batch(
                        replace(case, batch_size=case.batch_size * 2),
                        completed_ids,
                        oom_cutoff,
                    )
                    if candidate is not None:
                        sweep.append(candidate)
                if sweep:
                    submit_next(sweep)
                elif sweeps:
                    submit_next(sweeps.popleft())
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
             "batch quantum; by default doubles until baseline OOM",
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
    grow_batches = (
        args.batch_multipliers is None and args.max_batch_size is None)
    if args.dry_run:
        if grow_batches:
            print(
                f"Generated {len(cases)} batch sweeps; "
                "each doubles until baseline OOM"
            )
        else:
            print(f"Generated {len(cases)} cases")
        return 0

    latest_records = {record["case_id"]: record for record in existing}
    completed_ids = {
        case_id for case_id, record in latest_records.items()
        if not args.rerun_failures or record.get("status") == "ok"
    }
    cases_by_id = {case.case_id: case for case in cases}
    baseline_oom_cutoffs = {}
    if not args.rerun_failures:
        for case_id, record in latest_records.items():
            if record.get("status") != "oom":
                continue
            try:
                key = _record_sweep_key(record)
                batch_size = record["batch_size"]
            except KeyError:
                case = cases_by_id.get(case_id)
                if case is None:
                    continue
                key = _sweep_key(case)
                batch_size = case.batch_size
            baseline_oom_cutoffs[key] = min(
                batch_size,
                baseline_oom_cutoffs.get(key, batch_size),
            )
    remaining = sum(
        len(sweep) for sweep in
        _pending_sweeps(
            cases, completed_ids, baseline_oom_cutoffs, grow_batches)
    )
    if grow_batches:
        progress = (
            f"Generated {len(cases)} batch sweeps; {remaining} ready")
    else:
        progress = f"Generated {len(cases)} cases; {remaining} remaining"
    print(f"{progress}; workers={args.workers}", flush=True)
    new_records = _run_parallel(
        cases, args.workers, raw_path, completed_ids,
        baseline_oom_cutoffs, grow_batches)
    records = existing + new_records
    if not args.no_plot:
        write_outputs(
            args.output_dir, records, point_labels=args.point_labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
