"""Hardware spec and collective-time model for model-roofline.

Defines HardwareSpec (peak TFLOPS per dtype, HBM bandwidth, intra/inter-
node link bandwidth + latency) and a collective_time() function that
estimates wall-clock time for NCCL-style collectives under a tree-based
α + β·n model with pipelining.

Usage:
    hw = hardware_spec("b300")
    t  = collective_time("all_reduce", bytes_per_rank=16384, world=8,
                         kind="intra", hw=hw)
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class LinkSpec:
    """One side (intra-node or inter-node) of the interconnect."""
    bw_gbs: float       # per-GPU unidirectional bandwidth (aggregated), GB/s
    alpha_us: float     # per-message latency, microseconds


@dataclass
class InterNodeSpec:
    """Inter-node interconnect specification.

    Non-all-to-all collectives (AR, AG, RS, broadcast) aggregate bandwidth
    across all NICs in the node cooperatively → collective_bw_gbs.
    All-to-all traffic is point-to-point per GPU, using one NIC standalone
    → p2p_bw_gbs.
    """
    collective_bw_gbs: float    # per-GPU BW for non-A2A collectives
    p2p_bw_gbs: float           # per-GPU BW for A2A (single NIC)
    alpha_us: float             # per-message latency, microseconds


@dataclass
class HardwareSpec:
    """Per-GPU hardware specification for roofline + comm modelling."""
    peak_tflops: Dict[str, float]   # dtype → peak dense TFLOPS
    peak_bw_gbs: float              # HBM bandwidth, GB/s
    intra_node: LinkSpec            # NVLink / xGMI / etc.
    inter_node: InterNodeSpec       # InfiniBand / RoCE / etc.


# -- Built-in presets ---------------------------------------------------------

# HGX / DGX B300 per-GPU dense peaks (no 2:4 sparsity).
# Source: NVIDIA DGX B300 datasheet (8-GPU system values ÷ 8):
#   FP4 Tensor Core : 108 PFLOPS dense / 8 = 13.5 PFLOPS per GPU
#   FP8/FP6         :  36 PFLOPS dense / 8 =  4.5 PFLOPS per GPU
#   BF16/FP16       :  18 PFLOPS dense / 8 =  2.25 PFLOPS per GPU
#   FP32 TC         : derived as BF16/2 (Blackwell ratio, verify before citing)
# HBM3e: "62 TB/s" system / 8 = 7.75 TB/s per GPU.
# NVLink5: 1.8 TB/s bidirectional → 900 GB/s unidirectional.
# Inter-node: 8×800G NDR InfiniBand per node (1 NIC per GPU).
#   Per-NIC BW: 800 Gb/s = 100 GB/s unidirectional.
#   Non-A2A collectives: all 8 NICs cooperate → per-GPU = 100 GB/s.
#   A2A: each GPU uses 1 NIC standalone → per-GPU = 100 GB/s.
# Latency: NVLink ~1 µs (on-package), IB ~2 µs (switch hop).
HW_B300 = HardwareSpec(
    peak_tflops={
        "fp4":  13500.0,
        "fp8":   4500.0,
        "bf16":  2250.0,
        "fp16":  2250.0,
        "fp32":  1125.0,
    },
    peak_bw_gbs=7750.0,
    intra_node=LinkSpec(bw_gbs=900.0, alpha_us=1.0),
    inter_node=InterNodeSpec(
        collective_bw_gbs=100.0,    # 8×800G / 8 GPUs = 100 GB/s per GPU
        p2p_bw_gbs=100.0,          # 1 NIC per GPU = 100 GB/s
        alpha_us=2.0,
    ),
)

_HW_CATALOG: Dict[str, HardwareSpec] = {"b300": HW_B300}


def hardware_spec(name: str) -> HardwareSpec:
    """Return a HardwareSpec for a pre-defined preset name (e.g. "b300")."""
    if name not in _HW_CATALOG:
        raise ValueError(
            f"unknown hardware preset: {name!r}. "
            f"Available: {sorted(_HW_CATALOG.keys())}")
    return _HW_CATALOG[name]


# -- Collective bytes + time model --------------------------------------------
#
# Model: tree-based (recursive-halving / recursive-doubling) with pipelining.
#
# Latency (α) term:
#   The number of sequential message-initiation rounds is determined by the
#   tree depth = log2(W). Each round incurs one α startup.
#     - all_gather, reduce_scatter, broadcast: log2(W) rounds → log2(W)·α.
#     - all_reduce = pipelined RS + AG: 2·log2(W) rounds → 2·log2(W)·α.
#     - all_to_all: all sends are independent (full-bisection fabric) → 1·α.
#
# Bandwidth (β) term:
#   Total effective wire bytes per rank (from collective_bytes) divided by
#   link bandwidth. Pipelining within a phase means the β term is NOT
#   multiplied by the number of rounds — once the pipeline fills, one
#   chunk worth of data arrives per β·chunk_size interval. The aggregate
#   over all rounds equals collective_bytes(op, n, W) bytes.
#
# Combined: time = α_factor(op, W) · α + collective_bytes(op, n, W) · β


def collective_bytes(op: str, bytes_per_rank: float, world: int) -> float:
    """Effective wire bytes moved per rank for a collective.

    This is the bandwidth-cost portion (total data crossing the link from
    this rank's perspective), not the payload:
      - all_reduce (RS + AG): 2·(W-1)/W · n
      - all_gather:             (W-1)/W · n
      - reduce_scatter:         (W-1)/W · n
      - all_to_all:             (W-1)/W · n
      - broadcast:              n   (root sends full payload through tree)
    """
    if world <= 1:
        return 0.0
    W = world
    n = bytes_per_rank
    if op == "all_reduce":
        return 2.0 * (W - 1) / W * n
    elif op in ("all_gather", "reduce_scatter", "all_to_all"):
        return (W - 1) / W * n
    elif op == "broadcast":
        return n
    else:
        raise ValueError(f"unknown collective op: {op}")


def collective_time(op: str, bytes_per_rank: float, world: int,
                    kind: str, hw: HardwareSpec) -> float:
    """Estimate wall-clock time (seconds) for a NCCL-style collective.

    Model:
      time = α + collective_bytes(op, n, W) / effective_bw

    α is a single message-initiation latency (full-bisection fabric — all
    sends in parallel in one round).

    effective_bw depends on `kind` and `op`:
      - "intra": bw = intra_node.bw_gbs (NVLink).
      - "inter", non-A2A (AR, AG, RS, broadcast):
            bw = min(intra_node.bw_gbs, inter_node.collective_bw_gbs).
            All NICs cooperate on the collective; per-GPU BW is the
            aggregate node BW / gpus_per_node, bottlenecked against NVLink.
      - "inter", all_to_all:
            bw = min(intra_node.bw_gbs, inter_node.p2p_bw_gbs).
            Each GPU uses one NIC standalone; per-GPU BW is the single-NIC
            rate, bottlenecked against NVLink.

    For inter-node ops, α comes from inter_node (the dominant latency is
    the cross-node hop).

    Args:
      op   : "all_reduce" | "all_gather" | "reduce_scatter" |
             "all_to_all" | "broadcast"
      bytes_per_rank: payload bytes each rank contributes
      world: number of ranks in the group
      kind : "intra" | "inter" — selects latency source and BW bottleneck
      hw   : HardwareSpec instance

    Returns seconds. Returns 0 when world <= 1 (no comm needed).
    """
    if world <= 1:
        return 0.0

    if kind == "intra":
        alpha = hw.intra_node.alpha_us * 1e-6
        effective_bw = hw.intra_node.bw_gbs * 1e9
    else:
        alpha = hw.inter_node.alpha_us * 1e-6
        if op == "all_to_all":
            inter_bw = hw.inter_node.p2p_bw_gbs
        else:
            inter_bw = hw.inter_node.collective_bw_gbs
        effective_bw = min(hw.intra_node.bw_gbs, inter_bw) * 1e9

    wire_bytes = collective_bytes(op, bytes_per_rank, world)
    return alpha + wire_bytes / effective_bw
