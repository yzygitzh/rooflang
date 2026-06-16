"""Per-primitive sharding descriptors and placement metadata.

Design:
  There is no single global "parallel plan". Each primitive (or group of
  fused primitives) carries its own ShardingSpec describing how it is
  distributed. A post-sweep over the primitive graph resolves data
  dependencies between adjacent primitives and patches communication
  overhead where sharding boundaries mismatch.

Sharding dimensions (per-primitive):
  - TP (tensor parallel): shards hidden-D (weights column/row-split).
    Dense and MoE layers can have different TP degrees.
  - DP (data parallel): replicates weights, shards batch. Dense and MoE
    can have different DP degrees. For decode, dp_attn is merged here —
    attention uses a potentially larger DP degree (replicated KV-cache).
  - CP (context/sequence parallel): shards sequence dimension on attention;
    dense parameters are duplicated by CP degree.
  - EP (expert parallel): MoE-only. Shards experts across EP ranks.

PP (pipeline parallel) is not part of ShardingSpec — it doesn't shard any
single primitive. PP determines which layers live on which stage and adds
send/recv transfer overhead at stage boundaries, handled externally by the
enumerator.

Communication arises at sharding boundaries:
  - TP boundary (RowParallelLinear output): all-reduce over TP group.
  - EP boundary (MoE dispatch/combine): all-to-all over EP group.
  - CP boundary (ring-attention): send/recv of K/V tiles over CP group.
  - DP boundary (gradient sync in training): all-reduce or reduce-scatter
    over DP group, depending on ZeRO stage.
  - PP boundary: point-to-point send/recv of activations between stages.

The post-sweep (not in this module — lives in the enumerator's arch.py)
walks the primitive list, detects where adjacent primitives have mismatched
sharding, and inserts communication Kernel instances (using
hardware.collective_time for timing).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ShardingSpec:
    """Per-primitive sharding descriptor.

    Each primitive carries one of these describing how its computation and
    data are distributed. The enumerator tags primitives at construction
    time; the post-sweep uses these to resolve comm.

    Fields:
      tp: int — tensor-parallel degree for this primitive's weights/activations.
      dp: int — data-parallel degree (batch sharding).
      cp: int — context-parallel degree (sequence sharding on attention).
                For non-attention primitives, cp=1 (they see the full
                sequence but their weights are duplicated cp times).
      ep: int — expert-parallel degree (MoE primitives only; 1 for dense).
      zero: int — ZeRO stage (0/1/2/3) governing gradient comm on this primitive.
    """
    tp: int = 1
    dp: int = 1
    cp: int = 1
    ep: int = 1
    zero: int = 0

    def __post_init__(self):
        assert self.tp >= 1
        assert self.dp >= 1
        assert self.cp >= 1
        assert self.ep >= 1
        assert self.zero in (0, 1, 2, 3)
