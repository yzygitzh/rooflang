"""Communication Kernel subclasses for collective operations.

These model the cost of NCCL-style collectives at sharding boundaries.
The enumerator emits them wherever adjacent primitives have mismatched
sharding (TP boundary → AllReduce, EP boundary → AllToAll, etc.).

Reduce-included collectives (AllReduce, ReduceScatter) perform actual
arithmetic (element-wise addition across ranks), so flops > 0:
  - Per-rank reduction flops = (W-1)/W · n_elements
    (each rank reduces its local chunk across W-1 partial sums)

Transfer-only collectives (AllGather, Broadcast) have flops=0.

AllToAll is transfer-only — the permutation is a routing decision, not
an arithmetic operation.

Note: for comm kernels, transferred_bytes represents *wire* traffic,
which differs from (input_bytes + weight_bytes + output_bytes).
"""

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.utils import dtype_bytes


class AllReduce(Kernel):
    """All-reduce: every rank ends with the full reduced result.

    flops = (W-1)/W · n_elements (reduction adds).
    transferred_bytes = 2·(W-1)/W · bytes_per_rank (RS + AG wire cost).
    input_bytes = bytes_per_rank (local buffer read).
    output_bytes = bytes_per_rank (local buffer written in-place).
    """

    def __init__(self, bytes_per_rank: float, world: int,
                 dtype: str = "bf16"):
        self.bytes_per_rank = bytes_per_rank
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.bytes_per_rank / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return 2.0 * (self.world - 1) / self.world * self.bytes_per_rank

    @property
    def input_bytes(self) -> float:
        return self.bytes_per_rank

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.bytes_per_rank


class ReduceScatter(Kernel):
    """Reduce-scatter: each rank gets 1/W of the reduced result.

    flops = (W-1)/W · n_elements (reduction adds).
    transferred_bytes = (W-1)/W · bytes_per_rank.
    input_bytes = bytes_per_rank (local buffer read).
    output_bytes = bytes_per_rank / W (local shard written).
    """

    def __init__(self, bytes_per_rank: float, world: int,
                 dtype: str = "bf16"):
        self.bytes_per_rank = bytes_per_rank
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.bytes_per_rank / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.bytes_per_rank

    @property
    def input_bytes(self) -> float:
        return self.bytes_per_rank

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.bytes_per_rank / self.world


class AllGather(Kernel):
    """All-gather: each rank broadcasts its shard; all get the full tensor.

    flops = 0 (no arithmetic).
    transferred_bytes = (W-1)/W · bytes_per_rank.
    input_bytes = bytes_per_rank / W (local shard read).
    output_bytes = bytes_per_rank (full tensor written).
    """

    def __init__(self, bytes_per_rank: float, world: int):
        self.bytes_per_rank = bytes_per_rank
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.bytes_per_rank

    @property
    def input_bytes(self) -> float:
        return self.bytes_per_rank / self.world

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.bytes_per_rank


class AllToAll(Kernel):
    """All-to-all: each rank sends a distinct chunk to every other rank.

    flops = 0 (permutation routing, no arithmetic).
    transferred_bytes = (W-1)/W · bytes_per_rank.
    input_bytes = bytes_per_rank (local buffer read).
    output_bytes = bytes_per_rank (received chunks written).
    """

    def __init__(self, bytes_per_rank: float, world: int):
        self.bytes_per_rank = bytes_per_rank
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.bytes_per_rank

    @property
    def input_bytes(self) -> float:
        return self.bytes_per_rank

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.bytes_per_rank


class Broadcast(Kernel):
    """Broadcast: root sends full payload to all ranks.

    flops = 0.
    transferred_bytes = bytes_per_rank (full payload through tree).
    input_bytes = bytes_per_rank (root reads).
    output_bytes = bytes_per_rank (every rank writes the result).
    """

    def __init__(self, bytes_per_rank: float, world: int):
        self.bytes_per_rank = bytes_per_rank
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.bytes_per_rank

    @property
    def input_bytes(self) -> float:
        return self.bytes_per_rank

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.bytes_per_rank


class Send(Kernel):
    """Point-to-point send (e.g. PP stage boundary, sender side).

    flops = 0.
    transferred_bytes = bytes_total.
    input_bytes = bytes_total (read from local HBM).
    output_bytes = 0 (nothing written locally).
    """

    def __init__(self, bytes_total: float):
        self.bytes_total = bytes_total
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.bytes_total

    @property
    def input_bytes(self) -> float:
        return self.bytes_total

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 0.0


class Recv(Kernel):
    """Point-to-point recv (e.g. PP stage boundary, receiver side).

    flops = 0.
    transferred_bytes = bytes_total.
    input_bytes = 0 (nothing read locally).
    output_bytes = bytes_total (written to local HBM).
    """

    def __init__(self, bytes_total: float):
        self.bytes_total = bytes_total
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.bytes_total

    @property
    def input_bytes(self) -> float:
        return 0.0

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.bytes_total
