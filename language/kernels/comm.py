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


class CommKernel(Kernel):
    """Base class for all communication kernels.

    Communication kernels do not require placement — their cost is
    attributed to adjacent compute kernels in the simulator.
    """
    _requires_placement = False


class AllReduce(CommKernel):
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


class ReduceScatter(CommKernel):
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


class AllGather(CommKernel):
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


class AllToAll(CommKernel):
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


class Broadcast(CommKernel):
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


class Scatter(CommKernel):
    """Scatter: root distributes distinct slices to each rank.

    flops = 0.
    transferred_bytes = (W-1)/W · bytes_per_rank.
    input_bytes = bytes_per_rank (root reads full tensor).
    output_bytes = bytes_per_rank / W (each rank writes its slice).
    """

    def __init__(self, bytes_per_rank: float, world: int, dim: int = 0):
        self.bytes_per_rank = bytes_per_rank
        self.world = world
        self.dim = dim
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
        return self.bytes_per_rank / self.world


class Gather(CommKernel):
    """Gather: each rank sends its slice to root which concatenates.

    flops = 0.
    transferred_bytes = (W-1)/W · bytes_per_rank.
    input_bytes = bytes_per_rank / W (each rank reads its slice).
    output_bytes = bytes_per_rank (root writes full tensor).
    """

    def __init__(self, bytes_per_rank: float, world: int, dim: int = 0):
        self.bytes_per_rank = bytes_per_rank
        self.world = world
        self.dim = dim
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


class Reduce(CommKernel):
    """Reduce: each rank sends full buffer; root reduces element-wise.

    flops = (W-1)/W · n_elements (reduction adds).
    transferred_bytes = (W-1)/W · bytes_per_rank.
    input_bytes = bytes_per_rank (each rank reads full buffer).
    output_bytes = bytes_per_rank (root writes reduced result).
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
        return self.bytes_per_rank


class Send(CommKernel):
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


class Recv(CommKernel):
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
