"""Communication Kernel subclasses for collective operations.

These model the cost of NCCL-style collectives at sharding boundaries.

All collectives use the ring algorithm model:
  transferred_bytes = coeff × (W-1)/W × total_bytes

where coeff=1 for single-phase (Scatter, Gather, Broadcast, Reduce,
ReduceScatter, AllGather) and coeff=2 for two-phase (AllReduce = RS + AG).

total_bytes is the logical tensor size involved in the collective.

Reduce-included collectives (AllReduce, ReduceScatter, Reduce) perform
actual arithmetic (element-wise addition across ranks), so flops > 0:
  flops = (W-1)/W × n_elements
"""

from rooflang.language.kernels.kernel import Kernel
from rooflang.language.utils import dtype_bytes


class CommKernel(Kernel):
    """Base class for all communication kernels.

    Communication kernels do not require placement — their cost is
    derived from neighboring placed kernels in the simulator.

    Port constraints (enforced by validate_ports):
      Scatter/Broadcast: 1 input, world outputs
      Gather/Reduce:     world inputs, 1 output
      AllReduce/AllGather/ReduceScatter/AllToAll: world inputs, world outputs
    """
    _requires_placement = False
    _single_input = False   # True → exactly 1 input port
    _single_output = False  # True → exactly 1 output port

    def validate_ports(self) -> None:
        """Check port counts match the single-tensor constraint."""
        if not hasattr(self, 'world'):
            return
        if self._single_input and len(self.inputs) != 1:
            raise ValueError(
                f"{type(self).__name__}: expected 1 input, got {len(self.inputs)}")
        if not self._single_input and len(self.inputs) != self.world:
            raise ValueError(
                f"{type(self).__name__}: expected {self.world} inputs, "
                f"got {len(self.inputs)}")
        if self._single_output and len(self.outputs) != 1:
            raise ValueError(
                f"{type(self).__name__}: expected 1 output, got {len(self.outputs)}")
        if not self._single_output and len(self.outputs) != self.world:
            raise ValueError(
                f"{type(self).__name__}: expected {self.world} outputs, "
                f"got {len(self.outputs)}")


class AllReduce(CommKernel):
    """All-reduce: every rank ends with the full reduced result.

    Two-phase (ReduceScatter + AllGather):
      transferred_bytes = 2 × (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int, dtype: str = "bf16"):
        self.total_bytes = total_bytes
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.total_bytes / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return 2.0 * (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class ReduceScatter(CommKernel):
    """Reduce-scatter: each rank gets 1/W of the reduced result.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int, dtype: str = "bf16"):
        self.total_bytes = total_bytes
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.total_bytes / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes / self.world


class AllGather(CommKernel):
    """All-gather: each rank broadcasts its shard; all get the full tensor.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int):
        self.total_bytes = total_bytes
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes / self.world

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class AllToAll(CommKernel):
    """All-to-all: each rank sends a distinct chunk to every other rank.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = False

    def __init__(self, total_bytes: float, world: int):
        self.total_bytes = total_bytes
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class Broadcast(CommKernel):
    """Broadcast: root sends full payload to all ranks.

    Bottleneck link (root's outgoing) carries the full tensor.
    transferred_bytes = total_bytes.
    """
    _single_input = True
    _single_output = False

    def __init__(self, total_bytes: float, world: int):
        self.total_bytes = total_bytes
        self.world = world
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class Scatter(CommKernel):
    """Scatter: root distributes distinct slices to each rank.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = True
    _single_output = False

    def __init__(self, total_bytes: float, world: int, dim: int = 0):
        self.total_bytes = total_bytes
        self.world = world
        self.dim = dim
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes / self.world


class Gather(CommKernel):
    """Gather: each rank sends its slice to root which concatenates.

    transferred_bytes = (W-1)/W × total_bytes.
    """
    _single_input = False
    _single_output = True

    def __init__(self, total_bytes: float, world: int, dim: int = 0):
        self.total_bytes = total_bytes
        self.world = world
        self.dim = dim
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return (self.world - 1) / self.world * self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes / self.world

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class Reduce(CommKernel):
    """Reduce: each rank sends full buffer; root reduces element-wise.

    Bottleneck link (into root) carries the full tensor.
    transferred_bytes = total_bytes.
    """
    _single_input = False
    _single_output = True

    def __init__(self, total_bytes: float, world: int, dtype: str = "bf16"):
        self.total_bytes = total_bytes
        self.world = world
        self.dtype_ = dtype
        super().__init__()

    @property
    def flops(self) -> float:
        n_elements = self.total_bytes / dtype_bytes(self.dtype_)
        return (self.world - 1) / self.world * n_elements

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes


class Send(CommKernel):
    """Point-to-point send (e.g. PP stage boundary, sender side).

    transferred_bytes = total_bytes.
    """

    def __init__(self, total_bytes: float):
        self.total_bytes = total_bytes
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return self.total_bytes

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return 0.0


class Recv(CommKernel):
    """Point-to-point recv (e.g. PP stage boundary, receiver side).

    transferred_bytes = total_bytes.
    """

    def __init__(self, total_bytes: float):
        self.total_bytes = total_bytes
        super().__init__()

    @property
    def transferred_bytes(self) -> float:
        return self.total_bytes

    @property
    def input_bytes(self) -> float:
        return 0.0

    @property
    def weight_bytes(self) -> float:
        return 0.0

    @property
    def output_bytes(self) -> float:
        return self.total_bytes
