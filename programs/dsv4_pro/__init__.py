"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 DP=8.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model() — split_kernel for DP, control edges, placement
  C. simulate() — DES execution + trace export
"""

from .config import *  # noqa: F401,F403
from .model import declare_model
from .optimization import optimize_model, optimize_model_superchip
from .simulation import simulate
from .utils import DecodeStepMeta, LayerMeta, make_gated_up, make_gemm, make_norm
from .visualization import visualize_layer
