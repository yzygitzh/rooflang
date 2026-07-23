"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 DP=8.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model() — split_kernel for DP, control edges, placement
  C. simulate() — DES execution + trace export
"""

from rooflang.programs.dsv4_pro.config import *  # noqa: F401,F403
from rooflang.programs.dsv4_pro.model import (
    DecodeStepMeta, LayerMeta, declare_model, make_gated_up, make_gemm,
    make_norm,
)
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model, optimize_model_superchip,
)
from rooflang.programs.dsv4_pro.simulation import simulate
from rooflang.programs.dsv4_pro.visualization import visualize_layer
