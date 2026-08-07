"""DeepSeek V4 Pro inference simulation — 8k prefill on B300 DP=8.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model_*() — split_kernel for DP, control edges, placement
  C. simulate() — DES execution + trace export
"""

from rooflang.programs.dsv4_pro.config import *  # noqa: F401,F403
from rooflang.programs.dsv4_pro.model import (
    DecodeStepMeta, LayerMeta, declare_model, make_gated_up, make_gemm,
    make_norm,
)
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_b300_cluster_a_dp8_ep8_1node,
    optimize_model_b300_cluster_a_dp8_ep8_2node,
    optimize_model_b300_superchip_a,
)
from rooflang.programs.dsv4_pro.simulation import simulate
from rooflang.programs.dsv4_pro.visualization import visualize_layer
