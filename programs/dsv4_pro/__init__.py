"""DeepSeek V4 Pro inference simulation on B300 hardware.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model_*() — parallel graph transforms and placement
  C. simulate() — DES execution + trace export
"""

from rooflang.programs.dsv4_pro.config import *  # noqa: F401,F403
from rooflang.programs.dsv4_pro.model import (
    DecodeStepMeta, LayerMeta, declare_model, make_gated_up, make_gemm,
    make_norm,
)
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_b300_cluster_a_cp_dp_ep_pp_decode,
    optimize_model_b300_cluster_a_cp_dp_ep_pp_prefill,
    optimize_model_b300_superchip_a,
)
from rooflang.programs.dsv4_pro.simulation import simulate
from rooflang.programs.dsv4_pro.visualization import visualize_layer
