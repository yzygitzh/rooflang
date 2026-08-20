"""DeepSeek V4 Flash inference simulation on B300 hardware.

Three-phase structure:
  A. declare_model() — build logical compute graph (add_kernel + add_data_edge)
  B. optimize_model_*() — parallel graph transforms and placement
  C. simulate() — DES execution + trace export
"""

from rooflang.programs.models.dsv4_flash.config import *  # noqa: F401,F403
from rooflang.programs.models.dsv4_flash.model import (
    LayerMeta, declare_model,
)
from rooflang.programs.models.dsv4_flash.optimization import (
    optimize_model_cluster_decode,
    optimize_model_cluster_prefill,
    optimize_model_superchip,
)
from rooflang.programs.experiments.simulation import simulate


def __getattr__(name):
    if name == "visualize_layer":
        from rooflang.programs.experiments.visualization import visualize_layer
        return visualize_layer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
