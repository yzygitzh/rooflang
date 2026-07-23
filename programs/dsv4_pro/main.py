"""DeepSeek V4 Pro inference — Main entry point."""

from rooflang.programs.presets.b300 import B300ClusterA, B300SuperChipA

from rooflang.programs.dsv4_pro.model import declare_model
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model, optimize_model_superchip,
)
from rooflang.programs.dsv4_pro.simulation import simulate
from rooflang.programs.dsv4_pro.visualization import visualize_layer


def main():
    # A. Declaration
    hw = B300ClusterA(n_nodes=1)
    g, layers, decode_steps, emb, read_input, kv_cache_reads, pfx_out_head = \
        declare_model()

    visualize_layer(g, layers[0], extra_seeds={emb, read_input})

    # B. Optimization
    g, p = optimize_model(g, layers, hw, emb, read_input,
                          decode_steps, kv_cache_reads, pfx_out_head)

    # C. Simulation
    result = simulate(g, p, hw, "dsv4_pro_prefill.json")
    print(f"Prefill: {result.total_time_us:.1f} us "
          f"({result.total_time_us / 1000:.1f} ms)")

    # ── SuperChip (zero-comm) comparison ──
    hw_sc = B300SuperChipA()
    g_sc, layers_sc, _, emb_sc, _, _, _ = declare_model()

    # B. Optimization (no splits)
    g_sc, p_sc = optimize_model_superchip(g_sc, layers_sc, hw_sc, emb_sc)

    # C. Simulation
    result_sc = simulate(g_sc, p_sc, hw_sc, "dsv4_pro_superchip.json")
    print(f"Prefill (SuperChip): {result_sc.total_time_us:.1f} us "
          f"({result_sc.total_time_us / 1000:.1f} ms)")


if __name__ == "__main__":
    main()
