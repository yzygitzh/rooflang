"""DeepSeek V4 Pro inference — Main entry point."""

import argparse

from rooflang.programs.presets.b300 import B300ClusterA, B300SuperChipA

from rooflang.programs.dsv4_pro.model import declare_model
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model, optimize_model_superchip,
)
from rooflang.programs.dsv4_pro.simulation import simulate
from rooflang.programs.dsv4_pro.visualization import visualize_layer


HARDWARE_MAP = {
    "B300ClusterA": lambda: B300ClusterA(n_nodes=1),
    "B300SuperChipA": lambda: B300SuperChipA(),
}


def main():
    parser = argparse.ArgumentParser(description="DSV4 Pro inference simulation")
    parser.add_argument("--hardware", required=True,
                        choices=list(HARDWARE_MAP.keys()),
                        help="Hardware configuration")
    parser.add_argument("--prefill", action="store_true",
                        help="Run prefill phase")
    parser.add_argument("--decode", action="store_true",
                        help="Run decode phase")
    parser.add_argument("--visualization", action="store_true",
                        help="Export layer graph visualization")
    args = parser.parse_args()

    hw = HARDWARE_MAP[args.hardware]()
    is_superchip = args.hardware == "B300SuperChipA"

    has_sim = args.prefill or args.decode
    seq_prefill = 8192 if args.prefill else None
    n_decode_steps = 4 if args.decode else 0
    kv_prefill_len = 8192 if (args.decode and not args.prefill) else None

    # A. Declaration
    g, layers, decode_steps, emb, read_input, kv_cache_reads, \
        pfx_out_head = declare_model(
            seq_prefill=seq_prefill, n_decode_steps=n_decode_steps,
            kv_prefill_len=kv_prefill_len)

    # B. Visualization
    if args.visualization and layers:
        visualize_layer(g, layers[0], extra_seeds={emb, read_input})

    if has_sim:
        # C. Optimization
        if is_superchip:
            g, p = optimize_model_superchip(g, layers, hw, emb)
        else:
            g, p = optimize_model(g, layers, hw, emb, read_input,
                                  decode_steps, kv_cache_reads, pfx_out_head)

        # D. Simulation
        mode = []
        if args.prefill:
            mode.append("prefill")
        if args.decode:
            mode.append("decode")
        trace_name = f"dsv4_pro_{'_'.join(mode)}_{args.hardware}.json"

        result = simulate(g, p, hw, trace_name)
        print(f"{'+'.join(mode)} ({args.hardware}): "
              f"{result.total_time_us:.1f} us "
              f"({result.total_time_us / 1000:.1f} ms)")


if __name__ == "__main__":
    main()
