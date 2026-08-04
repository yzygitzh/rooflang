"""DeepSeek V4 Pro inference — Main entry point."""

import argparse

from rooflang.programs.presets.b300 import B300ClusterA, B300SuperChipA

from rooflang.programs.dsv4_pro.model import declare_model
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_b300_cluster_a_1node,
    optimize_model_b300_cluster_a_2node,
    optimize_model_b300_superchip_a,
)
from rooflang.programs.dsv4_pro.simulation import simulate
from rooflang.programs.dsv4_pro.visualization import visualize_layer


HARDWARE_MAP = {
    "B300ClusterA1Node": lambda: B300ClusterA(n_nodes=1),
    "B300ClusterA2Node": lambda: B300ClusterA(n_nodes=2),
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
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: config BATCH)")
    parser.add_argument("--n-decode-steps", type=int, default=1,
                        help="Number of decode steps to unroll (default 1)")
    parser.add_argument("--visualization", action="store_true",
                        help="Export layer graph visualization")
    args = parser.parse_args()

    hw = HARDWARE_MAP[args.hardware]()

    has_sim = args.prefill or args.decode
    seq_prefill = 8192 if args.prefill else None
    kv_prefill_len = 8192 if (args.decode and not args.prefill) else None

    # A. Declaration
    decl_kwargs = dict(
        seq_prefill=seq_prefill, decode=args.decode,
        kv_prefill_len=kv_prefill_len,
        n_decode_steps=args.n_decode_steps)
    if args.batch_size is not None:
        decl_kwargs["batch_size"] = args.batch_size
    g, layers, decode_steps, emb, read_input, kv_cache_reads, \
        pfx_out_head = declare_model(**decl_kwargs)

    # B. Visualization
    if args.visualization:
        viz_layer = None
        seeds = {emb, read_input}
        if layers:
            viz_layer = layers[0]
        elif decode_steps and decode_steps[0].layers:
            viz_layer = decode_steps[0].layers[0]
            seeds = {decode_steps[0].emb, decode_steps[0].read_input}
            if kv_cache_reads:
                seeds.add(kv_cache_reads[0])
        seeds.discard(None)
        if viz_layer:
            visualize_layer(g, viz_layer, extra_seeds=seeds)

    if has_sim:
        # C. Optimization
        if args.hardware == "B300SuperChipA":
            g, p = optimize_model_b300_superchip_a(g, hw)
        elif args.hardware == "B300ClusterA2Node":
            g, p = optimize_model_b300_cluster_a_2node(
                g, layers, hw, emb, read_input, decode_steps,
                kv_cache_reads, pfx_out_head)
        else:
            g, p = optimize_model_b300_cluster_a_1node(
                g, layers, hw, emb, read_input, decode_steps,
                kv_cache_reads, pfx_out_head)

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
