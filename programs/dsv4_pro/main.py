"""DeepSeek V4 Pro inference — Main entry point."""

import argparse

from rooflang.programs.presets.b300 import B300ClusterA, B300SuperChipA

from rooflang.programs.dsv4_pro.config import N_LAYERS
from rooflang.programs.dsv4_pro.model import declare_model
from rooflang.programs.dsv4_pro.optimization import (
    optimize_model_cluster_decode,
    optimize_model_cluster_prefill,
    optimize_model_superchip,
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
    parser.add_argument("--stage", required=True,
                        choices=("prefill", "decode"),
                        help="Inference stage to simulate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (default: config BATCH)")
    parser.add_argument("--cp", type=int, default=1,
                        help="Context-parallel degree (default: 1)")
    parser.add_argument("--dp", type=int, default=8,
                        help="Data-parallel degree (default: 8)")
    parser.add_argument("--ep", type=int, default=8,
                        help="Expert-parallel degree (default: 8)")
    parser.add_argument(
        "--pp-partition", type=int, nargs="+", default=[N_LAYERS],
        help="Layer counts assigned to successive pipeline stages")
    parser.add_argument("--visualization", action="store_true",
                        help="Export layer graph visualization")
    args = parser.parse_args()

    is_prefill = args.stage == "prefill"
    hw = HARDWARE_MAP[args.hardware]()

    # A. Declaration
    decl_kwargs = dict(
        seq_prefill=8192,
        decode=not is_prefill)
    if args.batch_size is not None:
        decl_kwargs["batch_size"] = args.batch_size
    g, layers, emb, read_input, kv_cache_reads, output_head = \
        declare_model(**decl_kwargs)

    # B. Visualization
    if args.visualization:
        viz_layer = None
        seeds = {emb, read_input}
        if layers:
            viz_layer = layers[0]
        if kv_cache_reads:
            seeds.add(kv_cache_reads[0])
        seeds.discard(None)
        if viz_layer:
            visualize_layer(g, viz_layer, extra_seeds=seeds)

    # C. Optimization
    if args.hardware == "B300SuperChipA":
        g, p = optimize_model_superchip(g, hw)
    else:
        n_gpus = 16 if args.hardware == "B300ClusterA2Node" else 8
        optimize_kwargs = dict(
            cp=args.cp, dp=args.dp, ep=args.ep,
            pp_partition=args.pp_partition,
            n_gpus=n_gpus)
        if is_prefill:
            g, p = optimize_model_cluster_prefill(
                g, layers, hw, emb, read_input, output_head,
                **optimize_kwargs)
        else:
            g, p = optimize_model_cluster_decode(
                g, layers, hw, emb, read_input, kv_cache_reads,
                output_head, **optimize_kwargs)

    # D. Simulation
    trace_name = f"dsv4_pro_{args.stage}_{args.hardware}.json"
    result = simulate(g, p, hw, trace_name)
    print(f"{args.stage} ({args.hardware}): "
          f"{result.total_time_us:.1f} us "
          f"({result.total_time_us / 1000:.1f} ms)")


if __name__ == "__main__":
    main()
