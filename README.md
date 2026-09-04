# RoofLang

RoofLang is a domain-specific language (DSL) for AI-driven architecting of LLM inference systems. It combines graph-based workload and hardware representations, semantics-preserving transformations, and roofline-based simulation to provide a verifiable, implementation-independent environment for exploring system architectures.

[Technical report]()<br>
[Project page](https://yzygitzh.github.io/rooflang)

## Code structure

```text
rooflang/
├── language/                     # Embedded DSL and graph IR
│   ├── graph.py                  # Compute graphs, hardware graphs, and fabrics
│   ├── tensor.py                 # Tensor shapes, dtypes, and sizes
│   ├── placement.py              # Operator/device and tensor/memory placement
│   ├── hardware/                 # Compute and memory components
│   ├── kernels/                  # Compute, communication, and identity kernels
│   └── optimization/             # Semantics-preserving graph transformations
├── runtime/
│   ├── simulator.py              # Roofline-based discrete-event simulator
│   ├── trace_export.py           # Google Trace Event Format export
│   └── graph_export.py           # Compute-graph visualization
├── programs/
│   ├── models/                   # LLM declarations and architecture optimizations
│   ├── presets/                  # Accelerator and cluster hardware specifications
│   └── experiments/              # Simulation and Pareto-search entry points
├── agents/find_pareto_frontier/  # Persistent optimizer-agent runner
└── tests/                        # Unit tests for the DSL, runtime, and programs
```

## Usage

### Run a single simulation

```bash
PYTHONPATH="$(pwd)/.." python -m rooflang.programs.experiments.main \
  --model dsv4_pro \
  --hardware B300Cluster \
  --stage decode \
  --batch-size 1
```

Use `--help` to see the complete option list.

### Search a throughput–interactivity Pareto frontier

The Pareto-search entry point sweeps legal parallel configurations and batch sizes. The following constrained example evaluates one workload and hardware family:

```bash
PYTHONPATH="$(pwd)/.." python -m rooflang.programs.experiments.find_pareto_frontier \
  --model dsv4_pro \
  --workloads decode-8k \
  --hardware b300 \
  --gpu-counts 8 \
  --pp-degrees 1 \
  --batch-multipliers 1 \
  --workers 1 \
  --output-dir results/dsv4_pro_decode
```

### Run the tests

```bash
PYTHONPATH="$(pwd)/.." pytest -q
```

## Cite us

If RoofLang is useful in your work, please cite the technical report:

```bibtex
@misc{yang2026rooflang,
  title        = {RoofLang: Enabling AI-Driven Architecting of LLM Inference Systems},
  author       = {{RoofLang Project}},
  year         = {2026},
  howpublished = {Technical report},
  url          = {https://yzygitzh.github.io/rooflang/}
}
```
