# RoofLang

**RoofLang is an embedded domain-specific language and intermediate representation for AI-driven architecting of LLM inference systems, combining semantics-preserving graph transformations with roofline-based discrete-event simulation.**

RoofLang provides a verification-constrained design space in which an optimizer can declare LLM workloads and hardware as graphs, transform and place the workload, and evaluate the resulting architecture before committing to an implementation. It is intended as an analytical oracle for architecture exploration, not as a high-fidelity production-performance predictor.

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
