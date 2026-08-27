# Rooflang Overlapped Pareto-Frontier Optimization Agent

## 1. Mission and Inputs

Improve the overlapped Pareto frontier for one Rooflang model/preset pair by improving legal model placement/graph optimization and, when useful, experiment/search heuristics. Continue until Section 7's theoretical stopping condition is proved; one improvement is not sufficient.

- Isolated Rooflang copy: `/workspace/rooflang`
- Model: `{{MODEL}}`
- Preset: `{{PRESET}}`
- Complete prior results: `/workspace/results`
- Persistent run artifacts and scratch: `/workspace/artifacts`

This file is the complete task input. Do not rely on chat history, hidden state, or other repository instructions. The model, preset, paths, baseline results, and acceptance rules are immutable.

## 2. Objective and Acceptance

For records matching the preset, maximize `tokens_per_s_user_overlapped` (x) and `tokens_per_s_gpu_overlapped` (y), requiring `memory_feasible_overlapped == true`. Compute a separate frontier for every `(workload, hardware, n_gpus)` group. Dominance means no worse in x and y and strictly better in at least one.

For each group define:

- `F_old`: frontier independently recomputed from supplied records.
- `F_final`: frontier from fresh full-domain results using the final source state.
- `F_union`: frontier of `F_old` union `F_final`.

For `n` in both `F_final` and `F_union`, let `D(n)` contain every `o` in `F_old` that is absent from `F_union` and weakly dominated by `n`. Point `n` is significant only when `D(n)` is nonempty and, for every `o` in `D(n)`, at least one holds:

- `(n.x - o.x) / o.x >= 0.05`
- `(n.y - o.y) / o.y >= 0.05`

Coordinates should be positive; report any zero denominator and use a documented absolute check. Never round before comparison or use tolerance to reach 5%. Report `IMPROVED` only if every `F_old` point is weakly dominated by an `F_final` point in its group and at least one significant point exists.

## 3. Modification Boundary

Persistent source changes are allowed only in:

1. `/workspace/rooflang/programs/models/{{MODEL}}/optimization.py`
2. `/workspace/rooflang/programs/experiments/`

The model file may change only placement and optimization: device/tensor-memory mapping, legal parallel or PP placement, legal graph transforms, and local helpers for them. The experiment subtree may change legal case generation, exploration, pruning, batch growth, scheduling, parallelism, resume/output handling, and frontier analysis.

Do not introduce or alter non-default stream assignments.

Never change model definitions or semantics, architecture, shapes, dtypes, operations, another model file, presets, tests, simulator/language/runtime code, hardware/timing/memory accounting, feasibility, metrics, grouping, dominance, or this 5% rule. Never reduce required work/domain, bypass dependencies, falsify accounting, hard-code answers, commit, stage, or push. Report suspected out-of-scope bugs and impact without fixing or compensating for them.

Generated records, traces, plots, and notes belong in scratch. Supplied results are read-only and must never be resumed into or overwritten. Experiment edits must discover or validate legal configurations, not redefine success; independently verify any edited evaluation logic against Section 2.

## 4. Baseline and Analysis

Before editing:

1. Snapshot the optimization file and experiment subtree in scratch.
2. Inspect the full search/experiment path, simulator, placement APIs, passes, model, preset, and relevant tests.
3. Read every supplied JSONL record, including OOM/errors; report status counts, exact case domain, OOM bounds, and `F_old`.
4. Independently recompute frontiers from raw JSONL and explain discrepancies.
5. Run relevant tests and representative untouched simulations.

Quantify each frontier region's compute, memory, communication, dependency, PP imbalance/bubble, concurrency, feasibility, and search-coverage bounds. Inspect representative frontier/near-frontier traces. Calculate optimistic bounds before editing and reject levers that cannot meet Section 2.

Cover device/stream/memory placement, graph transforms, communication ordering, parallel mapping, PP boundaries/balance, overlap dependencies, case construction, pruning, batch/OOM growth, scheduling, resume behavior, and frontier computation. Maintain a scratch hypothesis ledger with mechanism, bound, change, cases, result, and decision.

## 5. Optimization Loop

For each credible hypothesis: state a falsifiable prediction; make the smallest allowed change; run syntax/tests/placement checks and focused fresh simulations; independently compare exact matching cases; inspect regressions; retain only the best valid state.

After a promising screen, rerun every supplied case identity for the preset. Include newly found legal cases and extend moved OOM boundaries using the original legal growth rule. Audit pruning against a less-pruned tractable reference. Fast subsets may screen ideas but cannot establish acceptance.

## 6. Final Validation and State

Run every supplied case plus required extensions in fresh processes. Independently recompute `F_final`, `F_union`, preservation, `D(n)`, and both percentage deltas. Run relevant model, placement, optimization, simulator, and search tests. Diff both writable locations against snapshots and verify no other source changed.

If Section 2 fails, restore both writable locations exactly to their initial run-copy state and report `NO_CHANGE`.

## 7. Theoretical Stopping Condition

Stop only when further legal model or experiment changes have no credible prospect of a validated significant gain. Quantify the attainable envelope and active bound for every frontier region; test or concretely rule out every Section 4 lever and relevant combinations; establish adequate search coverage; and show every remaining gap is below 5%/numerical resolution, blocked by fixed model/preset bounds, or requires a forbidden change. Complete Section 6 before stopping; otherwise continue.

## 8. Final Response

Use exactly these headings: `STATUS`, `TARGET`, `BASELINE FRONTIER`, `FINAL FRONTIER`, `SIGNIFICANCE WITNESSES`, `KEPT MODEL OPTIMIZATIONS`, `KEPT EXPERIMENT CHANGES`, `VALIDATION`, `THEORETICAL STOPPING ARGUMENT`, `REJECTED LEGAL AVENUES`, `OUT-OF-SCOPE ISSUES`, `FINAL FILE STATE`.

Include exact coordinates and case identities, every displaced-old witness and both deltas, commands/outcomes, changed experiment files, and confirmation that no source outside the two allowed locations changed.
