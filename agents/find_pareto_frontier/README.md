# Rooflang Pareto Agent Launcher

Ubuntu 24.04 launcher for the English task in `AGENTS.md`, supporting Codex and Claude Code.

## Run

```bash
cp agent.env.example /secure/path/agent.env
./run-agent.sh \
  --agent codex \
  --model dsv4_pro \
  --preset b300 \
  --results ../../../dsv4_pro_pareto_frontier \
  --env-file /secure/path/agent.env \
  --build
```

Use `--agent claude` for Claude Code, `--agent-model` for a model override, and `--runs-root` to change the retained-run parent. Equivalent `ROOFLANG_*` environment variables are supported; see `./run-agent.sh --help`.

## Isolation and Output

Every invocation creates `runs/run-<timestamp>-<random>/` containing the rendered task, an independent Rooflang copy, and `agent.trace.log`. Runs are retained and safe to execute in parallel; the input Rooflang tree is untouched.

Only the selected `programs/models/<model>/optimization.py` and copied `programs/experiments/` are writable. Results, task, remaining source, and container root are read-only. Provider instruction/config files and `.git` are excluded from the copy.

The exact rendered `AGENTS.md` content is the sole prompt. Each session uses fresh tmpfs-backed HOME/CLI/XDG state and an empty work directory; host CLI config, auth directories, and proxy settings are not inherited.

Put credentials, provider endpoints, and proxies explicitly in `--env-file`. Codex and Claude are installed with their official native `install.sh` scripts and the selected CLI updates at every run; Node/npm are not installed.

Codex emits JSONL events and Claude emits verbose `stream-json`. CLI updates, versions, stdout, stderr, metadata, and exit status are all streamed to the terminal and saved in `agent.trace.log`.
