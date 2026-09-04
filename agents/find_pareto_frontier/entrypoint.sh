#!/usr/bin/env bash
# Copyright (c) 2026 Ziyue Yang
# Licensed under the MIT License.

set -euo pipefail

readonly task_file=/workspace/task/AGENTS.md

if [[ ! -r "$task_file" ]]; then
    echo "error: $task_file is not readable" >&2
    exit 2
fi

if [[ ! -r /workspace/rooflang/programs/experiments/find_pareto_frontier.py ]]; then
    echo "error: /workspace/rooflang is not a Rooflang source tree" >&2
    exit 2
fi

if [[ ! -d /workspace/artifacts || ! -w /workspace/artifacts ]]; then
    echo "error: /workspace/artifacts is not a writable directory" >&2
    exit 2
fi

mkdir -p \
    "$HOME" \
    "$CODEX_HOME" \
    "$CLAUDE_CONFIG_DIR" \
    "$XDG_CONFIG_HOME" \
    "$XDG_DATA_HOME" \
    "$XDG_STATE_HOME" \
    "$HOME/.local/bin" \
    /tmp/cache \
    /tmp/matplotlib \
    /tmp/rooflang-pareto-agent

agent=${ROOFLANG_AGENT:-}
agent_model=${ROOFLANG_AGENT_MODEL:-}
cd /tmp/rooflang-pareto-agent

case "$agent" in
    codex)
        mkdir -p "$CODEX_HOME/packages"
        cp -R /opt/rooflang-agent/codex/standalone \
            "$CODEX_HOME/packages/standalone"
        ln -s "$CODEX_HOME/packages/standalone/current/bin/codex" \
            "$HOME/.local/bin/codex"
        echo "[agent-update] Running codex update" >&2
        codex update >&2
        hash -r
        codex --version >&2

        command=(
            codex exec
            --dangerously-bypass-approvals-and-sandbox
            --skip-git-repo-check
            --ignore-user-config
            --ignore-rules
            --ephemeral
            --json
            --color never
            --cd /tmp/rooflang-pareto-agent
            --add-dir /workspace/rooflang
            --add-dir /workspace/artifacts
        )
        if [[ -n "$agent_model" ]]; then
            command+=(--model "$agent_model")
        fi
        exec "${command[@]}" - < "$task_file"
        ;;
    claude)
        mkdir -p "$XDG_DATA_HOME/claude"
        cp -R /opt/rooflang-agent/claude/versions \
            "$XDG_DATA_HOME/claude/versions"
        claude_seed=$(find "$XDG_DATA_HOME/claude/versions" \
            -mindepth 1 -maxdepth 1 -type f -print -quit)
        [[ -n "$claude_seed" ]] \
            || { echo "error: Claude CLI seed is missing" >&2; exit 2; }
        ln -s "$claude_seed" "$HOME/.local/bin/claude"
        echo "[agent-update] Running claude update" >&2
        claude update >&2
        hash -r
        claude --version >&2

        command=(
            claude
            --print
            --safe-mode
            --dangerously-skip-permissions
            --no-session-persistence
            --output-format stream-json
            --verbose
            --add-dir /workspace/rooflang
            --add-dir /workspace/artifacts
        )
        if [[ -n "$agent_model" ]]; then
            command+=(--model "$agent_model")
        fi
        exec "${command[@]}" < "$task_file"
        ;;
    *)
        echo "error: ROOFLANG_AGENT must be 'codex' or 'claude'" >&2
        exit 2
        ;;
esac
