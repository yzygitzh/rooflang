#!/usr/bin/env bash
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

mkdir -p \
    "$HOME" \
    "$CODEX_HOME" \
    "$CLAUDE_CONFIG_DIR" \
    "$XDG_CONFIG_HOME" \
    "$XDG_DATA_HOME" \
    "$XDG_STATE_HOME" \
    /tmp/cache \
    /tmp/matplotlib \
    /tmp/rooflang-pareto-agent

agent=${ROOFLANG_AGENT:-}
agent_model=${ROOFLANG_AGENT_MODEL:-}
cd /tmp/rooflang-pareto-agent

case "$agent" in
    codex)
        echo "[agent-update] Updating Codex with the official installer" >&2
        curl -fsSL https://chatgpt.com/codex/install.sh | sh
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
        )
        if [[ -n "$agent_model" ]]; then
            command+=(--model "$agent_model")
        fi
        exec "${command[@]}" - < "$task_file"
        ;;
    claude)
        echo "[agent-update] Updating Claude Code with the official installer" >&2
        curl -fsSL https://claude.ai/install.sh | bash
        hash -r
        claude update >&2
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
        )
        if [[ -n "$agent_model" ]]; then
            command+=(--model "$agent_model")
        fi
        exec "${command[@]}" "$(<"$task_file")"
        ;;
    *)
        echo "error: ROOFLANG_AGENT must be 'codex' or 'claude'" >&2
        exit 2
        ;;
esac
