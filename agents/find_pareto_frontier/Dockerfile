FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        openssh-client \
        procps \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        time \
        unzip \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Install both native CLIs through their official installers and verify them at
# build time. Each run installs/updates the selected CLI again inside its own
# ephemeral home, so no build-time or previous-run CLI state is reused.
RUN curl -fsSL https://chatgpt.com/codex/install.sh | sh \
    && codex --version

RUN curl -fsSL https://claude.ai/install.sh | bash \
    && claude --version

COPY requirements.txt /tmp/rooflang-requirements.txt
RUN python3 -m venv /opt/rooflang-venv \
    && /opt/rooflang-venv/bin/pip install --no-cache-dir \
        --requirement /tmp/rooflang-requirements.txt \
    && rm /tmp/rooflang-requirements.txt

COPY entrypoint.sh /usr/local/bin/rooflang-agent
RUN chmod 0755 /usr/local/bin/rooflang-agent \
    && mkdir -p \
        /workspace/task \
        /workspace/results \
        /workspace/rooflang \
    && touch /workspace/task/AGENTS.md /workspace/results/raw_results.jsonl

ENV HOME=/home/agent \
    CODEX_HOME=/home/agent/.codex \
    CLAUDE_CONFIG_DIR=/home/agent/.claude \
    PATH=/home/agent/.local/bin:/opt/rooflang-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/workspace \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CONFIG_HOME=/home/agent/.config \
    XDG_DATA_HOME=/home/agent/.local/share \
    XDG_STATE_HOME=/home/agent/.local/state \
    XDG_CACHE_HOME=/tmp/cache

WORKDIR /tmp/rooflang-pareto-agent

ENTRYPOINT ["/usr/local/bin/rooflang-agent"]
