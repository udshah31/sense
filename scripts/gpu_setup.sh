#!/usr/bin/env bash
# One-time bootstrap for the GPU environment (RunPod, Lambda Labs, Colab, or any bare
# Linux box with an NVIDIA GPU). Installs uv, syncs every package's venv, and swaps in
# a CUDA-enabled torch wheel — the default `uv sync` pulls the CPU wheel, which is
# correct for the dev laptop but wrong here.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/gpu_setup.sh
#
# Optional env vars:
#   CUDA_TAG      torch wheel index tag, e.g. cu121, cu124, cu126 (default: cu121 —
#                 check `nvidia-smi`'s reported CUDA version and adjust if installs fail)
#   REPO_DIR      where to clone/find the repo (default: current directory, assumes
#                 you're already inside a checkout)
set -euo pipefail

CUDA_TAG="${CUDA_TAG:-cu121}"
REPO_DIR="${REPO_DIR:-$(pwd)}"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set. Llama-3 (gated) downloads will fail auth." >&2
    echo "Get a token at https://huggingface.co/settings/tokens and accept the" >&2
    echo "Llama-3 license at https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct first." >&2
fi

cd "$REPO_DIR"

echo "=== nvidia-smi ==="
nvidia-smi || { echo "ERROR: no NVIDIA GPU visible. Wrong host?" >&2; exit 1; }

echo "=== installing uv ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# Every package here is an independent uv-managed venv (see CLAUDE.md's repo layout).
# services/neural is the one that actually needs the CUDA torch wheel; the rest
# depend on it transitively or don't touch torch at all.
PACKAGES="data services/neural services/orchestrator services/symbolic services/rag eval experiments"

for pkg in $PACKAGES; do
    echo "=== uv sync: $pkg ==="
    (cd "$pkg" && uv sync)
done

echo "=== installing CUDA torch wheel into services/neural's venv ($CUDA_TAG) ==="
(
    cd services/neural
    uv pip install "torch==2.7.0" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --force-reinstall --no-deps
)

echo "=== verifying CUDA is visible to torch ==="
(
    cd services/neural
    uv run python -c "
import torch
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'torch cannot see the GPU — check CUDA_TAG matches nvidia-smi\'s CUDA version'
print('device:', torch.cuda.get_device_name(0))
"
)

echo "=== writing HF_TOKEN to .env ==="
if [ -n "${HF_TOKEN:-}" ]; then
    if [ -f .env ] && grep -q "^HF_TOKEN=" .env; then
        sed -i.bak "s/^HF_TOKEN=.*/HF_TOKEN=${HF_TOKEN}/" .env && rm -f .env.bak
    else
        cp -n .env.example .env 2>/dev/null || true
        echo "HF_TOKEN=${HF_TOKEN}" >> .env
    fi
fi

echo ""
echo "=== GPU environment ready ==="
echo "Next: point HF_CACHE_DIR at a persistent volume if this host is ephemeral"
echo "(RunPod/Lambda: mount one at pod creation; Colab: mount Drive, see docs/gpu-environment-setup.md),"
echo "then flip configs/model.yaml's 'active' and the relevant experiment configs"
echo "(configs/rq1.yaml, rq2.yaml, rq3.yaml, rag.yaml) from cpu_test to llama3/mistral."
