#!/usr/bin/env bash
# One-command setup for a RunPod pod with a persistent network volume at /workspace.
# Clones the repo, keeps the repo + uv cache + Hugging Face cache on ONE filesystem (so uv
# hardlinks instead of copying — a split cache is what filled the disk on Kaggle), and runs
# scripts/gpu_setup.sh. See docs/gpu-environment-setup.md.
#
# Usage:
#   HF_TOKEN=hf_xxx bash runpod_setup.sh
#   (or: curl the file from the repo, since the repo does not exist on the pod yet)
#
# Optional env vars:
#   WORKSPACE   volume mount point (default: /workspace)
#   REPO_URL    repo to clone (default: https://github.com/udshah31/sense.git)
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/udshah31/sense.git}"
REPO_DIR="$WORKSPACE/sense"

[ -n "${HF_TOKEN:-}" ] || { echo "ERROR: set HF_TOKEN first (HF_TOKEN=hf_xxx bash $0)" >&2; exit 1; }
[ -d "$WORKSPACE" ] || { echo "ERROR: $WORKSPACE does not exist — is the network volume mounted there?" >&2; exit 1; }

# Everything on one filesystem. The token is deliberately NOT written to this file.
ENV_FILE="$WORKSPACE/sense-env.sh"
cat > "$ENV_FILE" <<ENVEOF
export HF_HOME="$WORKSPACE/hf-cache"
export HF_CACHE_DIR="\$HF_HOME"
export UV_CACHE_DIR="$WORKSPACE/uv-cache"
export REPO_DIR="$REPO_DIR"
export PATH="\$HOME/.local/bin:\$PATH"
ENVEOF
# shellcheck disable=SC1090
source "$ENV_FILE"

if [ -d "$REPO_DIR/.git" ]; then
    echo "=== repo exists, updating ==="
    git -C "$REPO_DIR" pull --ff-only
else
    echo "=== cloning $REPO_URL ==="
    git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" log --oneline -1

echo "=== running gpu_setup.sh ==="
export HF_TOKEN
bash "$REPO_DIR/scripts/gpu_setup.sh"

echo ""
echo "=== RunPod setup done ==="
echo "In every new shell:  source $ENV_FILE   (and export HF_TOKEN=...)"
echo "Run a stage in tmux:  cd $REPO_DIR && ./experiments/run_gpu_experiments.sh rq1 2>&1 | tee $WORKSPACE/rq1.log"
echo "Copy results off the pod before stopping it: $REPO_DIR/results/"
