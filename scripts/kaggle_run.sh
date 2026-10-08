#!/usr/bin/env bash
# One-shot Kaggle runner: pick the biggest disk, clone a branch, install, run a command,
# copy results to /kaggle/working/sense-results. Replaces the step-by-step notebook cells
# (docs/kaggle-characterize.ipynb) and the broken Kaggle path of gpu_setup.sh.
#
# Needs HF_TOKEN in the environment (read it from Kaggle Secrets in the calling cell).
# Usage:  kaggle_run.sh <branch> '<shell command, run from the repo root>'
#   e.g.  kaggle_run.sh master './experiments/run_gpu_experiments.sh characterize'
#
# Everything (repo, uv cache, HF cache) goes on ONE disk so uv can hardlink; the disk
# outside /kaggle/working is used because /kaggle/working is only ~20 GB.
set -euo pipefail

BRANCH="${1:?usage: kaggle_run.sh <branch> '<command>'}"
CMD="${2:?usage: kaggle_run.sh <branch> '<command>'}"
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF_TOKEN is not set" >&2; exit 1; }

best="" best_free=0
for d in /kaggle/temp /kaggle/working /tmp /root; do
    [ -d "$d" ] && [ -w "$d" ] || continue
    free=$(df -BG --output=avail "$d" | tail -1 | tr -dc '0-9')
    echo "$d  ${free} GB free"
    if [ "$free" -gt "$best_free" ]; then best="$d" best_free="$free"; fi
done
[ "$best_free" -ge 45 ] || { echo "FATAL: only ${best_free} GB free on the best disk; need about 45" >&2; exit 1; }

BASE="$best/sense-env"
export REPO_DIR="$BASE/sense" UV_CACHE_DIR="$BASE/uv-cache" HF_HOME="$BASE/hf-cache"
export HF_CACHE_DIR="$HF_HOME" PATH="$HOME/.local/bin:$PATH"
echo "picked $best; repo at $REPO_DIR"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
rm -rf "$REPO_DIR"
mkdir -p "$BASE"
git clone -q --depth 1 --branch "$BRANCH" https://github.com/udshah31/sense.git "$REPO_DIR"
echo "commit: $(git -C "$REPO_DIR" log --oneline -1)"

for pkg in data services/neural services/orchestrator services/symbolic services/rag eval experiments; do
    echo "=== uv sync: $pkg ==="
    (cd "$REPO_DIR/$pkg" && uv sync -q)
done

(cd "$REPO_DIR/experiments" && uv run python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())")

status=0
(cd "$REPO_DIR" && bash -c "$CMD") 2>&1 | tee "$BASE/run.log" || status=$?

out=/kaggle/working/sense-results
mkdir -p "$out"
cp "$REPO_DIR"/results/*.json "$BASE/run.log" "$out"/ 2>/dev/null || true
echo "results copied to $out: $(ls "$out" | tr '\n' ' ')"
exit "$status"
