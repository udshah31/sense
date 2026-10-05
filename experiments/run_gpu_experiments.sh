#!/usr/bin/env bash
# One-command runbook for the real five-checkpoint GPU run (CLAUDE.md's
# reproducibility requirement: "one command reproduces the headline result").
# Not runnable on the local CPU dev environment by design — see CLAUDE.md's
# "Development environment vs. GPU environment": every component is unit-tested
# on CPU with a tiny model, but the five real checkpoints only ever run here.
#
# Usage:
#   HF_TOKEN=... ./run_gpu_experiments.sh [characterize|rq1|rq2|rq3|rag|selfcheck|all]
#
# RUN `characterize` FIRST, on its own, before paying for `all`. It is the RQ1
# pre-flight diagnostic (experiments/characterize_entropy_distributions.py): it
# reports whether the checkpoints' entropy distributions differ enough for a fixed
# threshold to notice, on both the raw and normalized scales. If the raw-scale
# threshold spread comes back near 1.0, RQ1's premise does not hold as the proposal
# states it and the framing needs revisiting before five checkpoints of compute go
# into answering it. It is also the first time any real checkpoint is loaded, so it
# doubles as a cheap smoke test of 4-bit loading and the entropy monitor.
# `all` deliberately does NOT include it — it is a decision point, not a stage.
#
# Defaults to "all". Each stage is independent — a mid-run failure in RQ2, say,
# does not block re-running RQ1's already-written results; just re-invoke with
# that stage's name once the issue is fixed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENTS_DIR="$REPO_ROOT/experiments"
RESULTS_DIR="$REPO_ROOT/results"
STAGE="${1:-all}"

log() { printf '[run_gpu_experiments] %s\n' "$*" >&2; }
fail() { log "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------------------
# Preflight — fail loudly and early rather than burning GPU time on a run that
# was never going to produce comparable numbers (CLAUDE.md's non-negotiable
# constraints don't raise exceptions when violated at the data-split/gate
# level, so every check that CAN be made mechanical here, is).
# ---------------------------------------------------------------------------

cd "$EXPERIMENTS_DIR"

command -v uv >/dev/null 2>&1 || fail "uv not found on PATH — install it before running the GPU suite"

[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is not set — required for gated repos (Llama-3). Export it or source .env first."

uv run python - <<'PYEOF' || fail "CUDA is not available in this environment — this script targets the GPU environment only, not the CPU dev box"
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
PYEOF

log "preflight OK — capturing run metadata"

mkdir -p "$RESULTS_DIR"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
METADATA_FILE="$RESULTS_DIR/run_metadata_${RUN_TIMESTAMP}.json"

# CLAUDE.md's "Reproducibility requirements": every run logs hardware, GPU
# model, driver, CUDA version, model revision SHA (already embedded per-result
# by _common.py), quantization scheme (ditto), decoding config (ditto), seed.
# This captures the run-level facts the per-result JSON files don't carry
# themselves (hardware/driver/CUDA/git SHA) into one sidecar file per invocation.
uv run python - "$METADATA_FILE" "$RUN_TIMESTAMP" "$STAGE" <<'PYEOF'
import json
import platform
import subprocess
import sys

import torch

out_path, run_timestamp, stage = sys.argv[1], sys.argv[2], sys.argv[3]


def _sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None


metadata = {
    "run_timestamp_utc": run_timestamp,
    "stage": stage,
    "git_commit": _sh(["git", "rev-parse", "HEAD"]),
    "git_branch": _sh(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "driver_version": _sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
}
with open(out_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(json.dumps(metadata, indent=2))
PYEOF

log "run metadata written to $METADATA_FILE"

run_stage() {
    local name="$1" script="$2"
    log "=== starting $name ($script) ==="
    if uv run python "$script"; then
        log "=== $name complete ==="
    else
        fail "$name failed — see output above. Results already written by earlier stages are untouched."
    fi
}

case "$STAGE" in
    characterize)
        run_stage "entropy-distribution characterization (RQ1 pre-flight)" characterize_entropy_distributions.py
        log "Read threshold_spread.raw.max_over_min_ratio and summary_by_axis before running 'all'."
        ;;
    rq1)
        run_stage "RQ1 (fixed-threshold transfer)" transfer_threshold_halueval.py
        ;;
    rq2)
        run_stage "RQ2 (self-adaptive threshold)" adaptive_threshold_halueval.py
        ;;
    rq3)
        run_stage "RQ3 (accuracy-latency trade-off)" rq3_accuracy_latency_halueval.py
        ;;
    rag)
        run_stage "RAG baseline" rag_baseline_halueval.py
        ;;
    selfcheck)
        run_stage "SelfCheckGPT-NLI baseline (post-hoc verification)" selfcheck_baseline_halueval.py
        ;;
    all)
        run_stage "RQ1 (fixed-threshold transfer)" transfer_threshold_halueval.py
        run_stage "RQ2 (self-adaptive threshold)" adaptive_threshold_halueval.py
        run_stage "RQ3 (accuracy-latency trade-off)" rq3_accuracy_latency_halueval.py
        run_stage "RAG baseline" rag_baseline_halueval.py
        run_stage "SelfCheckGPT-NLI baseline (post-hoc verification)" selfcheck_baseline_halueval.py
        ;;
    *)
        fail "unknown stage '$STAGE' — expected one of: characterize, rq1, rq2, rq3, rag, selfcheck, all"
        ;;
esac

log "done. Results in $RESULTS_DIR, run metadata in $METADATA_FILE"
