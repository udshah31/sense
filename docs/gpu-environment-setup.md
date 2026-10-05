# GPU Environment Setup

The dev laptop runs everything CPU-only against `sshleifer/tiny-gpt2` (see
CLAUDE.md's dev/GPU split). All five real checkpoints — Llama-3 8B Instruct,
Mistral 7B Instruct v0.3 and the Qwen3 8B/4B/1.7B ladder — only ever run here,
on a separate GPU host. This doc covers getting that host ready with
`scripts/gpu_setup.sh`, for two providers: RunPod and Colab Pro.

## Prerequisites (either provider)

1. **A Hugging Face token** with read access: https://huggingface.co/settings/tokens
2. **Accept the Llama-3 license** — visit
   https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct while logged in and
   accept the gated-repo terms. Without this, downloads 403 regardless of
   token validity.
3. Know your target CUDA version (`nvidia-smi` on the host, top-right corner)
   so `scripts/gpu_setup.sh`'s `CUDA_TAG` matches — most current RunPod/Colab
   images are CUDA 12.1+, so the script's `cu121` default usually works, but
   verify before assuming.

## RunPod

1. **Create a pod**: runpod.io → Deploy → pick a GPU. Every checkpoint runs at
   **4-bit** (CLAUDE.md constraint #2, pinned since the 2026-08-10 scope
   reconciliation — the earlier bf16 plan is retired), so peak memory is roughly
   1–6 GB per model and the harnesses load one or two at a time. A 24GB RTX 4090
   is comfortable; an A100 40GB is more than enough. These are estimates from
   parameter counts, not measurements — confirm against `nvidia-smi` on the first
   real load rather than trusting them. Choose a **PyTorch** template (ships CUDA + drivers
   pre-configured) — do not pick a bare Ubuntu image unless you want to
   install CUDA yourself.
2. **Attach a persistent volume** (RunPod calls this a "Network Volume") and
   mount it at e.g. `/workspace/hf-cache` — model weights are multi-GB and
   you don't want to re-download them every time the pod restarts. Set
   `HF_CACHE_DIR=/workspace/hf-cache` before running the setup script.
3. **SSH in** (RunPod gives you a connect command from the pod's dashboard).
4. Clone the repo and run the bootstrap:
   ```bash
   git clone <your-repo-url> sense && cd sense
   export HF_TOKEN=hf_xxxxxxxxxxxx
   export HF_CACHE_DIR=/workspace/hf-cache
   ./scripts/gpu_setup.sh
   ```
5. **Stop the pod when you're not actively running experiments** — RunPod
   bills per-hour while running, even idle. The network volume persists
   between stop/start, so the HF cache and repo checkout survive; only the
   compute charge stops.

## Colab Pro

Colab has no persistent disk by default and no real root shell, but a `!`
cell runs bash, and mounting Drive gives you persistent storage across
sessions. Paste these into separate cells:

**Cell 1 — mount Drive (for a persistent HF cache and repo checkout):**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — clone (first time only; skip if already cloned into Drive):**
```python
!git clone <your-repo-url> /content/drive/MyDrive/sense
```

**Cell 3 — bootstrap:**
```python
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxx"  # paste your real token
os.environ["HF_CACHE_DIR"] = "/content/drive/MyDrive/hf-cache"
os.environ["REPO_DIR"] = "/content/drive/MyDrive/sense"
!bash /content/drive/MyDrive/sense/scripts/gpu_setup.sh
```

**Colab-specific caveats:**
- **Runtime → Change runtime type → GPU** must be selected before any of
  this, or `nvidia-smi` in the script fails immediately and you're on CPU.
- Colab's assigned GPU (A100 vs L4 vs T4) depends on availability even on
  Pro. At 4-bit a T4 (16GB) should hold any single checkpoint with room to
  spare, but RQ1 and RQ2 load a *pair* at once (source and target), so check
  `nvidia-smi`'s reported memory before running those two in particular. The
  SelfCheckGPT baseline loads one checkpoint but generates N+1 times per
  example, so it is time-bound rather than memory-bound.
- **Sessions are ephemeral and time out** (Colab Pro: up to ~24h, but
  disconnects on inactivity) — fine for a single RQ1/RQ2/RQ3/RAG run each
  (minutes to low hours), risky for anything you'd want unattended overnight.
  RunPod is the better fit if a run needs to survive disconnects.
- `uv`'s installer writes to `~/.local/bin`, which doesn't persist across
  Colab sessions (only `/content/drive/...` does) — the script re-installs
  `uv` each session, which is fast and fine, just don't expect it to skip
  that step on a fresh runtime.

## Running an experiment once the host is set up

The experiment scripts are model-agnostic (CLAUDE.md's config-driven rule) —
switching from the CPU-test model to Llama-3/Mistral is a config change, not
a code change:

1. `configs/model.yaml`: change `active: cpu_test` to `active: llama3` (used
   by `calibrate_gate_truthful_qa.py`).
2. `configs/rq1.yaml`, `configs/rq2.yaml`: change `source_model`/`target_model`
   from `cpu_test`/`cpu_test_transfer_target` to the real models (e.g.
   `source_model: llama3`, `target_model: mistral` — this is the actual
   cross-family transfer test the RQ1/RQ2 harnesses exist to run).
3. `configs/rq3.yaml`, `configs/rag.yaml`: change `model: cpu_test` to
   `model: llama3` (or `mistral`), and remove or raise `n_eval_examples` —
   the CPU-tractability subsampling those configs currently apply doesn't
   apply on real GPU hardware.
4. Run from `experiments/`: `uv run python transfer_threshold_truthful_qa.py`
   (etc. for each harness).

Results still write to `results/*.json` with the same schema as the CPU-test
runs, just without the "pipeline-mechanics validation, not a scientific
finding" caveat once real models are involved.
