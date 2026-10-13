# SENSE

Self-adaptive ENtropy-aware Symbolic Engine — a self-adaptive, entropy-gated
neuro-symbolic framework for cross-model hallucination mitigation: token entropy gates
routing to a symbolic verification stream, with thresholds calibrated per model.

Master's Culminating Project (Starred Paper, CSCI699), Computer Science MS,
St. Cloud State University.

See `CLAUDE.md` for full project context, architecture, and constraints.

## Status

Experiments complete (13 October 2026). Scope: five 4-bit checkpoints (Llama-3 8B, Mistral 7B,
Qwen3 8B/4B/1.7B), HaluEval for the headline results, and a Z3 symbolic backend with an
atomic-claim decomposition front-end. All RQ1 transfer pairs ran on the development split and,
once, on the held-out test split; RQ2 is derived from RQ1; RQ3, SelfCheckGPT-NLI (two
checkpoints) and a retrieval-gated smoke test (one checkpoint) ran on the development split.
See `results/README.md` for which file holds what, `docs/test-split-protocol.md` for how the test
split was handled, and the "2026-10-13 — experiments complete" section of `CLAUDE.md` for
the caveats that must accompany every number (recall ceiling, routing-rate effect, AI-assigned
judge labels, annotate-only merge-back).

Not done, by design or for budget: the entropy-aggregator ablation, the always-on RAG baseline,
Chain-of-Verification, and Docker image digest pinning (`experiments/Dockerfile.gpu` pins the
CUDA base image by tag, not digest).

Model-free and tiny-model test suites pass (`services/neural`, `eval`, `experiments`; run each
with `uv run pytest -q` from its directory).

## Layout

```
sense/
├── configs/                  # YAML: model, gate, dataset, rq1-3, run (seed), judge, baselines
├── services/
│   ├── neural/               # model loading, entropy monitor, latency
│   ├── symbolic/             # Z3 verifier, claim decomposition/extraction (+ SPARQL fallback)
│   ├── orchestrator/         # gate policy, routing, truncated-entropy bounds
│   └── rag/                  # FAISS + MiniLM comparison baseline
├── experiments/              # one script per table/figure; run_gpu_experiments.sh
├── eval/                     # routing quality, factuality, NLI judge, bootstrap, SelfCheck
├── data/                     # HaluEval/FActScore/TruthfulQA loaders, committed split indices
├── scripts/                  # kaggle_run.sh, colab_job.sh (GPU jobs); gpu_setup.sh, runpod_setup.sh
├── docs/                     # test-split protocol, specs, plans, proposal review, literature review
└── results/                  # saved runs; see results/README.md
```

## Setup

1. Copy `.env.example` to `.env` and fill in `HF_TOKEN` (required for gated Llama-3 access).
2. Local (CPU) development doesn't need Docker — see individual service READMEs
   and `experiments/README.md`. `docker-compose.yml` defines a single GPU-capable
   `experiments` service for the real five-checkpoint runs
   (`docker compose run experiments [characterize|rq1|rq2|rq3|rag|selfcheck|all]`), matching
   `experiments/run_gpu_experiments.sh`.
