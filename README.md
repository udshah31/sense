# SENSE

Self-adaptive ENtropy-aware Symbolic Engine — a self-adaptive, entropy-gated
neuro-symbolic framework for cross-model hallucination mitigation: token entropy gates
routing to a symbolic verification stream, with thresholds calibrated per model.

Master's Culminating Project (Starred Paper, CSCI699), Computer Science MS,
St. Cloud State University.

See `CLAUDE.md` for full project context, architecture, and constraints.

## Status

Phase 0. Design decisions are committed (see `CLAUDE.md`). Scope: five 4-bit
checkpoints (Llama-3 8B, Mistral 7B, Qwen3 8B/4B/1.7B), HaluEval + FActScore,
and a Z3 symbolic backend with an atomic-claim decomposition front-end.

Working end-to-end on CPU against a tiny model, with the model-free and tiny-model test
suites passing: token entropy monitor (no-op verified; keeps raw and normalized
series), latency instrumentation, leakage-guarded calibration/development/test splits,
the entropy-gating policy (refuses to run uncalibrated), truncated-vs-exact entropy
bounds, the Z3 symbolic service (SPARQL/Wikidata kept as a fallback), the
gate-to-symbolic router (annotate-only), an NLI judge, and bootstrap confidence
intervals. Harnesses: RQ1 (fixed-threshold transfer), RQ2 (self-adaptive threshold),
RQ3 (accuracy-latency), the RAG and SelfCheckGPT-NLI baselines, FActScore symbolic
verification, and an RQ1 pre-flight entropy characterization.

Not yet run: any five-checkpoint HaluEval experiment on a GPU. `results/` holds only
pilot runs from the retired TruthfulQA scope. Run the `characterize` stage first.

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
├── docs/                     # specs, plans, proposal review, literature review
└── results/                  # logged runs
```

## Setup

1. Copy `.env.example` to `.env` and fill in `HF_TOKEN` (required for gated Llama-3 access).
2. Local (CPU) development doesn't need Docker — see individual service READMEs
   and `experiments/README.md`. `docker-compose.yml` defines a single GPU-capable
   `experiments` service for the real five-checkpoint runs
   (`docker compose run experiments [characterize|rq1|rq2|rq3|rag|selfcheck|all]`), matching
   `experiments/run_gpu_experiments.sh`.
