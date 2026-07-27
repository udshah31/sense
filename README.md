# SENSE

Self-adaptive ENtropy-aware Symbolic Engine — a neuro-symbolic framework for real-time
hallucination mitigation via entropy-gated routing to a symbolic verification stream.

Master's Culminating Project (Starred Paper, CSCI699), Computer Science MS,
St. Cloud State University.

See `CLAUDE.md` for full project context, architecture, and constraints.

## Status

Phase 0. Design decisions committed (see `CLAUDE.md`). Working end-to-end on a tiny
CPU model against real TruthfulQA data: token entropy monitor (no-op verified),
generation latency instrumentation (TTFT, inter-token), calibration/development/test
split machinery (leakage-guarded), entropy-gating policy (refuses to run
uncalibrated), truncated-vs-exact entropy bounds, a symbolic backend (SPARQL against
Wikidata), and the gate-to-symbolic router (annotate-only). All three research
questions have working harnesses on real data: RQ1 (fixed-threshold transfer), RQ2
(self-adaptive threshold), RQ3 (accuracy-latency trade-off, with a placeholder
factuality metric). The RAG comparison baseline (FAISS + all-MiniLM-L6-v2 over a
committed Wikipedia passage corpus) also ran end-to-end on the full 327-example
TruthfulQA test split, with the same tiny-gpt2 caveat as the RQ harnesses. Not yet
built: real (non-tiny) model runs on GPU.

## Layout

```
sense/
├── configs/                  # YAML: model, thresholds, dataset, splits, RQ configs
├── services/
│   ├── neural/                # Python: model loading, entropy monitor, latency
│   ├── symbolic/               # Python: SPARQL/Wikidata verification backend
│   └── orchestrator/          # Python: gate policy, routing, instrumentation
├── experiments/               # one script per table/figure in the paper (RQ1-3 harnesses)
├── eval/                      # metrics: factuality (placeholder) + latency
├── data/                      # loaders and committed split index files
└── results/                   # logged runs, metrics
```

## Setup

1. Copy `.env.example` to `.env` and fill in `HF_TOKEN` (required for gated Llama-3 access).
2. `docker-compose.yml` defines the service containers; see individual service READMEs
   for local (non-Docker) development.
