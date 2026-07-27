# SENSE

Self-adaptive ENtropy-aware Symbolic Engine — a neuro-symbolic framework for real-time
hallucination mitigation via entropy-gated routing to a symbolic verification stream.

Master's Culminating Project (Starred Paper, CSCI699), Computer Science MS,
St. Cloud State University.

See `CLAUDE.md` for full project context, architecture, and constraints.

## Status

Phase 0 — environment bootstrap. Nothing implemented yet.

## Layout

```
sense/
├── configs/                  # YAML: model, thresholds, dataset, splits
├── services/
│   ├── neural/               # Python: model loading, LogitsProcessor, entropy, probes
│   ├── symbolic/             # symbolic backend (TBD, behind an HTTP contract)
│   └── orchestrator/         # Python: gate policy, routing, instrumentation
├── experiments/              # one script per table/figure in the paper
├── eval/                     # metrics: factuality + latency
├── data/                     # loaders and split index files (no large datasets)
└── results/                  # logged runs, metrics, plots
```

## Setup

1. Copy `.env.example` to `.env` and fill in `HF_TOKEN` (required for gated Llama-3 access).
2. `docker-compose.yml` defines the service containers; see individual service READMEs
   for local (non-Docker) development.
