# Test-split protocol (declared before any test-split run)

Written 2026-10-10, before the test split has been evaluated by any harness. CLAUDE.md
constraint #1: the test split is touched once, at the end, for the reported numbers.

## What runs on the test split

- **RQ1 (and therefore RQ2, which is derived from it): all five transfer pairs.**
  `configs/rq1.yaml` with `eval_split: test` and `n_eval_examples: 1000`, i.e. the first
  1,000 indices of `data/splits/halueval.json`'s committed test list (a fixed prefix, not a
  runtime sample). Calibration stays on each checkpoint's own calibration split; nothing is
  fit on test data.
- Reported alongside, not instead of, the development-split results already recorded in
  `results/`.

## What stays on the development split, and why

- **RQ3** (latency; accuracy is unchanged by construction, annotate-only merge-back),
  **retrieval-gated**, and **SelfCheck**: their claims are about cost and about a comparison
  between mechanisms, not threshold fitting, and their GPU cost is large. They are reported
  as development-split results and labelled as such.
- The always-on RAG baseline (`rag_baseline_halueval.py`, which reads the test split) is
  **not** run; the development-split diagnostic in `results/retrieval_gated_halueval.console_summary.json`
  covers its purpose.

## Frozen before the run

Judge: binary verdict, `short_answer_entailment_threshold: 0.7`. Gate quantile 0.9, greedy
decoding, `max_new_tokens: 20`, 4-bit, pinned model revisions, seed 42, bootstrap 1,000
resamples at 95%. No setting is changed after the test run starts.

## Reporting rule

All five pairs are reported whatever they show. If the test results disagree with the
development results, both are reported and the disagreement is discussed, not resolved by
choosing the more favorable one.

## Known limitations carried into the test run

- The judge's 0.7 threshold is supported by 120 labels assigned by an AI assistant (not a
  human), 19 of them positive; a human spot-check is pending.
- ~85-89% of answers are incorrect while the gate routes ~10%, so gate recall is capped near
  12% by construction.
