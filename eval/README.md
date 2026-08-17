Metrics: factuality + latency.

## Setup

```
uv sync
```

## factuality.py / nli_judge.py

`nli_judge.py` is the real factuality scorer: a pinned NLI model
(`configs/nli_judge.yaml` — `cliang1453/deberta-v3-xsmall-mnli`) scores
semantic entailment rather than substring containment.
`nli_verdict_short_answer` handles HaluEval-shaped short-answer QA;
`factscore_style_verdict` handles FActScore-shaped open-ended generation by
splitting it into atomic (sentence-level) claims and scoring the supported
fraction against a reference text, matching FActScore's own methodology.
Every result reports a `factuality_metric` field naming the model and
pinned revision that produced it (`NLI_METRIC_LABEL_TEMPLATE`).

**Named limitation** (see `docs/superpowers/specs/2026-08-16-nli-judge-design.md`):
the entailment/fraction thresholds in `configs/nli_judge.yaml` are a first
cut chosen for reasonable behavior on manual spot checks, not calibrated
against a human-labeled validation set — none exists yet for this project.
This judge is real (semantic entailment, not substring matching), but treat
its threshold calibration as unvalidated until that ground truth exists.

`summarize_factuality` and `assert_factuality_metrics_reported_together` are the
reporting layer, and apply regardless of which scorer eventually produces the
verdicts. CLAUDE.md/the proposal require `task_accuracy`, `hallucination_rate`,
and `abstention_rate` to always be reported together, in every condition — never
`hallucination_rate` alone, since abstaining trivially drives it toward zero at the
cost of accuracy. `summarize_factuality` computes all three off one shared
denominator (every example attempted, including abstained ones), and
`assert_factuality_metrics_reported_together` — wired into every experiment's
`write_results` call — raises `IncompleteFactualityReportError` if a result ever
reports some but not all three.

Latency metrics live in `services/neural/src/sense_neural/latency.py` — that's where
token arrival time is directly observable (inside `generate()`), not here.

## Tests

```
uv run pytest
```
