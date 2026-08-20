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

**Threshold calibration status** (see `docs/superpowers/specs/2026-08-16-nli-judge-design.md`):
`short_answer_entailment_threshold` — used by `nli_verdict_short_answer`
(HaluEval-shaped harnesses: RQ1-RQ3, the RAG baseline) — is now calibrated
(2026-08-20, `experiments/calibrate_nli_short_answer_threshold.py`) against
HaluEval's own real `right_answer`/`hallucinated_answer` construction as
ground truth, over the 5,000-example union of all five checkpoints'
calibration splits — no human labeling needed, since the dataset already
encodes which answer is correct and which is a deliberate fabrication. See
`results/nli_short_answer_calibration.json`: 0.7 (the existing value)
scored 98.7% verdict-accuracy on the fit portion and 98.2% on a held-out
20%. **Honest limitation of this calibration itself**: it uses the
canonical answer text as a stand-in "generation" (self-entailment, always
near-maximal), so it tests whether the threshold correctly prefers the
matching answer over the wrong one — not how the judge performs on a
realistic, paraphrased or hedged real model generation. Real progress over
an arbitrary spot-check value, not a substitute for calibrating against
actual model outputs a human has judged.

`claim_supported_threshold`/`fraction_correct_threshold`/`fraction_incorrect_threshold`
— used by `factscore_style_verdict` (the FActScore-shaped harness) — remain
an uncalibrated first cut. FActScore has no equivalent free "known right vs.
known wrong" pair the way HaluEval does; calibrating these needs either real
model generations judged by a human, or reused human annotations for
different models' outputs (not this project's own five checkpoints). This
judge is real either way (semantic entailment, not substring matching), but
treat the FActScore thresholds specifically as unvalidated until that
ground truth exists.

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
