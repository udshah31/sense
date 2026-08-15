Metrics: factuality + latency.

## Setup

```
uv sync
```

## factuality.py

`lexical_containment_verdict` is a **placeholder** factuality proxy: "correct" if
generated text contains the best/a correct answer, "incorrect" if it contains a
known incorrect answer, "unknown" otherwise (used against both TruthfulQA's
best/correct/incorrect answers and HaluEval's right/hallucinated answer). This is
not the project's eventual factuality judge — no fine-tuned judge model or
LLM-as-judge API is wired in yet; that's unfinished work, not a design choice. It
exists to exercise the RQ3 and RAG-baseline harnesses end-to-end on a tiny CPU
model; any number this metric produces measures whether the pipeline plumbing
works, not paper-grade accuracy. Every result scored with it carries
`PLACEHOLDER_FACTUALITY_METRIC_LABEL` verbatim in its `factuality_metric` field,
and `write_results` (`experiments/_common.py`) prints a loud stderr warning
whenever it sees that label — so a pipeline-mechanics number can't quietly sit in
`results/` looking like a final one. Replace `lexical_containment_verdict` (not the
reporting layer below it) once a real judge is ready to wire in.

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
