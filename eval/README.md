Metrics: factuality + latency.

## Setup

```
uv sync
```

## factuality.py

`lexical_containment_verdict` is a **placeholder** factuality proxy: "correct" if
generated text contains the best/a correct TruthfulQA answer, "incorrect" if it
contains a known incorrect answer, "unknown" otherwise. This is not the project's
eventual factuality judge — real TruthfulQA evaluation typically uses a fine-tuned
judge model or human annotation. It exists to exercise the RQ3 harness end-to-end on
a tiny CPU model; any number this metric produces measures whether the pipeline
plumbing works, not paper-grade accuracy. Replace before reporting a result.

Latency metrics live in `services/neural/src/sense_neural/latency.py` — that's where
token arrival time is directly observable (inside `generate()`), not here.

## Tests

```
uv run pytest
```
