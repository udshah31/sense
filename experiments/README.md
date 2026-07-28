One script per table/figure in the paper.

## Setup

```
uv sync
```

## calibrate_gate_truthful_qa.py

Wires `GatePolicy` against real TruthfulQA calibration entropies: for each example
in the committed calibration split, greedily generates a short answer with
`TokenEntropyMonitor` attached and reduces its per-token trace to a mean-entropy
scalar, then fits the gate threshold on those scalars (leakage-guarded against the
committed split). Model identity, quantile, and decoding config all come from
`../configs/model.yaml` / `../configs/gate.yaml` — swapping `active` in `model.yaml`
to `llama3`/`mistral` on the GPU environment runs the identical code path.

```
uv run python calibrate_gate_truthful_qa.py
```

Writes `../results/gate_calibration_truthful_qa_<model_name>.json`.

## transfer_threshold_truthful_qa.py

RQ1 harness: calibrates natively on the source model, calibrates a separate native
baseline on the target model (comparison only, never applied), then applies the
source's threshold directly to the target model via `GatePolicy.set_threshold` and
compares its routing decisions against the target's own native gate. Evaluated on
the **development** split only — test stays untouched until the final reported
numbers. Model pair and eval split come from `../configs/rq1.yaml`;
`cpu_test_transfer_target` in `../configs/model.yaml` is a second CPU-testable model
from a different family (GPT-NeoX, 1024-token vocab) standing in for "a model from a
different family" the way llama3/mistral do on the GPU environment.

```
uv run python transfer_threshold_truthful_qa.py
```

Writes `../results/rq1_transfer_truthful_qa_<source>_to_<target>.json`.

## adaptive_threshold_truthful_qa.py

RQ2 harness: does a self-adaptive threshold — recalibrated on the target model's own
calibration data — preserve routing quality on held-out data where a fixed
(transferred) threshold does not? Operationalized as calibration fidelity: a
threshold at quantile q should route ~(1 - q) of held-out examples if it's
generalizing correctly. Compares that gap for the fixed gate (source's threshold
applied unmodified — same as RQ1's transferred gate) against the adaptive gate
(recalibrated on the target's own calibration split), both evaluated on the same
development split. Reuses `_common.py`'s shared model/entropy plumbing so all three
conditions (source, fixed, adaptive) run under identical data and decoding config.
Model pair and eval split come from `../configs/rq2.yaml`.

```
uv run python adaptive_threshold_truthful_qa.py
```

Writes `../results/rq2_adaptive_truthful_qa_<source>_to_<target>.json`.

## rq3_accuracy_latency_truthful_qa.py

RQ3 harness: what accuracy-latency trade-off does routing introduce? Starts the real
symbolic backend (`services/symbolic`) as a live local server, then for each
development-split example generates twice under identical greedy decoding — once
ungated, once gated (entropy monitor + gate + symbolic round-trip on routed
examples) — and asserts the two texts are byte-identical, since merge-back is
annotate-only (CLAUDE.md's design decisions) and must never change output. Reports
routing rate, per-stage latency (ungated/gated generation, gate evaluation, symbolic
round-trip when routed), and a factuality proxy score (`eval/`'s lexical-containment
placeholder — see its README; not a paper-grade judge). The symbolic round-trip uses
a fixed, always-resolvable probe triple (`../configs/rq3.yaml`) rather than one
derived from the question — there's no free-text-to-triple extractor yet, so this
measures real round-trip cost against the real backend, not a factuality check of
the question itself. `n_eval_examples` deliberately subsamples the split for CPU
tractability and is logged, not silently capped.

```
uv run python rq3_accuracy_latency_truthful_qa.py
```

Writes `../results/rq3_accuracy_latency_truthful_qa_<model_name>.json`.

## rag_baseline_truthful_qa.py

RAG comparison baseline (CLAUDE.md's Phase-0 design decisions): retrieve-then-generate
accuracy on TruthfulQA, using FAISS + all-MiniLM-L6-v2 retrieval over a committed,
pre-built Wikipedia passage corpus (`services/rag/scripts/build_corpus.py`, subset
tied to the TruthfulQA question set) instead of the entropy gate. Accuracy-only — no
latency instrumentation and no gate/symbolic interaction, deliberately, per CLAUDE.md
("a comparison baseline, not the contribution — capped effort on purpose"). For each
example in the eval split, retrieves the top-`k` passages, prepends them as context,
and generates greedily under the same model and decoding config as RQ1-RQ3. Model,
`top_k`, corpus path, and eval split come from `../configs/rag.yaml`; unlike the RQ
harnesses this runs against the **test** split, since it isn't fitting any threshold
and has nothing to leak. Reports a `factuality_accuracy_proxy` — this uses the same
lexical-containment placeholder metric as RQ3 (`eval/src/sense_eval/factuality.py`,
see its README), not a paper-grade judge, so the number is a pipeline sanity check,
not a reportable accuracy figure. Note this baseline runs on the **test** split
(327 examples) while RQ3 runs on the **development** split (20 examples,
subsampled) — the two `factuality_accuracy_proxy` numbers are not directly
comparable until both are evaluated on the same split.

```
uv run python rag_baseline_truthful_qa.py
```

Writes `../results/rag_baseline_truthful_qa_<model_name>.json`.

## Tests

```
uv run pytest
```

Runs the same wiring on a small subset (for speed) plus a leakage-refusal check,
rather than the full 327-example calibration split.
