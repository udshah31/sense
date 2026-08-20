One script per table/figure in the paper.

## Setup

```
uv sync
```

All four harnesses below run against HaluEval with per-checkpoint calibration
splits (`../data/splits/halueval.json`), per the 2026-08-10 scope reconciliation —
see `../CLAUDE.md`. `calibrate_gate_truthful_qa.py` is the one script still on
TruthfulQA, kept deliberately (the excluded-benchmark discussion, `../CLAUDE.md`'s
design decisions) rather than migrated.

## run_gpu_experiments.sh

One-command runbook for the real five-checkpoint GPU run (CLAUDE.md's "one command
reproduces the headline result" reproducibility requirement). Not runnable on the
local CPU dev box by design — every component is separately unit-tested there
against tiny models; this script is for the GPU environment only. Fails fast on a
missing `HF_TOKEN` or unavailable CUDA rather than burning compute on a run that was
never going to produce comparable numbers, and writes a `run_metadata_<timestamp>.json`
sidecar (hardware, GPU model, driver, CUDA version, git commit) alongside each
stage's usual result files, since the per-experiment JSON files carry model
revision/quantization/decoding config but not those run-level facts.

```
HF_TOKEN=... ./run_gpu_experiments.sh          # all four stages
HF_TOKEN=... ./run_gpu_experiments.sh rq3      # one stage only
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

## transfer_threshold_halueval.py

RQ1 harness: calibrates natively on the source model, calibrates a separate native
baseline on the target model (comparison only, never applied), then applies the
source's threshold directly to the target model via `GatePolicy.set_threshold` and
compares its routing decisions against the target's own native gate. Evaluated on
the **development** split only — test stays untouched until the final reported
numbers. Each checkpoint calibrates on its own disjoint per-checkpoint calibration
subset (`../data/splits/halueval.json`), not a shared list. Loops over every pair in
`../configs/rq1.yaml`'s `transfer_pairs` (cross-family anchors + the Qwen3 scale
ladder), writing one result file per pair.

```
uv run python transfer_threshold_halueval.py
```

Writes `../results/rq1_transfer_halueval_<source>_to_<target>.json`.

## adaptive_threshold_halueval.py

RQ2 harness: does a self-adaptive threshold — recalibrated on the target model's own
calibration data — preserve routing quality on held-out data where a fixed
(transferred) threshold does not? Operationalized as calibration fidelity: a
threshold at quantile q should route ~(1 - q) of held-out examples if it's
generalizing correctly. Compares that gap for the fixed gate (source's threshold
applied unmodified — same as RQ1's transferred gate) against the adaptive gate
(recalibrated on the target's own calibration split), both evaluated on the same
development split. Same five transfer pairs as RQ1, from `../configs/rq2.yaml`,
looping and writing one result file per pair.

```
uv run python adaptive_threshold_halueval.py
```

Writes `../results/rq2_adaptive_halueval_<source>_to_<target>.json`.

## rq3_accuracy_latency_halueval.py

RQ3 harness: what accuracy-latency trade-off does routing introduce, and does it
hold across all five checkpoints? Starts the real symbolic backend (Z3 by default,
`services/symbolic`) as a live local server once for the whole run, then loops over
every checkpoint in `../configs/rq3.yaml`'s `models` list. For each
development-split example, generates twice under identical greedy decoding — once
ungated, once gated (entropy monitor + gate + symbolic round-trip on routed
examples) — and asserts the two texts are byte-identical, since merge-back is
annotate-only (CLAUDE.md's design decisions) and must never change output. Each
checkpoint calibrates its own gate on its own disjoint calibration subset, then is
loaded, run, and freed before the next checkpoint loads. Reports routing rate,
per-stage latency (ungated/gated generation, gate evaluation, symbolic round-trip
when routed), and a factuality score from `eval/`'s NLI-based judge (see its README;
real semantic entailment scoring, though its threshold calibration is a
named, unvalidated first cut) as
task_accuracy/hallucination_rate/abstention_rate together. The symbolic round-trip
uses a fixed, always-resolvable probe triple (`../configs/rq3.yaml`) rather than one
derived from the question — there's no free-text-to-triple extractor yet, so this
measures real round-trip cost against the real backend, not a factuality check of
the question itself. `n_eval_examples` deliberately subsamples the split when it's
below the available size and logs that it's doing so, never silently.

```
uv run python rq3_accuracy_latency_halueval.py
```

Writes `../results/rq3_accuracy_latency_halueval_<model_name>.json`.

## rag_baseline_halueval.py

RAG comparison baseline (CLAUDE.md's design decisions): retrieve-then-generate
accuracy on HaluEval, using FAISS + all-MiniLM-L6-v2 retrieval over a committed
passage corpus instead of the entropy gate. Accuracy-only — no latency
instrumentation and no gate/symbolic interaction, deliberately, per CLAUDE.md
("a comparison baseline, not the contribution — capped effort on purpose"). For each
example in the eval split, retrieves the top-`k` passages, prepends them as context,
and generates greedily under the same model and decoding config as RQ1-RQ3. Model,
`top_k`, corpus path, and eval split come from `../configs/rag.yaml`. Reports task_accuracy/hallucination_rate/abstention_rate scored by the real
NLI judge (`eval/src/sense_eval/nli_judge.py`, see its README) — semantic
entailment, not substring matching, though its threshold calibration is a
named, unvalidated first cut.

```
uv run python rag_baseline_halueval.py
```

Writes `../results/rag_baseline_halueval_<model_name>.json`.

## factscore_symbolic_verification.py

Not an RQ1-RQ3 harness. Exercises the decomposition front-end -> Z3 backend
round-trip (`services/symbolic`) against FActScore's biography-prompt entities
(`../data/splits/factscore.json`), generating a biography per example (same
decoding config as the other harnesses) for pipeline exercise but never scoring
it — this project has no free-text-to-triple extractor yet (out of scope, same
reason `rq3_accuracy_latency_halueval.py`'s probe triple isn't derived from the
question either). For each entity that resolves against the Z3 backend's small
fixed domain KB (`services/symbolic/src/sense_symbolic/domain.py`), it builds a
known ground-truth probe claim (e.g. Einstein's birth year) and checks the
backend verifies its own fact correctly; entities that don't resolve — most of
FActScore's 500, since the KB is deliberately small — are recorded as abstained,
not guessed at. So `task_accuracy` here means "entity resolved and its
ground-truth probe claim verified correctly," not "generated biography was
non-hallucinatory." Model and eval split come from `../configs/factscore_symbolic.yaml`.

Also reports a second, independent factuality result under `factscore_*`-prefixed
keys: every example's generated biography is scored by the NLI judge's
atomic-decomposition verdict against its own FActScore reference text
(`example.wikipedia_text`) — this project's first real FActScore factuality
number, kept separate from the Z3-round-trip keys above so the two are never
confused.

```
uv run python factscore_symbolic_verification.py
```

Writes `../results/factscore_symbolic_verification_<model_name>.json`.

## Tests

```
uv run pytest
```

Runs the same wiring on a small subset (for speed) plus a leakage-refusal check,
rather than the full 327-example calibration split.
