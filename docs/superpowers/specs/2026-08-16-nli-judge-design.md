# NLI-Based Factuality Judge — Design

**Status:** approved for implementation planning.

## Problem

Every factuality number this project reports (RQ3, the RAG baseline, and
FActScore pipeline checks) currently goes through
`lexical_containment_verdict` (`eval/src/sense_eval/factuality.py`) — a
substring-containment heuristic explicitly labeled in code and in
`eval/README.md` as a placeholder: "not the project's eventual factuality
judge... any number this metric produces measures whether the pipeline
plumbing works, not paper-grade accuracy." CLAUDE.md's "Current status"
section (2026-08-16) names replacing it as the one remaining open item on the
metrics side. This design scopes that replacement.

Separately, `experiments/factscore_symbolic_verification.py` (added
2026-08-16) generates a biography per FActScore entity but never scores it —
scoring was blocked on a free-text-to-Wikidata-triple extractor for the Z3
symbolic backend, which is explicitly out of scope (tied to the undecided
"replace" merge-back policy — see that script's docstring and
`configs/rq3.yaml`). This design also closes that gap, because FActScore
factuality scoring does **not** need that extractor — it's a different,
independently solvable problem (see "Two problems, not one" below).

## Two problems, not one

It's tempting to read "extract claims from generated text" as one blocked
task. It's actually two, and only one is blocked:

1. **Structured triple extraction for Z3** (`services/symbolic`) — turn free
   text into a `(subject, predicate_pid, object)` triple matching one of
   `domain.py`'s five Wikidata-predicate relation kinds, resolvable against a
   ~10-entity fixed KB. Hard: requires reliable entity/relation/value
   extraction with near-zero tolerance for autoformalization error (a wrong
   extraction makes Z3 confidently verify the wrong claim). **Stays out of
   scope**, unchanged by this design.
2. **Atomic-claim scoring against a reference text for a judge** (this
   design) — split generated text into sentence-level claims, ask an NLI
   model "does the reference text entail this claim," aggregate a supported
   fraction. Wrong claims here just move the entailment probability, not
   silently return a false "verified" — the tri-state discretization step
   still fails toward "unknown," not a fabricated verdict. Directly matches
   the real FActScore paper's own methodology (decompose into atomic facts,
   score the supported fraction) — CLAUDE.md's design decisions already name
   this as FActScore's contribution.

## Model choice

`MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-wanli`, ~70M parameters,
purpose-trained for NLI. Chosen over `facebook/bart-large-mnli` (10x larger,
no CPU-testability benefit for this project's scale) and over building a
from-scratch classifier (real, unfinished work with no clear payoff over an
existing pinned NLI model).

Precedent: `services/rag/src/sense_rag/index.py` already uses one small real
model (`all-MiniLM-L6-v2`) directly in both production code and
`services/rag/tests/`, with no separate "tiny CPU stand-in" the way the five
LLM generation checkpoints have `cpu_test`. This design follows that same
pattern — one small model, used everywhere, real behavior in tests.

Commit SHA: pinned at implementation time (CLAUDE.md #5 — never pin `main`),
looked up from the model's actual HF repo, not fabricated in this doc.

Label ordering: NLI models don't share one canonical label order for
entailment/neutral/contradiction. Read `model.config.id2label` at load time
and map by string name, never by hardcoded index — same principle CLAUDE.md
already states for `tokenizer.vocab_size`.

## Module: `eval/src/sense_eval/nli_judge.py`

```
entailment_scores(premise: str, hypothesis: str) -> dict[str, float]
    # {"entailment": p, "neutral": p, "contradiction": p}, softmax over logits,
    # labels resolved via id2label.

split_into_atomic_claims(text: str) -> list[str]
    # Regex sentence splitter (`. ! ?` boundaries), drops empty/whitespace-only
    # fragments. Deliberately not an LLM extractor — sentence-level granularity
    # is what FActScore's own released decomposition approximates when a full
    # extraction model isn't in scope; this is a simplification, documented as
    # such in the module docstring, not presented as FActScore's actual
    # decomposition model.

nli_verdict_short_answer(
    generated_text: str, right_answer: str, hallucinated_answer: str,
    entailment_threshold: float,
) -> FactualityVerdict
    # HaluEval-shaped verdict: "correct" if entailment_scores(generated_text,
    # right_answer)["entailment"] >= threshold and the hallucinated_answer
    # comparison doesn't also clear it; "incorrect" the symmetric case;
    # "unknown" otherwise (including both/neither clearing threshold) — same
    # tri-state contract lexical_containment_verdict already has, so call
    # sites don't change shape, only semantics.

factscore_style_verdict(
    generated_text: str, reference_text: str,
    supported_threshold: float, unsupported_threshold: float,
) -> tuple[FactualityVerdict, dict]
    # Splits generated_text into atomic claims, scores each claim's entailment
    # against reference_text, computes supported_fraction = (claims with
    # entailment >= supported_threshold) / (total claims). Discretizes:
    # >= supported_threshold-on-fraction -> "correct",
    # <= unsupported_threshold-on-fraction -> "incorrect", else "unknown".
    # Returns the verdict plus a detail dict (per-claim scores,
    # supported_fraction, n_claims) for per_example logging — the aggregate
    # number alone would hide exactly the kind of nuance CLAUDE.md's
    # abstention-rate warning is about.
```

Thresholds live in `configs/nli_judge.yaml`, not as function defaults —
explicit and reviewable, not a silent magic number baked into code.

**Named limitation, stated in the module docstring and `eval/README.md`:**
these thresholds are a first cut, chosen for reasonable behavior on manual
spot checks, not calibrated against a human-labeled validation set (none
exists yet for this project). This judge is real (not a placeholder — it
does semantic entailment, not substring matching) but its threshold
calibration is unvalidated. State this plainly wherever its numbers are
reported, the same way the lexical-containment placeholder's limitation was
stated.

## Integration (full replacement)

- `eval/src/sense_eval/factuality.py`: `lexical_containment_verdict` and
  `PLACEHOLDER_FACTUALITY_METRIC_LABEL` are removed. `FactualityVerdict`,
  `summarize_factuality`, `assert_factuality_metrics_reported_together` stay
  unchanged — they're the reporting layer, agnostic to which scorer produced
  the verdicts.
- `experiments/rag_baseline_halueval.py`,
  `experiments/rq3_accuracy_latency_halueval.py`: swap
  `lexical_containment_verdict(...)` for
  `nli_verdict_short_answer(...)`. Each result's `factuality_metric` field
  changes from the placeholder label to a descriptive real-metric label
  (e.g. `"nli_entailment (MoritzLaurer/DeBERTa-v3-xsmall-... @ <sha>)"`).
- `experiments/factscore_symbolic_verification.py`: gains
  `factscore_style_verdict(generated_text, example.wikipedia_text, ...)` per
  example, reported as this script's *actual* factuality result — separate
  and clearly labeled from its existing (unchanged, still
  not-a-factuality-judgment) Z3 decomposition round-trip check on known
  entities. Both coexist in the same result file under distinct keys.
- `eval/pyproject.toml`: add `transformers` and `torch` as real dependencies
  (currently `dependencies = []`).

## Testing

`eval/tests/test_nli_judge.py`, using the real pinned model directly (no
fake stand-in, per the RAG precedent):
- Known entailment pair (e.g. "Paris is the capital of France." /
  "France's capital is Paris.") scores high entailment.
- Known contradiction pair scores low entailment / high contradiction.
- `split_into_atomic_claims` splits a multi-sentence fixture correctly,
  drops empty fragments.
- `factscore_style_verdict` on a fixture with one true and one false claim
  against a reference produces the expected mixed/"unknown" verdict and a
  `supported_fraction` of 0.5.
- `nli_verdict_short_answer` mirrors `lexical_containment_verdict`'s existing
  test cases (correct/incorrect/unknown) with semantic instead of
  lexical inputs.

Each touched experiment's existing fixture-based test
(`test_rag_baseline_halueval.py`, `test_rq3_accuracy_latency_halueval.py`,
`test_factscore_symbolic_verification.py`) updates its assertions for the
new verdict function and `factuality_metric` label, following the same
pattern already used when those tests were written.

## Out of scope

- The Z3 backend's structured triple extractor — unrelated, still blocked,
  still tied to the undecided "replace" merge-back policy.
- Threshold calibration against human-labeled ground truth — flagged as a
  named limitation, not solved here; a future task once labeled data exists.
- API-based LLM-as-judge — explicitly not chosen (see brainstorming
  session, 2026-08-16): breaks CPU-only testability and adds per-run cost
  and non-determinism for a project whose test suite must run on CPU with a
  small model.
