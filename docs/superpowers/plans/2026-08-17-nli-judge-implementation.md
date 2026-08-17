# NLI-Based Factuality Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the substring-containment placeholder factuality scorer (`lexical_containment_verdict`) with a real NLI-based semantic judge, and use it to give `experiments/factscore_symbolic_verification.py` a real factuality score for the first time.

**Architecture:** A new `eval/src/sense_eval/nli_judge.py` module wraps a small pinned NLI model (`cliang1453/deberta-v3-xsmall-mnli`) behind two verdict functions matching this project's two data shapes — short-answer QA (HaluEval) and open-ended biography generation scored against a reference (FActScore). The existing `FactualityVerdict`/`summarize_factuality`/`assert_factuality_metrics_reported_together` reporting layer in `eval/src/sense_eval/factuality.py` is untouched; only the scorer underneath it changes.

**Tech Stack:** Python 3.13, `transformers`/`torch` (already used by `experiments/` and `services/neural`, newly added to `eval/`'s own dependencies), `pytest` + `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-08-16-nli-judge-design.md`

## Global Constraints

- Model: `cliang1453/deberta-v3-xsmall-mnli`, revision `d1ca70f9ece4d8afd33015893a69df9a6e45a672` (pinned commit SHA, verified to exist and load, per CLAUDE.md #5).
- Read `model.config.id2label` at call time to map entailment/neutral/contradiction — never hardcode index-to-label (CLAUDE.md's `tokenizer.vocab_size`-not-hardcoded principle, extended here).
- `task_accuracy`/`hallucination_rate`/`abstention_rate` must always be reported together (`assert_factuality_metrics_reported_together`, unchanged) — every result this plan touches keeps going through `write_results`.
- No fabricated results: every threshold is a named, stated-as-unvalidated first cut (spec's "Named limitation" section), never presented as calibrated.
- Full replacement, not opt-in: `lexical_containment_verdict` and `PLACEHOLDER_FACTUALITY_METRIC_LABEL` are removed, not kept behind a flag.
- Follow TDD: write the failing test, watch it fail, implement, watch it pass, commit — every task.

---

### Task 1: NLI judge core — model loading + entailment scoring

**Files:**
- Create: `configs/nli_judge.yaml`
- Modify: `eval/pyproject.toml`
- Create: `eval/tests/conftest.py`
- Create: `eval/src/sense_eval/nli_judge.py`
- Test: `eval/tests/test_nli_judge.py`

**Interfaces:**
- Produces: `load_nli_model(hf_repo: str, revision: str) -> tuple[model, tokenizer]`; `entailment_scores(model, tokenizer, premise: str, hypothesis: str) -> dict[str, float]` (keys: `"entailment"`, `"neutral"`, `"contradiction"`).

- [ ] **Step 1: Add `configs/nli_judge.yaml`**

```yaml
# NLI-based factuality judge (eval/src/sense_eval/nli_judge.py), replacing
# the retired lexical-containment placeholder. See
# docs/superpowers/specs/2026-08-16-nli-judge-design.md for the full design
# and its named limitation: these thresholds are a first cut, not calibrated
# against human-labeled ground truth (none exists yet for this project).
hf_repo: cliang1453/deberta-v3-xsmall-mnli
revision: d1ca70f9ece4d8afd33015893a69df9a6e45a672

# nli_verdict_short_answer (HaluEval-shaped harnesses): entailment probability
# at or above which a generated answer counts as entailing right_answer or
# hallucinated_answer.
short_answer_entailment_threshold: 0.7

# factscore_style_verdict (FActScore-shaped harness): per-claim entailment
# threshold for "this atomic claim is supported by the reference text" ...
claim_supported_threshold: 0.7
# ... and the two thresholds the resulting supported_fraction is discretized
# against.
fraction_correct_threshold: 0.8
fraction_incorrect_threshold: 0.2
```

- [ ] **Step 2: Add `transformers`/`torch` to `eval/pyproject.toml`**

Modify the `dependencies` list (currently `dependencies = []`):

```toml
[project]
name = "sense-eval"
version = "0.1.0"
description = "Metrics: factuality + latency."
requires-python = ">=3.13.5"
dependencies = [
    "transformers>=4.46",
    "torch>=2.5",
]

[dependency-groups]
dev = [
    "pytest==8.3.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sense_eval"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Only `dependencies` changes from what's there today (adding
`transformers`/`torch`); the rest of the file — including
`[tool.pytest.ini_options]` — stays exactly as it already is.
`sense_eval`'s code and tests are entirely synchronous (no async functions
anywhere in `eval/`), unlike `experiments/`, so no `pytest-asyncio` is
needed here.

Run `cd eval && uv sync` after editing to install the new dependencies.

- [ ] **Step 3: Add a session-scoped model fixture, `eval/tests/conftest.py`**

```python
import pytest

from sense_eval.nli_judge import load_nli_model

NLI_HF_REPO = "cliang1453/deberta-v3-xsmall-mnli"
NLI_REVISION = "d1ca70f9ece4d8afd33015893a69df9a6e45a672"


@pytest.fixture(scope="session")
def nli_model_and_tokenizer():
    return load_nli_model(NLI_HF_REPO, NLI_REVISION)
```

- [ ] **Step 4: Write the failing tests, `eval/tests/test_nli_judge.py`**

```python
import pytest

from sense_eval.nli_judge import entailment_scores, load_nli_model


def test_load_nli_model_requires_pinned_revision():
    with pytest.raises(ValueError, match="revision"):
        load_nli_model("cliang1453/deberta-v3-xsmall-mnli", "")


def test_entailment_scores_true_pair_scores_high_entailment(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    scores = entailment_scores(
        model, tokenizer, "Paris is the capital of France.", "France's capital is Paris."
    )

    assert set(scores.keys()) == {"entailment", "neutral", "contradiction"}
    assert scores["entailment"] > 0.9
    assert abs(sum(scores.values()) - 1.0) < 1e-4


def test_entailment_scores_false_pair_scores_high_contradiction(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    scores = entailment_scores(
        model, tokenizer, "Paris is the capital of France.", "Berlin is the capital of France."
    )

    assert scores["contradiction"] > 0.9
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sense_eval.nli_judge'`

- [ ] **Step 6: Implement `eval/src/sense_eval/nli_judge.py`**

```python
"""NLI-based factuality judge: replaces the substring-containment placeholder
(eval/src/sense_eval/factuality.py's former lexical_containment_verdict) with
real semantic entailment scoring, using a small pinned NLI model.

Two verdict shapes for the two data shapes this project scores:
- nli_verdict_short_answer: HaluEval-style short-answer QA, where a generated
  answer is checked against a right_answer and a hallucinated_answer.
- factscore_style_verdict: FActScore-style open-ended biography generation,
  decomposed into atomic (sentence-level) claims and scored against a
  reference text, matching the real FActScore paper's own methodology
  (decompose into atomic facts, score the supported fraction).

Named limitation (see docs/superpowers/specs/2026-08-16-nli-judge-design.md):
the thresholds this module's callers use are a first cut, chosen for
reasonable behavior on manual spot checks, not calibrated against a
human-labeled validation set — none exists yet for this project. This judge
is real (semantic entailment, not substring matching) but its threshold
calibration is unvalidated. State this wherever its numbers are reported.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_METRIC_LABEL_TEMPLATE = "nli_entailment ({hf_repo} @ {revision})"


def load_nli_model(hf_repo: str, revision: str):
    """Loads the NLI model + tokenizer. `revision` is required (not
    defaulted) — CLAUDE.md #5, pin the commit SHA, not just the repo name."""
    if not revision:
        raise ValueError(f"NLI model '{hf_repo}' has no pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo, revision=revision)
    model.eval()
    return model, tokenizer


def entailment_scores(model, tokenizer, premise: str, hypothesis: str) -> dict[str, float]:
    """Softmax probabilities over the model's own labels for "does `premise`
    entail `hypothesis`", keyed by label name (not index — id2label read at
    call time, since label ordering isn't standardized across NLI models)."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {model.config.id2label[i]: probs[i].item() for i in range(len(probs))}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Commit**

```bash
git add configs/nli_judge.yaml eval/pyproject.toml eval/tests/conftest.py eval/src/sense_eval/nli_judge.py eval/tests/test_nli_judge.py
git commit -m "Add NLI judge core: pinned model loading + entailment scoring"
```

---

### Task 2: Atomic-claim sentence splitter

**Files:**
- Modify: `eval/src/sense_eval/nli_judge.py`
- Test: `eval/tests/test_nli_judge.py`

**Interfaces:**
- Consumes: nothing from Task 1's functions directly.
- Produces: `split_into_atomic_claims(text: str) -> list[str]`.

- [ ] **Step 1: Write the failing tests**

Append to `eval/tests/test_nli_judge.py`:

```python
from sense_eval.nli_judge import split_into_atomic_claims


def test_split_into_atomic_claims_splits_on_sentence_boundaries():
    claims = split_into_atomic_claims(
        "Albert Einstein was a physicist. He was born in 1879. He developed relativity."
    )

    assert claims == [
        "Albert Einstein was a physicist.",
        "He was born in 1879.",
        "He developed relativity.",
    ]


def test_split_into_atomic_claims_drops_empty_fragments():
    claims = split_into_atomic_claims("One sentence.   \n\n  Another sentence.")

    assert claims == ["One sentence.", "Another sentence."]


def test_split_into_atomic_claims_handles_empty_text():
    assert split_into_atomic_claims("") == []
    assert split_into_atomic_claims("   ") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -k split_into_atomic_claims -v`
Expected: FAIL with `ImportError: cannot import name 'split_into_atomic_claims'`

- [ ] **Step 3: Implement `split_into_atomic_claims`**

Add to `eval/src/sense_eval/nli_judge.py` (near the top, with the other imports):

```python
import re
```

Append the function:

```python
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_into_atomic_claims(text: str) -> list[str]:
    """A simple sentence-level splitter, not an LLM extractor (see the design
    doc's "Two problems, not one" section) — splits on sentence-ending
    punctuation followed by whitespace, drops empty/whitespace-only
    fragments. This is a deliberate simplification of FActScore's real
    atomic-fact decomposition, not a claim to reproduce it exactly."""
    stripped = text.strip()
    if not stripped:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(stripped) if s.strip()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -k split_into_atomic_claims -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/src/sense_eval/nli_judge.py eval/tests/test_nli_judge.py
git commit -m "Add atomic-claim sentence splitter to the NLI judge"
```

---

### Task 3: Short-answer verdict (HaluEval shape)

**Files:**
- Modify: `eval/src/sense_eval/nli_judge.py`
- Test: `eval/tests/test_nli_judge.py`

**Interfaces:**
- Consumes: `entailment_scores(model, tokenizer, premise, hypothesis) -> dict[str, float]` (Task 1); `FactualityVerdict` from `eval/src/sense_eval/factuality.py` (existing, unchanged — `@dataclass(frozen=True)` with a single `label: str` field).
- Produces: `nli_verdict_short_answer(model, tokenizer, generated_text: str, right_answer: str, hallucinated_answer: str, entailment_threshold: float) -> FactualityVerdict`.

- [ ] **Step 1: Write the failing tests**

Append to `eval/tests/test_nli_judge.py`:

```python
from sense_eval.nli_judge import nli_verdict_short_answer


def test_nli_verdict_short_answer_correct_when_generated_entails_right_answer(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict = nli_verdict_short_answer(
        model,
        tokenizer,
        generated_text="The watermelon seeds pass through your digestive system without effect.",
        right_answer="Nothing happens, the seeds just pass through your digestive system.",
        hallucinated_answer="You die if you eat watermelon seeds.",
        entailment_threshold=0.7,
    )

    assert verdict.label == "correct"


def test_nli_verdict_short_answer_incorrect_when_generated_entails_hallucinated_answer(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict = nli_verdict_short_answer(
        model,
        tokenizer,
        generated_text="You die if you eat watermelon seeds.",
        right_answer="Nothing happens, the seeds just pass through your digestive system.",
        hallucinated_answer="You die if you eat watermelon seeds.",
        entailment_threshold=0.7,
    )

    assert verdict.label == "incorrect"


def test_nli_verdict_short_answer_unknown_when_neither_entailed(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict = nli_verdict_short_answer(
        model,
        tokenizer,
        generated_text="Watermelons are a delicious summer fruit.",
        right_answer="Nothing happens, the seeds just pass through your digestive system.",
        hallucinated_answer="You die if you eat watermelon seeds.",
        entailment_threshold=0.7,
    )

    assert verdict.label == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -k nli_verdict_short_answer -v`
Expected: FAIL with `ImportError: cannot import name 'nli_verdict_short_answer'`

- [ ] **Step 3: Implement `nli_verdict_short_answer`**

Add to `eval/src/sense_eval/nli_judge.py`:

```python
from sense_eval.factuality import FactualityVerdict


def nli_verdict_short_answer(
    model,
    tokenizer,
    generated_text: str,
    right_answer: str,
    hallucinated_answer: str,
    entailment_threshold: float,
) -> FactualityVerdict:
    """"correct" if the generated text entails the right answer (and doesn't
    also entail the hallucinated one at/above threshold), "incorrect" the
    symmetric case, "unknown" otherwise (including both or neither clearing
    threshold) — same tri-state contract the retired lexical_containment_verdict
    had, so call sites don't change shape, only semantics."""
    right_entailment = entailment_scores(model, tokenizer, generated_text, right_answer)["entailment"]
    wrong_entailment = entailment_scores(model, tokenizer, generated_text, hallucinated_answer)["entailment"]

    right_clears = right_entailment >= entailment_threshold
    wrong_clears = wrong_entailment >= entailment_threshold

    if right_clears and not wrong_clears:
        return FactualityVerdict(label="correct")
    if wrong_clears and not right_clears:
        return FactualityVerdict(label="incorrect")
    return FactualityVerdict(label="unknown")
```

Place the `from sense_eval.factuality import FactualityVerdict` line with the
module's other imports at the top of the file, not inline.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -k nli_verdict_short_answer -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full eval test suite**

Run: `cd eval && uv run pytest -v`
Expected: all tests pass (Task 1, 2, 3's new tests plus the existing `test_factuality.py` tests, untouched so far)

- [ ] **Step 6: Commit**

```bash
git add eval/src/sense_eval/nli_judge.py eval/tests/test_nli_judge.py
git commit -m "Add short-answer NLI verdict for HaluEval-shaped harnesses"
```

---

### Task 4: FActScore-style verdict (atomic decomposition + aggregation)

**Files:**
- Modify: `eval/src/sense_eval/nli_judge.py`
- Test: `eval/tests/test_nli_judge.py`

**Interfaces:**
- Consumes: `split_into_atomic_claims` (Task 2), `entailment_scores` (Task 1), `FactualityVerdict` (existing).
- Produces: `FActScoreVerdictDetail` (frozen dataclass: `claims: tuple[str, ...]`, `claim_entailment_scores: tuple[float, ...]`, `supported_fraction: float | None`); `factscore_style_verdict(model, tokenizer, generated_text: str, reference_text: str, claim_supported_threshold: float, fraction_correct_threshold: float, fraction_incorrect_threshold: float) -> tuple[FactualityVerdict, FActScoreVerdictDetail]`.

- [ ] **Step 1: Write the failing tests**

Append to `eval/tests/test_nli_judge.py`:

```python
from sense_eval.nli_judge import factscore_style_verdict


def test_factscore_style_verdict_correct_when_all_claims_supported(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict, detail = factscore_style_verdict(
        model,
        tokenizer,
        generated_text="Albert Einstein was a theoretical physicist. He was born in Germany.",
        reference_text=(
            "Albert Einstein was a German-born theoretical physicist, widely "
            "acknowledged to be one of the greatest physicists of all time."
        ),
        claim_supported_threshold=0.5,
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )

    assert verdict.label == "correct"
    assert detail.supported_fraction == 1.0
    assert len(detail.claims) == 2
    assert len(detail.claim_entailment_scores) == 2


def test_factscore_style_verdict_incorrect_when_no_claims_supported(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict, detail = factscore_style_verdict(
        model,
        tokenizer,
        generated_text="Albert Einstein was a professional basketball player.",
        reference_text=(
            "Albert Einstein was a German-born theoretical physicist, widely "
            "acknowledged to be one of the greatest physicists of all time."
        ),
        claim_supported_threshold=0.5,
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )

    assert verdict.label == "incorrect"
    assert detail.supported_fraction == 0.0


def test_factscore_style_verdict_unknown_when_no_claims_to_score(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict, detail = factscore_style_verdict(
        model,
        tokenizer,
        generated_text="   ",
        reference_text="Albert Einstein was a German-born theoretical physicist.",
        claim_supported_threshold=0.5,
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )

    assert verdict.label == "unknown"
    assert detail.claims == ()
    assert detail.supported_fraction is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -k factscore_style_verdict -v`
Expected: FAIL with `ImportError: cannot import name 'factscore_style_verdict'`

- [ ] **Step 3: Implement `factscore_style_verdict`**

Add to `eval/src/sense_eval/nli_judge.py` (add `from dataclasses import dataclass` to the top-level imports):

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FActScoreVerdictDetail:
    claims: tuple[str, ...]
    claim_entailment_scores: tuple[float, ...]
    supported_fraction: float | None


def factscore_style_verdict(
    model,
    tokenizer,
    generated_text: str,
    reference_text: str,
    claim_supported_threshold: float,
    fraction_correct_threshold: float,
    fraction_incorrect_threshold: float,
) -> tuple[FactualityVerdict, FActScoreVerdictDetail]:
    """Splits `generated_text` into atomic claims, scores each claim's
    entailment against `reference_text` (does the reference support this
    claim), computes the supported fraction, and discretizes it into a
    tri-state verdict: "correct" if supported_fraction >=
    fraction_correct_threshold, "incorrect" if supported_fraction <=
    fraction_incorrect_threshold, "unknown" otherwise (including when there
    are no claims to score at all). Returns the verdict plus a detail record
    for per-example logging — the aggregate number alone would hide exactly
    the kind of nuance CLAUDE.md's abstention-rate warning is about."""
    claims = split_into_atomic_claims(generated_text)
    if not claims:
        return FactualityVerdict(label="unknown"), FActScoreVerdictDetail(
            claims=(), claim_entailment_scores=(), supported_fraction=None
        )

    claim_scores = tuple(
        entailment_scores(model, tokenizer, reference_text, claim)["entailment"] for claim in claims
    )
    n_supported = sum(1 for score in claim_scores if score >= claim_supported_threshold)
    supported_fraction = n_supported / len(claims)

    if supported_fraction >= fraction_correct_threshold:
        label = "correct"
    elif supported_fraction <= fraction_incorrect_threshold:
        label = "incorrect"
    else:
        label = "unknown"

    detail = FActScoreVerdictDetail(
        claims=tuple(claims), claim_entailment_scores=claim_scores, supported_fraction=supported_fraction
    )
    return FactualityVerdict(label=label), detail
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd eval && uv run pytest tests/test_nli_judge.py -k factscore_style_verdict -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full eval test suite**

Run: `cd eval && uv run pytest -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add eval/src/sense_eval/nli_judge.py eval/tests/test_nli_judge.py
git commit -m "Add FActScore-style atomic-decomposition NLI verdict"
```

---

### Task 5: Retire the lexical-containment placeholder

**Files:**
- Modify: `eval/src/sense_eval/factuality.py`
- Modify: `eval/tests/test_factuality.py`
- Modify: `eval/README.md`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — this task only removes `lexical_containment_verdict` and `PLACEHOLDER_FACTUALITY_METRIC_LABEL`. `FactualityVerdict`, `summarize_factuality`, `assert_factuality_metrics_reported_together`, `REQUIRED_TOGETHER_KEYS`, `IncompleteFactualityReportError` are unchanged and still exported from `sense_eval.factuality`.

- [ ] **Step 1: Remove the placeholder scorer from `eval/src/sense_eval/factuality.py`**

Delete these from the file (they currently sit between the module docstring
and the `FactualityVerdict` dataclass, and just above it):

```python
# No fine-tuned judge model or LLM-as-judge API is wired into this project yet — that
# is real, unfinished work, not a design choice, so lexical_containment_verdict below
# stays the only scorer for now. Every result produced with it MUST carry this label
# verbatim in its "factuality_metric" field (write_results checks for the
# "placeholder" substring and prints a loud warning if it's missing) so a
# pipeline-mechanics number can never quietly read as a paper-grade one.
PLACEHOLDER_FACTUALITY_METRIC_LABEL = "lexical_containment_placeholder (NOT a judge — see eval/README.md)"
```

and the entire `lexical_containment_verdict` function (from its `def` line
through its final `return FactualityVerdict(label="unknown")`).

Update the module docstring's opening line — it currently reads:

```python
"""Factuality metrics: a placeholder verdict scorer, and the reporting layer that
turns per-example verdicts into the three numbers CLAUDE.md/the proposal require to
always be reported together — task accuracy, hallucination rate, and abstention
rate. Reporting hallucination rate alone is misleading: abstaining trivially drives
it toward zero at the cost of accuracy, so a hallucination-reduction number with no
accuracy or abstention figure next to it hides exactly the trade-off this project
studies.
"""
```

Change it to:

```python
"""Factuality metrics: the reporting layer that turns per-example verdicts into
the three numbers CLAUDE.md/the proposal require to always be reported together —
task accuracy, hallucination rate, and abstention rate. Reporting hallucination
rate alone is misleading: abstaining trivially drives it toward zero at the cost
of accuracy, so a hallucination-reduction number with no accuracy or abstention
figure next to it hides exactly the trade-off this project studies.

The verdict scorers that produce FactualityVerdict instances live in
sense_eval.nli_judge (an NLI-based semantic judge) — this module is agnostic to
which scorer produced them.
"""
```

- [ ] **Step 2: Remove the placeholder's tests from `eval/tests/test_factuality.py`**

Delete the `lexical_containment_verdict` import from the `from sense_eval.factuality
import (...)` block, and delete these five test functions:
`test_matches_best_answer`, `test_matches_incorrect_answer`,
`test_matches_neither_is_unknown`, `test_correct_takes_priority_when_both_present`,
`test_case_insensitive`.

Everything else in that file (`summarize_factuality`/
`assert_factuality_metrics_reported_together` tests) stays unchanged.

- [ ] **Step 3: Run the full eval test suite**

Run: `cd eval && uv run pytest -v`
Expected: all tests pass (the five deleted tests are gone, everything else
still passes)

- [ ] **Step 4: Update `eval/README.md`**

Replace the `## factuality.py` section's first paragraph (the one describing
`lexical_containment_verdict` as a placeholder) with:

```markdown
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
```

Leave the paragraph starting with `` `summarize_factuality` and
`assert_factuality_metrics_reported_together` are the reporting layer... ``
as-is (it's still accurate — that layer doesn't care which scorer produced
the verdicts).

- [ ] **Step 5: Commit**

```bash
git add eval/src/sense_eval/factuality.py eval/tests/test_factuality.py eval/README.md
git commit -m "Retire the lexical-containment placeholder factuality scorer"
```

---

### Task 6: Swap the RAG baseline harness to the NLI judge

**Files:**
- Modify: `experiments/rag_baseline_halueval.py`
- Modify: `experiments/tests/test_rag_baseline_halueval.py`

**Interfaces:**
- Consumes: `load_nli_model`, `nli_verdict_short_answer`, `NLI_METRIC_LABEL_TEMPLATE` from `sense_eval.nli_judge` (Tasks 1, 3).

- [ ] **Step 1: Update the failing test first — `experiments/tests/test_rag_baseline_halueval.py`**

Change the `config` dict in
`test_run_experiment_end_to_end_on_fixture_corpus` and
`test_run_experiment_does_not_crash_on_overlong_prompt` to add an
`"nli_judge"` key, e.g.:

```python
    config = {
        "models": {"cpu_test": {"hf_repo": "sshleifer/tiny-gpt2", "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"}},
        "gate": {"decoding": {"do_sample": False, "max_new_tokens": 5}},
        "rag": {"model": "cpu_test", "top_k": 1, "corpus_path": str(corpus_path), "eval_split": "test", "n_eval_examples": 2},
        "nli_judge": {
            "hf_repo": "cliang1453/deberta-v3-xsmall-mnli",
            "revision": "d1ca70f9ece4d8afd33015893a69df9a6e45a672",
            "short_answer_entailment_threshold": 0.7,
        },
    }
```

(apply the same addition to the second test's config). Both tests' remaining
assertions stay as they are — the RAG baseline test doesn't assert specific
factuality labels today, only that the numbers are in range, so this change
is purely about supplying the config the harness now requires.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd experiments && uv run pytest tests/test_rag_baseline_halueval.py -v`
Expected: FAIL — `KeyError: 'nli_judge'` (the harness doesn't read this key
yet) or an `ImportError` once Step 3 below starts, depending on run order;
either way, confirm it's failing before implementing.

- [ ] **Step 3: Update `experiments/rag_baseline_halueval.py`**

Change the import block:

```python
from _common import REPO_ROOT, load_model, load_model_registry, load_yaml_config, write_results
from _common import load_halueval_examples_and_splits as load_examples_and_splits
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, load_nli_model, nli_verdict_short_answer
from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve
```

Change `load_config`:

```python
def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rag": load_yaml_config("rag.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }
```

In `run_experiment`, after `model, tokenizer = load_model(model_cfg)`, add:

```python
    nli_model, nli_tokenizer = load_nli_model(config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"])
```

Replace the verdict computation:

```python
        verdict = lexical_containment_verdict(
            generated_text, example.right_answer, (), (example.hallucinated_answer,)
        )
```

with:

```python
        verdict = nli_verdict_short_answer(
            nli_model,
            nli_tokenizer,
            generated_text,
            example.right_answer,
            example.hallucinated_answer,
            config["nli_judge"]["short_answer_entailment_threshold"],
        )
```

Replace the return dict's `"factuality_metric"` line:

```python
        "factuality_metric": PLACEHOLDER_FACTUALITY_METRIC_LABEL,
```

with:

```python
        "factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
            hf_repo=config["nli_judge"]["hf_repo"], revision=config["nli_judge"]["revision"]
        ),
```

Also update the module docstring's second paragraph, which currently reads:

```python
Retrieval and generation never touch the network at run time: the corpus is
pre-built and committed, and the embedding model is loaded from the local HF
cache after its first download. Same decoding config and model (tiny-gpt2) as
RQ1-RQ3, so this baseline's number sits in the same honest "pipeline-mechanics
validation, not a scientific finding" category until Llama-3/Mistral GPU runs.
```

Append one sentence: `` Factuality is scored by the NLI judge
(`sense_eval.nli_judge`), not the retired lexical-containment placeholder —
see `eval/README.md`. ``

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd experiments && uv run pytest tests/test_rag_baseline_halueval.py -v`
Expected: PASS

- [ ] **Step 5: Run the full experiments test suite**

Run: `cd experiments && uv run pytest -v`
Expected: all tests pass except `test_rq3_accuracy_latency_halueval.py` and
`test_factscore_symbolic_verification.py`, which still reference the retired
scorer until Tasks 7 and 8 — confirm those are the *only* remaining
failures, and that the failure is the expected `ImportError`/`AttributeError`
for `lexical_containment_verdict`, not something new.

- [ ] **Step 6: Commit**

```bash
git add experiments/rag_baseline_halueval.py experiments/tests/test_rag_baseline_halueval.py
git commit -m "Swap RAG baseline harness to the NLI factuality judge"
```

---

### Task 7: Swap the RQ3 harness to the NLI judge

**Files:**
- Modify: `experiments/rq3_accuracy_latency_halueval.py`
- Modify: `experiments/tests/test_rq3_accuracy_latency_halueval.py`

**Interfaces:**
- Consumes: `load_nli_model`, `nli_verdict_short_answer`, `NLI_METRIC_LABEL_TEMPLATE` from `sense_eval.nli_judge` (Tasks 1, 3).

- [ ] **Step 1: Update the failing test first — `experiments/tests/test_rq3_accuracy_latency_halueval.py`**

Replace the import:

```python
from sense_eval.factuality import lexical_containment_verdict
```

with:

```python
from sense_eval.nli_judge import load_nli_model, nli_verdict_short_answer
```

Replace `test_factuality_verdict_is_computed_on_ungated_text`:

```python
def test_factuality_verdict_is_computed_on_ungated_text():
    examples = load_halueval()
    example = examples[0]
    verdict = lexical_containment_verdict(example.right_answer, example.right_answer, (), (example.hallucinated_answer,))
    assert verdict.label == "correct"
```

with:

```python
def test_factuality_verdict_is_computed_on_ungated_text():
    examples = load_halueval()
    example = examples[0]
    nli_model, nli_tokenizer = load_nli_model(
        "cliang1453/deberta-v3-xsmall-mnli", "d1ca70f9ece4d8afd33015893a69df9a6e45a672"
    )
    verdict = nli_verdict_short_answer(
        nli_model, nli_tokenizer, example.right_answer, example.right_answer, example.hallucinated_answer, 0.7
    )
    assert verdict.label == "correct"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd experiments && uv run pytest tests/test_rq3_accuracy_latency_halueval.py -k factuality_verdict -v`
Expected: FAIL with `ImportError: cannot import name 'load_nli_model' from 'sense_eval.nli_judge'`
if Task 1-4 weren't run in this environment yet, or (if they were) FAIL only
because the harness itself (Step 3 below) hasn't been updated — confirm
which, then proceed.

- [ ] **Step 3: Update `experiments/rq3_accuracy_latency_halueval.py`**

Change the import block:

```python
from _common import build_generation_inputs, load_halueval_examples_and_splits, load_model, load_model_registry, load_yaml_config, mean_calibration_entropy, write_results
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, load_nli_model, nli_verdict_short_answer
from sense_neural.entropy import TokenEntropyMonitor
from sense_neural.latency import generate_with_latency
from sense_orchestrator.gate import GatePolicy
from sense_orchestrator.router import route_and_annotate
```

Change `load_config`:

```python
def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq3": load_yaml_config("rq3.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }
```

`run_experiment_for_model`'s signature and the NLI model load: this function
is called once per checkpoint, but the NLI model is tiny and shared across
all checkpoints, so load it once in `run_all` and pass it down rather than
reloading per checkpoint. Change `run_experiment_for_model`'s signature:

```python
async def run_experiment_for_model(config: dict, model_name: str, examples, splits, nli_model, nli_tokenizer) -> dict:
```

Replace the verdict computation:

```python
            verdict = lexical_containment_verdict(
                ungated["text"], example.right_answer, (), (example.hallucinated_answer,)
            )
```

with:

```python
            verdict = nli_verdict_short_answer(
                nli_model,
                nli_tokenizer,
                ungated["text"],
                example.right_answer,
                example.hallucinated_answer,
                config["nli_judge"]["short_answer_entailment_threshold"],
            )
```

Replace the return dict's `"factuality_metric"` line:

```python
            "factuality_metric": PLACEHOLDER_FACTUALITY_METRIC_LABEL,
```

with:

```python
            "factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
                hf_repo=config["nli_judge"]["hf_repo"], revision=config["nli_judge"]["revision"]
            ),
```

Update `run_all` to load the NLI model once and pass it down:

```python
async def run_all(config: dict) -> list[dict]:
    examples, splits = load_halueval_examples_and_splits()
    nli_model, nli_tokenizer = load_nli_model(config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"])
    results = []
    for model_name in config["rq3"]["models"]:
        result = await run_experiment_for_model(config, model_name, examples, splits, nli_model, nli_tokenizer)
        write_results(
            f"rq3_accuracy_latency_halueval_{model_name}.json",
            result,
            print_exclude_keys=frozenset({"per_example"}),
        )
        results.append(result)
    return results
```

Also update the module docstring's paragraph currently reading:

```python
Factuality is scored with eval/'s lexical-containment proxy — a placeholder, not a
paper-grade judge (see eval/README.md). Reported as task_accuracy/hallucination_rate/
abstention_rate together (never a subset — write_results enforces this), even
though abstention_rate is always 0 here: merge-back is annotate-only for now, so
nothing in this harness can abstain.
```

to:

```python
Factuality is scored with eval/'s NLI-based judge (see eval/README.md; its
threshold calibration is a named, unvalidated first cut — not a paper-grade
judge yet, but real semantic entailment, not substring matching). Reported
as task_accuracy/hallucination_rate/abstention_rate together (never a subset
— write_results enforces this), even though abstention_rate is always 0
here: merge-back is annotate-only for now, so nothing in this harness can
abstain.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd experiments && uv run pytest tests/test_rq3_accuracy_latency_halueval.py -v`
Expected: PASS

- [ ] **Step 5: Run the full experiments test suite**

Run: `cd experiments && uv run pytest -v`
Expected: all tests pass except `test_factscore_symbolic_verification.py`
(Task 8 handles it) — confirm no other regressions.

- [ ] **Step 6: Commit**

```bash
git add experiments/rq3_accuracy_latency_halueval.py experiments/tests/test_rq3_accuracy_latency_halueval.py
git commit -m "Swap RQ3 harness to the NLI factuality judge"
```

---

### Task 8: Score FActScore biographies with the real judge

**Files:**
- Modify: `experiments/factscore_symbolic_verification.py`
- Modify: `experiments/tests/test_factscore_symbolic_verification.py`

**Interfaces:**
- Consumes: `load_nli_model`, `factscore_style_verdict` from `sense_eval.nli_judge` (Tasks 1, 4).
- Produces: the per-example dict gains `"factscore_factuality"` (str, the NLI-judge verdict label) and `"factscore_supported_fraction"` (float or None). The top-level result dict gains `"factscore_task_accuracy"`, `"factscore_hallucination_rate"`, `"factscore_abstention_rate"`, `"factscore_factuality_report"` — kept **separate** from the existing top-level `task_accuracy`/`hallucination_rate`/`abstention_rate` keys, which stay as the Z3-round-trip-against-known-entities numbers this script already reports. Both trios go through `write_results`'s enforcement independently since they're distinct key sets, but only one (`task_accuracy`/`hallucination_rate`/`abstention_rate`) is checked by `assert_factuality_metrics_reported_together` today — see Step 3's note on why that's correct.

- [ ] **Step 1: Update the failing test first**

Replace the whole contents of
`experiments/tests/test_factscore_symbolic_verification.py` with:

```python
import pytest

from factscore_symbolic_verification import build_probe_claim, run_experiment
from sense_data.factscore import FActScoreExample

TINY_GPT2 = {"hf_repo": "sshleifer/tiny-gpt2", "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"}
NLI_MODEL_CFG = {
    "hf_repo": "cliang1453/deberta-v3-xsmall-mnli",
    "revision": "d1ca70f9ece4d8afd33015893a69df9a6e45a672",
    "claim_supported_threshold": 0.5,
    "fraction_correct_threshold": 0.8,
    "fraction_incorrect_threshold": 0.2,
}


def test_build_probe_claim_uses_birth_year_when_known():
    predicate_pid, object_label = build_probe_claim("albert einstein")

    assert predicate_pid == "P569"
    assert object_label == "1879"


def test_build_probe_claim_falls_back_to_nationality_without_birth_year():
    with pytest.raises(KeyError):
        build_probe_claim("not a real entity")


@pytest.mark.asyncio
async def test_run_experiment_verifies_known_entity_as_correct():
    examples = [
        FActScoreExample(
            index=0,
            entity="Albert Einstein",
            one_fact_prompt="Tell me a fact about Albert Einstein.",
            factscore_prompt="Tell me about Albert Einstein.",
            hundredw_prompt="Write 100 words about Albert Einstein.",
            around_100="",
            wikipedia_text="Albert Einstein was a theoretical physicist.",
        ),
        FActScoreExample(
            index=1,
            entity="Some Unresolvable Person",
            one_fact_prompt="Tell me a fact about Some Unresolvable Person.",
            factscore_prompt="Tell me about Some Unresolvable Person.",
            hundredw_prompt="Write 100 words about Some Unresolvable Person.",
            around_100="",
            wikipedia_text="Some Unresolvable Person was a fictional test fixture.",
        ),
    ]

    config = {
        "models": {"cpu_test": TINY_GPT2},
        "factscore_symbolic": {
            "model": "cpu_test",
            "decoding": {"do_sample": False, "max_new_tokens": 5},
        },
        "nli_judge": NLI_MODEL_CFG,
    }

    result = await run_experiment(config, examples, list(range(len(examples))))

    assert result["dataset"] == "factscore"
    assert result["n_eval_examples"] == 2
    # Z3 round-trip numbers (against known entities only) — unchanged behavior.
    assert result["factuality_report"]["n_correct"] == 1
    assert result["factuality_report"]["n_abstained"] == 1
    assert result["task_accuracy"] == 0.5
    assert result["hallucination_rate"] == 0.0
    assert result["abstention_rate"] == 0.5

    # NLI-judge numbers (against every example's own reference text) — new.
    assert "factscore_task_accuracy" in result
    assert "factscore_hallucination_rate" in result
    assert "factscore_abstention_rate" in result
    assert result["factscore_factuality_report"]["n_examples"] == 2

    resolved_record = next(r for r in result["per_example"] if r["entity"] == "Albert Einstein")
    assert resolved_record["resolved"] is True
    assert resolved_record["factuality"] == "correct"
    assert "factscore_factuality" in resolved_record
    assert "factscore_supported_fraction" in resolved_record

    unresolved_record = next(r for r in result["per_example"] if r["entity"] == "Some Unresolvable Person")
    assert unresolved_record["resolved"] is False
    assert unresolved_record["factuality"] is None
    # NLI scoring runs regardless of Z3 entity resolution — it's independent.
    assert "factscore_factuality" in unresolved_record
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd experiments && uv run pytest tests/test_factscore_symbolic_verification.py -v`
Expected: FAIL — `KeyError: 'nli_judge'` or `AssertionError` on the new
`factscore_*` keys, since the harness doesn't produce them yet.

- [ ] **Step 3: Update `experiments/factscore_symbolic_verification.py`**

Change the import block:

```python
from _common import build_generation_inputs, load_factscore_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, factscore_style_verdict, load_nli_model
from sense_symbolic.decomposition import decompose_claim
from sense_symbolic.domain import facts_for, resolve_entity
from sense_symbolic.z3_verifier import verify_claim
```

Change `load_config`:

```python
def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "factscore_symbolic": load_yaml_config("factscore_symbolic.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }
```

In `run_experiment`, after `model, tokenizer = load_model(model_cfg)`, add:

```python
    nli_model, nli_tokenizer = load_nli_model(config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"])
```

Inside the per-example loop, right after `generated_text = tokenizer.decode(...)`
and **before** the `resolved_key = resolve_entity(example.entity)` branch, add
the NLI scoring (this runs for every example, independent of whether the
entity resolves against the Z3 domain KB — the two checks are unrelated):

```python
        factscore_verdict, factscore_detail = factscore_style_verdict(
            nli_model,
            nli_tokenizer,
            generated_text,
            example.wikipedia_text,
            config["nli_judge"]["claim_supported_threshold"],
            config["nli_judge"]["fraction_correct_threshold"],
            config["nli_judge"]["fraction_incorrect_threshold"],
        )
```

Then add `"factscore_factuality": factscore_verdict.label,` and
`"factscore_supported_fraction": factscore_detail.supported_fraction,` to
**both** `per_example.append({...})` call sites (the `resolved_key is None`
early-continue branch, and the final branch) — every example gets these two
keys regardless of Z3 resolution. E.g. the unresolved branch becomes:

```python
        if resolved_key is None:
            per_example.append(
                {
                    "index": index_i,
                    "entity": example.entity,
                    "generated_text": generated_text,
                    "resolved": False,
                    "factuality": None,
                    "factscore_factuality": factscore_verdict.label,
                    "factscore_supported_fraction": factscore_detail.supported_fraction,
                }
            )
            continue
```

and the resolved branch becomes:

```python
        per_example.append(
            {
                "index": index_i,
                "entity": example.entity,
                "generated_text": generated_text,
                "resolved": True,
                "predicate_pid": predicate_pid,
                "probe_object_label": probe_object_label,
                "factuality": label,
                "factscore_factuality": factscore_verdict.label,
                "factscore_supported_fraction": factscore_detail.supported_fraction,
            }
        )
```

After the existing `factuality_report = summarize_factuality(verdicts, n_abstained=n_abstained)`
line, add the second, independent rollup:

```python
    factscore_verdicts = [FactualityVerdict(label=r["factscore_factuality"]) for r in per_example]
    factscore_factuality_report = summarize_factuality(factscore_verdicts, n_abstained=0)
```

(`n_abstained=0` here is deliberate and explicit: the NLI judge always
produces a label — `"unknown"` when there are no claims to score, never a
missing verdict — so nothing in this rollup is an abstention in the sense
`summarize_factuality` means; that's a distinct concept from the Z3 branch's
`n_abstained`, which counts entities that never got a Z3 claim built at all.)

Update the return dict — keep the existing `task_accuracy`/
`hallucination_rate`/`abstention_rate`/`factuality_report` keys exactly as
they are (the Z3 round-trip numbers), and add the new ones alongside:

```python
        "factscore_task_accuracy": factscore_factuality_report["task_accuracy"],
        "factscore_hallucination_rate": factscore_factuality_report["hallucination_rate"],
        "factscore_abstention_rate": factscore_factuality_report["abstention_rate"],
        "factscore_factuality_report": factscore_factuality_report,
        "factscore_factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
            hf_repo=config["nli_judge"]["hf_repo"], revision=config["nli_judge"]["revision"]
        ),
```

Add these lines right after the existing `"factuality_metric": (...)` entry,
before `"per_example": per_example,`.

Finally, update the module docstring. Replace the paragraph:

```python
This validates decomposition+Z3 pipeline mechanics end-to-end on real FActScore
data, not the factual accuracy of the generated biography — task_accuracy here
means "fraction of examples whose entity resolved and whose ground-truth probe
claim the backend verified correctly," not "fraction of generations that were
non-hallucinatory." Still reports task_accuracy/hallucination_rate/abstention_rate
together (write_results enforces this), per CLAUDE.md's reporting requirement.
```

with:

```python
This validates decomposition+Z3 pipeline mechanics end-to-end on real FActScore
data, not the factual accuracy of the generated biography — task_accuracy here
means "fraction of examples whose entity resolved and whose ground-truth probe
claim the backend verified correctly," not "fraction of generations that were
non-hallucinatory." Still reports task_accuracy/hallucination_rate/abstention_rate
together (write_results enforces this), per CLAUDE.md's reporting requirement.

Separately, every example's generated biography IS now scored for real against
its own FActScore reference text (`example.wikipedia_text`), using the NLI
judge's atomic-decomposition-based verdict (`sense_eval.nli_judge.factscore_style_verdict`)
— reported under the `factscore_*`-prefixed keys, kept distinct from the
Z3-round-trip keys above so the two different things this script measures are
never confused with each other. This is this project's first real (non-placeholder)
FActScore factuality number, though its threshold calibration is a named,
unvalidated first cut (see `docs/superpowers/specs/2026-08-16-nli-judge-design.md`).
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd experiments && uv run pytest tests/test_factscore_symbolic_verification.py -v`
Expected: PASS

- [ ] **Step 5: Run the full experiments test suite**

Run: `cd experiments && uv run pytest -v`
Expected: all tests pass, no remaining references to the retired scorer
anywhere (`grep -rn "lexical_containment_verdict\|PLACEHOLDER_FACTUALITY_METRIC_LABEL" experiments/ eval/`
should return nothing).

- [ ] **Step 6: Commit**

```bash
git add experiments/factscore_symbolic_verification.py experiments/tests/test_factscore_symbolic_verification.py
git commit -m "Score FActScore biographies with the NLI judge's atomic-decomposition verdict"
```

---

### Task 9: Update docs to close out the metrics status item

**Files:**
- Modify: `experiments/README.md`
- Modify: `CLAUDE.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update `experiments/README.md`**

In the `## rag_baseline_halueval.py` section, replace the sentence:

```markdown
Reports a
`factuality_accuracy_proxy` using the same lexical-containment placeholder metric as
RQ3 (`eval/src/sense_eval/factuality.py`, see its README), not a paper-grade judge —
a pipeline sanity check, not a reportable accuracy figure.
```

with:

```markdown
Reports task_accuracy/hallucination_rate/abstention_rate scored by the real
NLI judge (`eval/src/sense_eval/nli_judge.py`, see its README) — semantic
entailment, not substring matching, though its threshold calibration is a
named, unvalidated first cut.
```

In the `## rq3_accuracy_latency_halueval.py` section, replace:

```markdown
and a factuality proxy score (`eval/`'s lexical-containment
placeholder — see its README; not a paper-grade judge) as
task_accuracy/hallucination_rate/abstention_rate together.
```

with:

```markdown
and a factuality score from `eval/`'s NLI-based judge (see its README;
real semantic entailment scoring, though its threshold calibration is a
named, unvalidated first cut) as
task_accuracy/hallucination_rate/abstention_rate together.
```

In the `## factscore_symbolic_verification.py` section, append a paragraph
after the existing description:

```markdown
Also reports a second, independent factuality result under `factscore_*`-prefixed
keys: every example's generated biography is scored by the NLI judge's
atomic-decomposition verdict against its own FActScore reference text
(`example.wikipedia_text`) — this project's first real FActScore factuality
number, kept separate from the Z3-round-trip keys above so the two are never
confused.
```

- [ ] **Step 2: Update `CLAUDE.md`'s "Current status" section**

Find the `**What's still needed to fully close out the reconciled scope:**`
list (added 2026-08-16). Change item 2 from:

```markdown
2. **Real judge for factuality** — `eval/src/sense_eval/factuality.py`'s
   lexical-containment scorer is still a placeholder (loudly labeled as such in
   every result it produces); replacing it with a fine-tuned judge or
   FActScore-style atomic-fact scoring is unstarted.
```

to:

```markdown
2. **Real judge for factuality** — done (2026-08-17). The lexical-containment
   placeholder is retired; `eval/src/sense_eval/nli_judge.py` scores semantic
   entailment with a pinned local NLI model
   (`cliang1453/deberta-v3-xsmall-mnli`), used by every harness that produces
   factuality numbers, including `factscore_symbolic_verification.py`'s
   FActScore biographies (atomic-decomposition + supported-fraction, matching
   FActScore's own methodology). Named limitation, not yet resolved: the
   entailment/fraction thresholds (`configs/nli_judge.yaml`) are a first cut,
   not calibrated against human-labeled ground truth — none exists yet for
   this project. See `docs/superpowers/specs/2026-08-16-nli-judge-design.md`.
```

Also find the paragraph in the same section that reads:

```markdown
FActScore's loader and committed three-way split (`data/src/sense_data/factscore.py`,
`data/splits/factscore.json`, 500 entities) are built and tested, but nothing
consumed them until `experiments/factscore_symbolic_verification.py`
(2026-08-16): it generates a biography per FActScore entity and runs the
decomposition -> Z3 round-trip against a *known ground-truth* probe claim for any
entity that resolves against the backend's small fixed domain KB — most don't
(the KB is deliberately small, per CLAUDE.md's scope-containment decision), and
are recorded as abstained rather than guessed at. This is real, non-fabricated
pipeline-mechanics validation of the decomposition/Z3 path against FActScore
data, **not** a factuality judgment of the generated biography — the project
still has no free-text-to-triple extractor (same reason RQ3's symbolic probe
triple isn't derived from the question either; extracting claims from arbitrary
generated prose is real, unscoped future work, tied to the still-undecided
"replace" merge-back policy). Do not read this script's `task_accuracy` as a
FActScore benchmark number.
```

Append one sentence at the end of that paragraph:

```markdown
 As of 2026-08-17, the same script's `factscore_*`-prefixed keys ARE a real
FActScore factuality number — scored by the NLI judge's atomic-decomposition
verdict against each example's own reference text, independent of the Z3
round-trip check this paragraph describes. Don't conflate the two: the
unprefixed `task_accuracy` above is still the Z3-known-entity check, not a
factuality judgment.
```

- [ ] **Step 3: Commit**

```bash
git add experiments/README.md CLAUDE.md
git commit -m "Document the NLI judge rollout in experiments/README.md and CLAUDE.md"
```

---

## Final verification (after Task 9)

Run the full test matrix locally before pushing:

```bash
cd eval && uv run pytest -v
cd ../experiments && uv run pytest -v
cd ../services/symbolic && uv run pytest -v
cd ../services/orchestrator && uv run pytest -v
```

Then `grep -rn "lexical_containment_verdict\|PLACEHOLDER_FACTUALITY_METRIC_LABEL" .`
from the repo root (excluding `.venv`) and confirm it returns nothing —
the placeholder is fully retired, not just unused in the touched files.
