# Free-Text Claim Extraction for the Z3 Symbolic Backend — Design

**Status:** approved for implementation planning.

## Problem

`services/symbolic`'s Z3 verifier (`z3_verifier.py`) checks one
`AtomicClaim` — an already-structured `(subject_label, relation_kind,
object_label)` triple restricted to `domain.py`'s five fixed relation kinds
(`birth_year`, `death_year`, `nationality`, `occupation`, `employer`) plus a
two-entity `before_year` relation. Nothing in the codebase today turns free
text (an LLM's own generated prose) into that triple. Every place the Z3
backend is exercised — `experiments/factscore_symbolic_verification.py`'s
`build_probe_claim`, `experiments/rq3_accuracy_latency_halueval.py`'s
symbolic probe — hand-builds a single known-ground-truth triple per example
instead of extracting claims from the model's actual output.

CLAUDE.md names this the last major gap in the reconciled scope: "the
project still has no free-text-to-triple extractor... real, unscoped future
work" and "its own scoped task, not a quick addition." This design scopes
that task for `factscore_symbolic_verification.py` only (see "Out of
scope").

This is a different problem from the one
`docs/superpowers/specs/2026-08-16-nli-judge-design.md` already solved.
That design's "Two problems, not one" section named this exact gap and
explicitly deferred it: turning free text into a *Z3-checkable structured
triple* has near-zero tolerance for autoformalization error (a wrong
extraction makes Z3 confidently verify the wrong claim), unlike NLI
entailment scoring, where a wrong claim just moves a probability and still
fails toward "unknown." This design is the deferred half.

## Approach: rule-based pattern extraction, not NER or an LLM

Considered and rejected:
- **A small NER + relation-classifier model** — more general than fixed
  patterns, but adds a new model dependency and a second, harder-to-audit
  source of extraction error on top of Z3's own autoformalization risk.
- **LLM-based structured extraction** — most flexible, but reintroduces
  exactly the kind of free-form NL→formal translation risk
  decomposition-first was designed to narrow in the first place, and
  resists deterministic CPU testing.

Chosen: fixed regex/keyword patterns, one per relation kind, biased toward
precision over recall — a sentence yielding no claims is preferred over a
sentence yielding a wrong one. This mirrors `decompose_claim`'s existing
"refuse rather than guess" contract for the structured path (an
unrecognized `predicate_pid` returns `None`, not a coerced guess) and keeps
the whole extractor CPU-testable with no model dependency, matching the
project's CPU-testability rule.

## Module: `services/symbolic/src/sense_symbolic/extraction.py`

```
extract_claims(subject_label: str, sentence: str) -> list[AtomicClaim]
    # Runs each relation-kind pattern against `sentence` independently and
    # emits one AtomicClaim per match — a sentence can yield 0, 1, or
    # several claims (e.g. "Einstein was a German physicist born in 1879"
    # yields nationality + occupation + birth_year claims). subject_label
    # is supplied by the caller, not inferred from the sentence (see
    # "Subject resolution" below) — every AtomicClaim this function returns
    # already carries the same subject_label passed in.
```

Per-relation pattern sketch (deliberately narrow; each pattern is its own
small, independently testable unit inside the module, not one combined
mega-pattern):

- `birth_year`: `\bborn\b` co-occurring with a 4-digit year token
  (`1[0-9]{3}|20[0-9]{2}`) within the sentence.
- `death_year`: `\bdied\b` co-occurring with a 4-digit year token, same
  shape as `birth_year`.
- `nationality`: whole-word match against a small fixed vocabulary of
  nationality adjectives (German, English, Polish, American, ...) —
  closed vocabulary, not general adjective detection. Seeded from the
  nationalities already present in `domain.py`'s `KNOWN_ENTITIES`, so the
  vocabulary only needs to cover what the fixed KB could ever confirm or
  refute.
- `occupation`: whole-word match against the KB's own occupation
  vocabulary (physicist, mathematician, chemist, ...) plus a small
  synonym list — same closed-vocabulary reasoning as nationality.
- `employer`: `\bworked (at|for)\b` followed by a short trailing phrase up
  to the next sentence-ending punctuation. Named as the lowest-precision
  pattern of the five; first candidate to cut if manual review of test
  fixtures shows it producing bad extractions.

**`before_year` is out of scope for this extractor.** It is a two-entity
relation ("X before Y") that does not arise from a single free-text
sentence about one biography subject the way the other five do. It remains
reachable only through the existing structured/probe path
(`build_probe_claim`).

**Negation is a named, unsolved limitation**, stated in the module
docstring: a sentence like "he was not born in 1879" extracts the same
`birth_year` claim a positive statement would, because none of the five
patterns check for a preceding negation. This is not fixed by this design —
flagged the same way the NLI judge's threshold calibration was flagged, as
a first cut, not a complete extractor.

## Subject resolution: always the prompt's title entity

Every sentence in a generated FActScore biography is about the entity the
prompt asked about (`example.entity`) — the same one-entity-per-prompt
framing `factscore_symbolic_verification.py` already uses for
`build_probe_claim`. No coreference or pronoun resolution is implemented:
`extract_claims` takes `subject_label` as a parameter, supplied by the
caller as `example.entity` for every sentence in that example's
generation, rather than inferring it from pronouns sentence-by-sentence.
This is a scope decision, not an oversight — it sidesteps a second,
independent failure mode (wrong antecedent) that pronoun resolution would
introduce, in exchange for correctly handling only the common case
(subject named or implied once, referred to by pronoun afterward, always
the same entity throughout).

## Integration: a third, independent measurement in `factscore_symbolic_verification.py`

For each example, after `generated_text` is produced:

1. Split `generated_text` into sentences using the existing splitter,
   `eval/src/sense_eval/nli_judge.py`'s `split_into_atomic_claims` — reused
   rather than reimplemented, since it already does exactly this job for
   the NLI judge's atomic-decomposition path and is already a dependency
   this file's neighbor module carries.
2. For each sentence, call `extract_claims(example.entity, sentence)` and
   flatten the results into one list of `AtomicClaim`s for the example.
3. Run each extracted `AtomicClaim` through the existing, unchanged
   `verify_claim` (`z3_verifier.py`) — no changes needed to
   `z3_verifier.py` or `domain.py`.
4. Aggregate per example into a `FactualityVerdict`:
   - any extracted claim's `verify_claim` result is `False` → `"incorrect"`
   - else, if at least one extracted claim resolved (`True`) → `"correct"`
   - else (no claims extracted, or all extracted claims returned `None`,
     e.g. unresolved entity) → `"unknown"`

This is a simpler aggregation than the NLI judge's `supported_fraction`
threshold logic, deliberately: Z3's per-claim verdicts are already
tri-state and exact (True/False/None against the fixed KB), not a
continuous score needing a threshold, so a single `False` is decisive.

Reported on the result dict as a third, independent trio —
`extracted_task_accuracy` / `extracted_hallucination_rate` /
`extracted_abstention_rate` / `extracted_factuality_report` — following
the same pattern Task 8 of the NLI-judge plan established for
`factscore_*`, kept fully separate from both the existing hand-built
probe-claim Z3 check (unprefixed `task_accuracy`/...) and the NLI-judge
`factscore_*` trio. `per_example` also gains `extracted_claims_count` (how
many claims were actually extractable from that example's biography) for
transparency — given the KB's deliberately small size (10 entities, most
FActScore entities won't resolve), this number will likely be low, and
that should be visible, not hidden inside an aggregate.

`eval/src/sense_eval/factuality.py`'s `assert_factuality_metrics_reported_together`
already generalizes over any `<prefix>_`-suffixed triple (added in the
NLI-judge plan's final-review fix round) — it should gate `extracted_*`
correctly with no further change, but this design adds a test confirming
that specifically rather than assuming it.

## No fabrication

Extraction only decides *what claim to check* — it never invents an
answer. Every check still runs against `domain.py`'s existing fixed, real,
cited biographical KB, through the unmodified `verify_claim`. An
extraction bug can at worst cause a real claim to go unchecked (false
negative → `"unknown"`) or a wrong claim to be checked against real facts
(which Z3 will then correctly refute or fail to determine, per its
existing tri-state contract) — it cannot cause a fabricated fact to enter
the KB.

## Testing

`services/symbolic/tests/test_extraction.py`, CPU-only, no model
dependency:
- One positive-match test per relation kind (5 patterns × 1 sentence
  fixture each).
- A "no relation-kind pattern matches" case → `[]`.
- A "multiple claims in one sentence" case (the German-physicist-1879
  example above) → 3 claims from one sentence.
- A false-positive-guard case per pattern where a superficially similar
  sentence should NOT match (e.g. a sentence mentioning a year that isn't
  adjacent to "born"/"died").
- A negation case is included but asserts the *documented limitation*
  (the claim IS extracted despite negation) rather than asserting correct
  negation handling — makes the limitation testably explicit instead of
  silently uncovered later.

`experiments/tests/test_factscore_symbolic_verification.py` extends with
the new `extracted_*` keys and `extracted_claims_count`, verified
end-to-end against `sshleifer/tiny-gpt2` the same way the existing
`factscore_*` keys were tested in the NLI-judge plan's Task 8.

## Out of scope

- **RQ3's symbolic probe** (`experiments/rq3_accuracy_latency_halueval.py`)
  has the identical gap per CLAUDE.md, but is deliberately excluded from
  this pass. The extractor module is built shared/reusable
  (`services/symbolic`, not embedded in the FActScore harness), so wiring
  it into RQ3 is a small, separate follow-on task once this one is proven
  — not because RQ3 doesn't need it, but to keep this task's diff
  reviewable as one thing at a time, per CLAUDE.md's own framing of this
  as "its own scoped task."
- **Negation handling** — named limitation, not solved here.
- **Coreference/pronoun resolution for multi-entity text** — not needed
  under the "always the title entity" subject-resolution decision above;
  would only become relevant if this extractor were ever pointed at text
  discussing more than one entity per prompt, which none of this project's
  current harnesses do.
- **Widening `domain.py`'s fixed KB or `RELATION_KINDS` registry** —
  unrelated to this design; extraction only ever produces claims within
  the existing five (of six) relation kinds and the existing ~10-entity
  KB. A claim about an entity or relation outside that closed world simply
  fails to resolve later, in `verify_claim`, exactly as it does today for
  the hand-built probe-claim path.
- **The "replace" merge-back policy** this extractor was originally
  blocked behind (per the retired framing in
  `factscore_symbolic_verification.py`'s pre-existing docstring) — this
  design does not touch merge-back at all; it only extracts claims for
  offline FActScore scoring, the same annotate-only-adjacent measurement
  context the NLI-judge `factscore_*` trio already operates in.
