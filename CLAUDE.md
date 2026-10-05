# SENSE — Project Context

**SENSE: A Self-Adaptive Entropy-Gated Neuro-Symbolic Framework for Cross-Model Hallucination Mitigation**
Self-adaptive ENtropy-aware Symbolic Engine

Master's Culminating Project (Starred Paper, CSCI699) · Computer Science MS · St. Cloud State University

---

## Related documents (external, not in this repo)

Working reference and drafting material live outside the repo, in
`/Users/udaysah/Documents/files/filess/`:

- **`SENSE_starred_paper_reference.md`** — working reference consolidating the chosen
  direction, adopted title, degree structure (Starred Paper track), reading list, and
  advisor-response log. Source of truth for scope/timeline decisions; updated
  incrementally as advisor feedback lands.
- **`SENSE_starred_paper_writing_guide.md`** — draft and writing guide: paper structure
  (IEEE format), section-by-section guidance, committee/procedure notes.
- **`SENSE_starred_paper_proposal_v5.docx`** — the current written proposal document
  (v5, revised 2026-08-25). Supersedes `SENSE_starred_paper_proposal_v4.docx` (kept
  alongside it for history) — v5's only substantive changes from v4 are marking the
  benchmark pair, symbolic backend, and merge-back policy as confirmed with the
  advisor in person on 2026-08-25 (v4 had them as `[PROPOSED — pending advisor
  confirmation]`); RQs, model table, and phase plan are unchanged between the two.

Title updated 2026-08-10 to match the reference doc's adopted title (was "...Real-Time
Hallucination Mitigation", retired the same day per advisor feedback — see the
reference doc §3). RQ1–RQ3 already matched verbatim, no change needed there. Title
formally confirmed in person 2026-08-25 (proposal v5 §8.1) — no wording change, just
sign-off.

**Scope reconciled 2026-08-10** against the reference doc's §5 scoping (confirmed
with advisor the same day): five checkpoints across three families, 4-bit
quantization, HaluEval + FActScore in place of TruthfulQA, Z3-based symbolic backend
in place of SPARQL/Wikidata. Reflected below in "Non-negotiable scientific
constraints," "Design decisions," and "Current status." The code and configs built
under the retired scope (TruthfulQA data, SPARQL backend, Llama-3/Mistral-only,
bf16) still work and are not thrown away — see "Current status" for exactly what
still needs to move to match the reconciled decisions.

**Benchmark pair, symbolic backend, and merge-back policy confirmed with the advisor
in person on 2026-08-25** (proposal v5 §4.3, §5, §8.1) — the "still PROPOSED pending
advisor confirmation" hedge that applied to the benchmark and symbolic-backend
choices from 2026-08-10 through 2026-08-25 no longer applies. See "Design decisions"
below.

---

## What this system does

An LLM generates normally. A monitor computes predictive uncertainty at each decoding
step. When uncertainty crosses a **per-model calibrated threshold**, the generation is
routed to a symbolic verification stream instead of being accepted as-is. The symbolic
result is merged back and generation resumes.

The research question is **not** "does entropy gating work." It is **does a gating policy
calibrated on one model transfer to a model from a different family or scale.** Every
design decision serves that question.

### Research questions

- **RQ1** — Does a fixed entropy-gating threshold transfer across model families?
- **RQ2** — Does a self-adaptive threshold that recalibrates per model preserve routing
  quality where a fixed one does not?
- **RQ3** — What accuracy–latency trade-off does routing introduce, and does it hold
  across all checkpoints?

### What "routing quality" means (RQ1, RQ2)

Defined 2026-10-04 (proposal-v5 review issue C1); before that the proposal used the
phrase without defining it and the two harnesses each substituted something else.

**Routing quality is detection performance of the gate against the ungated model's own
correctness.** The gate's job is to fire on examples that would otherwise be
hallucinated and stay quiet on examples that would be answered correctly, so quality
is precision/recall/F1 of routed-vs-hallucinated, plus the threshold-free AUROC of the
entropy signal. Labels come from HaluEval's own `right_answer`/`hallucinated_answer`
pair via the NLI judge — no extra annotation. Implemented in
`eval/src/sense_eval/routing_quality.py`.

Two quantities that are **not** routing quality, both retained as secondary
diagnostics and neither to be reported as the headline:

- `transfer_agreement_rate` (RQ1) — concordance between the transferred and native
  gates. Reads 1.0 whenever both route the same examples, right or wrong.
- `*_calibration_fidelity_gap` (RQ2) — whether a threshold still hits its own design
  firing rate. A gate firing on exactly `1 - quantile` of examples chosen at random
  scores perfectly.

### Models

Five checkpoints across three families, per advisor-confirmed scope
(2026-08-10) — both transfer axes get tested: cross-*family* via the three
anchors, cross-*scale* via the Qwen3 ladder.

| Role | Checkpoint |
|---|---|
| Cross-family anchor | Llama-3 8B Instruct |
| Cross-family anchor | Mistral 7B Instruct v0.3 |
| Cross-family anchor / scale-ladder top | Qwen3 8B |
| Scale ladder | Qwen3 4B |
| Scale ladder | Qwen3 1.7B |

Excluded on purpose, not by omission: reasoning-distilled models (extended-CoT
token-entropy profile differs) and MoE models (per-token uncertainty less
interpretable). Qwen3 runs in non-thinking mode throughout. The scale axis is
acknowledged as capped at 1.7B–8B — state that as a limitation, don't imply a
larger model was tested.

---

## Non-negotiable scientific constraints

These do not raise exceptions when violated. They silently produce results that look
fine and mean nothing. Treat any change touching them as requiring explicit human
sign-off.

### 1. Data split discipline

Three disjoint splits: **calibration**, **development**, **test**.

- Gate thresholds are fit **only** on the calibration split.
- The test split is touched once, at the end, for the reported numbers.
- Any code path that reads test data during threshold fitting is a bug of the highest
  severity, even if the pipeline runs cleanly.
- Splits are defined by fixed index lists committed to the repo, not by a runtime
  random shuffle.

### 2. Quantization is held constant

One quantization scheme, applied identically to every model and every condition. It
perturbs the logit distribution and therefore the gate input. If quantization varies
across conditions, cross-model comparisons are meaningless.

**Scheme: 4-bit, all five checkpoints** (confirmed with advisor 2026-08-10 as a hard
scientific constraint, not a memory-driven convenience). This replaces the earlier
bf16-only plan from the two-model (Llama-3/Mistral) scope — bf16 configs built under
that plan need re-pinning to 4-bit before any of the five-checkpoint runs.

### 3. Decoding configuration is held constant

Same temperature, same `max_new_tokens`, same sampling strategy across all conditions
within an experiment. Entropy characterization runs use greedy decoding.

### 4. Entropy is not comparable across models in raw form

Llama-3, Mistral, and Qwen3 each have different tokenizers and vocabulary sizes —
and the Qwen3 ladder (8B/4B/1.7B) adds a same-family, same-tokenizer axis where raw
entropy is comparable across scale but must still be checked, not assumed. Raw
entropy values are not directly comparable across families. Normalize by `ln(V)` or
compare quantiles. Read `V` from `tokenizer.vocab_size` at load time — never
hard-code it.

**Both scales are recorded, and the "or" above is load-bearing (2026-10-04).** The
gate calibrates on the normalized scale AND by quantile, which is "and," not "or" —
each step independently removes a source of cross-model difference, and RQ1 exists to
measure cross-model difference. `TokenEntropyMonitor` therefore keeps
`raw_entropies` alongside the normalized `entropies`, and RQ1/RQ2 each run a full
transfer arm on both scales. Do not go back to discarding the raw value: a run that
records only the normalized scale cannot test RQ1's premise afterwards, which is the
state the first pilot
(`results/rq1_transfer_truthful_qa_llama3_to_mistral.json`, agreement 1.0) was left
in. See `configs/rq1.yaml` and the proposal-v5 review (issue C3).

### 5. Model revisions are pinned

Hugging Face repos are mutable. Pin the commit SHA, not just the repo name. Record the
revision in every run's metadata.

### 6. Terminology

- **Token entropy** — Shannon entropy over the next-token distribution. Cheap, per-step.
- **Semantic entropy** — meaning-level uncertainty requiring multiple sampled
  generations or a hidden-state probe. Expensive.

These are different quantities. Do not use "entropy" to refer to both. Variable names,
comments, and log fields must say which one they mean.

---

## Architecture

### Language — Python throughout

| Component | Language | Why |
|---|---|---|
| Entropy monitor, model loading, probes | **Python** | Per-step logits and hidden states are only reachable in-process via Hugging Face Transformers |
| Gate policy, routing, symbolic interface, orchestration | **Python** | Single-language stack; kept as a separate service module so it only talks to the neural process over HTTP, preserving the same interface boundary a separate codebase would give |
| Symbolic backend | TBD | Runs as its own container behind a stable HTTP contract |

The HTTP serving path (vLLM, OpenAI-compatible) exposes only truncated top-*k* logprobs
and no hidden states. It is usable for baseline throughput, **not** for the gated
pipeline.

---

## Development environment vs. GPU environment

Local development almost certainly has no suitable GPU. Do **not** treat "can't run it
locally" as a reason to skip tests.

**Rule: every component must be testable on CPU with a tiny model.**

- Keep model identity in config. Nothing hard-codes Llama-3, Mistral, or Qwen3.
- The test suite runs against a small CPU-friendly model so the full pipeline —
  entropy computation, gate decisions, routing, merge-back, instrumentation — is
  exercised on every change.
- All five checkpoints (Llama-3 8B, Mistral 7B, Qwen3 8B/4B/1.7B) run on the GPU
  environment only, using the identical code path.

If a change cannot be tested against the tiny model, that is a design problem with the
change, not an excuse.

---

## Correctness checks that must exist

- **The entropy monitor is a no-op on generation.** Assert that generation with the
  monitor attached is token-identical to generation without it, on a fixed seed. If
  the monitor perturbs the distribution, every result is invalid.
- **Truncated vs. exact entropy.** The orchestrator's API-derived estimate carries
  explicit lower and upper bounds on the true entropy. Test that the exact in-process
  value falls within those bounds.
- **Gate refuses to run uncalibrated.** Calling the routing decision before calibration
  raises, never defaults to a magic number.
- **Split leakage test.** A test that fails if calibration code reads test indices.

---

## Things to never do

- **Never fabricate or placeholder experimental results.** No made-up accuracy numbers,
  no invented latency figures, no illustrative tables presented as measurements. If a
  number is not measured, it does not exist. This applies to code comments, README
  examples, and docstrings.
- **Never invent citations.** Reference numbering maps to the proposal's IEEE list.
  If a citation is needed and unknown, flag the gap.
- **Never commit secrets.** `HF_TOKEN` lives in `.env`, which is gitignored. Llama-3 is
  a gated repo and requires it.
- **Never bake model weights into a Docker image.** Mount a host cache as a volume.
- **Never fit a threshold on anything but the calibration split.**
- **Never add a default threshold value** as a convenience fallback.

---

## Reproducibility requirements

- Pinned Docker images by digest, not tag.
- Pinned Python dependencies.
- Fixed seeds, recorded per run — `configs/run.yaml`'s `seed`, applied by
  `_common.seed_everything()` (called from every reportable harness's entry point)
  and stamped into every result file by `_common.write_results()`. Before
  2026-10-04 the only seed in the repo governed split construction, and no result
  recorded one.
- Config-driven experiments — no parameters passed as edited source.
- Every run logs: hardware, GPU model, driver, CUDA version, image digest, model
  revision SHA, quantization scheme, decoding config, seed.
- One command reproduces the headline result.

---

## Latency instrumentation

RQ3's accuracy–latency trade-off claim rests entirely on these numbers — the title no
longer promises real-time (retired 2026-08-10; see "Related documents"), so latency is
reported as a **result**, not asserted upfront. That makes the instrumentation more
important, not less: an unsupported trade-off claim is exactly as bad as an
unsupported real-time claim. Instrument from the start; retrofitting timing after
experiments run is how the claim ends up unsupported. No numeric latency target is
committed to in advance — state actuals once Phase 2+ produces them, per the
reference doc's advice (§6, open items).

- Measure **per stage**: time-to-first-token, mean inter-token latency, gate evaluation
  cost, symbolic round-trip, merge-back.
- Discard warm-up runs. First-call kernel compilation otherwise dominates.
- Measure inside the container, not from the host.

---

## Design decisions

Decided 2026-07-26 as a working proposal; **benchmark, symbolic backend, and scope
reconfirmed/superseded 2026-08-10** per the reference doc's advisor-response pass
(§5, §7). **Benchmark and symbolic backend confirmed with the advisor in person on
2026-08-25** (proposal v5 §4.3, §5, §8.1) — no longer `PROPOSED`; the two bullets
below have been updated accordingly.

- **Uncertainty signal: token entropy.** The sole online gating signal — the only
  option cheap enough to sit in the per-step decoding loop. Optionally aggregated
  over a span or combined with a semantic-uncertainty measure at the *definition*
  level (writing guide §3.3), but semantic entropy itself stays out of the online
  gate; if used at all, it's a single offline correlation study, not part of the
  pipeline.
- **Quantization: 4-bit, all five checkpoints** (2026-08-10, supersedes the earlier
  bf16-only plan). See "Non-negotiable scientific constraints" #2 — this is a hard
  constraint, not a convenience.
- **Merge-back policy: annotate-only first, replace as a later follow-on** (confirmed
  with the advisor in person 2026-08-25, proposal v5 §4.3). Run the
  RQ1–RQ3 routing/latency experiments on annotate-only so routing cost is measured
  without confounding it with a correction strategy. Add a replace condition only
  once the symbolic backend is stable. **Replace is out of scope entirely**, not just
  deferred — it needs mid-generation resume, which makes routing overhead a function
  of regeneration length and directly confounds RQ3 (reference doc §5.3). Abstain is
  configurable and shares nearly all code with annotate.
  **Reporting requirement:** always report hallucination rate, task accuracy, and
  abstention rate together — abstaining trivially drives hallucination rate toward
  zero at the cost of accuracy, and a hallucination-reduction number reported alone
  is misleading (reference doc §5.3).
- **Benchmark: HaluEval + FActScore** (2026-08-10, supersedes TruthfulQA). HaluEval's
  volume (35k samples) supports genuine held-out calibration splits *per checkpoint*
  across all five models without eating the test set; FActScore's atomic-fact
  decomposition operates at the granularity a per-span gate needs, and that
  decomposition step is shared with the symbolic backend below — one implementation,
  two uses. **TruthfulQA is deliberately excluded**, not omitted: it targets
  imitative falsehoods (confidently-held misconceptions), which is exactly the
  regime where entropy is *low* and the gate will not fire. State this as a named
  limitation, not a silent gap — it's more informative than a weak number against a
  benchmark the mechanism isn't built to catch. **Confirmed with the advisor in
  person on 2026-08-25** (proposal v5 §5, §8.1) — no longer `PROPOSED`.
- **Symbolic backend: Z3 SMT constraint checking with an atomic-claim decomposition
  front-end** (2026-08-10, supersedes SPARQL/Wikidata). Decompose the generation into
  atomic claims first, then verify with Z3 — decomposition-first matters because the
  documented failure mode of solver-backed NL verification is *autoformalization
  error* (a syntactically valid but semantically wrong translation makes the solver
  verify the wrong proposition — "verified hallucinations"); short atomic claims
  narrow that surface. Closest precedent: VERGE (arXiv:2601.20055) does
  decompose→autoformalize→solve plus multi-model consensus and error localization;
  SENSE's contribution beyond it is *gating* the symbolic call on the model's own
  uncertainty signal rather than always-on iterative refinement. Keep the interface
  swappable — KG lookup (the retired SPARQL/Wikidata approach) remains available as a
  fallback if the solver-based verifier proves brittle. **Contain scope**: fixed
  constraint set, one benchmark domain — the symbolic backend can quietly become a
  research project of its own and eat the 3-credit budget. **Confirmed with the
  advisor in person on 2026-08-25** (proposal v5 §4.3, §8.1) — no longer `PROPOSED`;
  the advisor's own question ("what other structured verifiers are there?") now has
  a concrete, citable answer.
- **Baseline RAG stack: FAISS + all-MiniLM-L6-v2 + a small Wikipedia passage subset.**
  In-memory vector store (no server), small CPU-friendly embedding model, modest
  reproducible corpus. This is a comparison baseline, not the contribution — capped
  effort on purpose. Built and validated against the now-retired TruthfulQA corpus
  (see "Current status") — needs re-pointing at HaluEval/FActScore data before it's
  usable as a baseline for the reconciled scope.

Build against these decisions as if final. If the real advisor changes one, treat it
as a design change, not a correction — update this file and the affected code
together, don't leave the doc stale.

---

## Current status

Phase 0, built under the **retired** two-model/TruthfulQA/SPARQL scope (decided
2026-07-26); the **title, RQ wording, and code** were then reconciled to the
five-checkpoint/HaluEval+FActScore/Z3 scope starting 2026-08-10. As of 2026-08-16,
the reconciliation is mostly complete — see the checklist below for what's left.
TruthfulQA's loader/split/`calibrate_gate_truthful_qa.py` are kept deliberately
(the excluded-benchmark limitation citation), not migrated.

**What works today, as built:** end-to-end on CPU against `sshleifer/tiny-gpt2`,
against real data: token entropy monitor with no-op verification, generation
latency instrumentation (TTFT, inter-token, via
`services/neural/src/sense_neural/latency.py`), split machinery with a leakage
guard (including HaluEval's per-checkpoint calibration shape and FActScore's
three-way split), the entropy-gating policy that refuses to run uncalibrated, the
truncated-vs-exact entropy bounds check, the Z3 symbolic backend with its
atomic-claim decomposition front-end (`services/symbolic`, SPARQL/Wikidata kept
as the swappable fallback, not deleted), and the gate-to-symbolic router
(`services/orchestrator`, annotate-only merge-back). All four required
correctness checks pass. `configs/model.yaml` carries all five checkpoints
(Llama-3 8B Instruct, Mistral 7B Instruct v0.3, Qwen3 8B/4B/1.7B), pinned to
4-bit quantization and enforced by `experiments/_common.py`'s
`assert_pinned_gpu_quantization` (constraint #2, non-negotiable — every GPU
harness calls this before running). `eval/src/sense_eval/factuality.py` requires
task_accuracy/hallucination_rate/abstention_rate to always be reported together
(`assert_factuality_metrics_reported_together`, called from every harness's
`write_results`) — enforced, not just documented.

All three research-question harnesses (`experiments/`) are re-pointed to HaluEval
with per-checkpoint calibration splits (`data/splits/halueval.json`) and run
end-to-end on real data: RQ1 `transfer_threshold_halueval.py` (fixed-threshold
transfer), RQ2 `adaptive_threshold_halueval.py` (self-adaptive threshold), RQ3
`rq3_accuracy_latency_halueval.py` (accuracy-latency trade-off — factuality now
scored by the NLI judge, not a placeholder; see `eval/README.md`). This migration
(plan
`docs/superpowers/plans/2026-08-14-rq1-rq3-halueval-five-checkpoint.md`) is
complete and merged. The RAG comparison baseline is re-pointed to HaluEval too
(`experiments/rag_baseline_halueval.py`, `services/rag`'s corpus tied to
HaluEval's test-split questions).

FActScore's loader and committed three-way split (`data/src/sense_data/factscore.py`,
`data/splits/factscore.json`, 500 entities) are built and tested, but nothing
consumed them until `experiments/factscore_symbolic_verification.py`
(2026-08-16): it generates a biography per FActScore entity and runs the
decomposition -> Z3 round-trip against a *known ground-truth* probe claim for any
entity that resolves against the backend's small fixed domain KB — most don't
(the KB is deliberately small, per CLAUDE.md's scope-containment decision), and
are recorded as abstained rather than guessed at. This is real, non-fabricated
pipeline-mechanics validation of the decomposition/Z3 path against FActScore
data, **not** a factuality judgment of the generated biography — this
Z3-known-entity probe check predates the free-text claim extractor and does
not use it (same reason RQ3's symbolic probe triple isn't derived from the
question either; RQ3 wiring remains real, unscoped future work, tied to the
still-undecided "replace" merge-back policy). Do not read this script's
`task_accuracy` as a FActScore benchmark number. As of 2026-08-17, the same script's `factscore_*`-prefixed keys ARE a real
FActScore factuality number — scored by the NLI judge's atomic-decomposition
verdict against each example's own reference text, independent of the Z3
round-trip check this paragraph describes. Don't conflate the two: the
unprefixed `task_accuracy` above is still the Z3-known-entity check, not a
factuality judgment. As of 2026-08-20, the same script's `extracted_*`-prefixed
keys ARE real symbolic verification of claims extracted from the model's own
generated biography text (via `services/symbolic/src/sense_symbolic/extraction.py`),
independent of both the Z3-known-entity probe and the NLI-judge `factscore_*`
keys described above — don't conflate any of the three.

**What's still needed to fully close out the reconciled scope:**
1. **Free-text claim extraction** — done for FActScore (2026-08-20). A
   rule-based extractor (`services/symbolic/src/sense_symbolic/extraction.py`)
   turns generated biography sentences into `AtomicClaim`s across the same
   five relation kinds the fixed KB can verify, reported under
   `factscore_symbolic_verification.py`'s `extracted_*`-prefixed keys — see
   `docs/superpowers/specs/2026-08-20-freetext-claim-extraction-design.md`.
   Named limitations: precision-biased fixed patterns (no negation handling,
   no coreference resolution). **Does not apply to RQ3** (2026-08-20,
   revised from an earlier "small follow-on" framing): HaluEval's QA
   examples have no title-entity field the way FActScore's biography
   examples do (`example.entity`), so the extractor would have no reliable
   subject to anchor claims to and would return zero claims on most
   open-domain questions — that would silently turn most routed examples
   into "nothing to verify," defeating RQ3's actual purpose (a real
   round-trip against the real backend on every routed example, regardless
   of question content). `rq3_accuracy_latency_halueval.py`'s fixed,
   always-resolvable probe triple is intentional, not a gap — see
   `configs/rq3.yaml`'s comment.
2. **Real judge for factuality** — done (2026-08-17). The lexical-containment
   placeholder is retired; `eval/src/sense_eval/nli_judge.py` scores semantic
   entailment with a pinned local NLI model
   (`cliang1453/deberta-v3-xsmall-mnli`), used by every harness that produces
   factuality numbers, including `factscore_symbolic_verification.py`'s
   FActScore biographies (atomic-decomposition + supported-fraction, matching
   FActScore's own methodology). `short_answer_entailment_threshold`
   (HaluEval-shaped harnesses: RQ1-RQ3, RAG baseline) is calibrated
   (2026-08-20, `experiments/calibrate_nli_short_answer_threshold.py`)
   against HaluEval's own real right_answer/hallucinated_answer construction
   — no human labeling needed; 0.7 (unchanged) scores 98.7%/98.2%
   verdict-accuracy fit/held-out (`results/nli_short_answer_calibration.json`).
   Honest limitation of that calibration: it uses canonical answer text as a
   stand-in generation, not a real paraphrased model output. FActScore's
   three thresholds (`claim_supported_threshold`, `fraction_correct_threshold`,
   `fraction_incorrect_threshold`) are now calibrated too (2026-08-26,
   `experiments/calibrate_factscore_thresholds.py`) — closing the gap noted
   above, since FActScore has no HaluEval-style free known-right/known-wrong
   pair. Took the human-labeling route via the original FActScore paper's own
   released 183-entity human-annotated set (InstructGPT/ChatGPT/PerplexityAI
   generations, real human S/NS/IR labels — `sense_data.factscore_labeled`,
   provenance in `data/fixtures/factscore_labeled/PROVENANCE.md`), not
   this project's own five checkpoints' generations. `claim_supported_threshold`
   = 0.02 (80.7% claim-level verdict accuracy fit/held-out — a genuine
   interior peak, not a swept-range boundary artifact: the first calibration
   attempt landed exactly on its sweep's lower edge and was rejected and
   re-run with an extended range before trusting the result).
   `fraction_correct_threshold`/`fraction_incorrect_threshold` = 0.09/0.0
   (90.8%/94.3% fit/held-out, a genuine accuracy plateau across
   `fraction_incorrect_threshold` in [0.0, 0.09], not a runaway edge value).
   Full sweep results and honest limitations (2026 Wikipedia reference-text
   snapshot vs. the annotators' 2023-vintage text; judged generations aren't
   this project's own models) in `results/factscore_threshold_calibration.json`
   and `configs/nli_judge.yaml`'s comments. See
   `docs/superpowers/specs/2026-08-16-nli-judge-design.md`.

Not yet built regardless of scope: real-model GPU runs against the reconciled
current scope).

---

## 2026-10-04 — proposal-v5 review changes

A peer-review pass over proposal v5 (`docs/research/sense-proposal-v5-review.md`, run
via the `feynman-research-review` workflow) raised three Critical issues. Two were
code-level and are addressed here; the third is document-level and is not yet done.

**Addressed in code** (branch `review-c1-c3-routing-quality`):

- **C1 — routing quality was undefined.** Added
  `eval/src/sense_eval/routing_quality.py` and wired it into RQ1, RQ2, and RQ3. See
  "What routing quality means" above. RQ1 and RQ2 now generate, score with the NLI
  judge, and report the required factuality triple under an `ungated_` prefix; the
  marginal cost over the previous entropy-only path is the judge pass, not a second
  round of generation.
- **C3 — normalization and quantile calibration together may make a negative RQ1
  result unreachable.** `TokenEntropyMonitor` retains `raw_entropies`; RQ1 and RQ2
  each run both a `normalized` and a `raw` transfer arm. See constraint #4 above.
- **M3 — no statistical treatment.** Two parts, both done. (a) A per-run seed:
  `configs/run.yaml` + `seed_everything()`, stamped into every result file. (b) The
  statistical treatment itself — see the note below, because it is deliberately **not**
  what the writing guide asks for.

**Added alongside, not from the review itself:**

- `experiments/characterize_entropy_distributions.py` + `configs/characterize.yaml` —
  the RQ1 pre-flight diagnostic. Reports each checkpoint's entropy distribution on
  both scales, the threshold spread across checkpoints
  (`max_over_min_ratio` — the headline number), and what each checkpoint's routing
  rate would do under every other checkpoint's threshold, split into cross-family and
  same-family (scale-ladder) pairs. Uses no factuality labels and makes no
  routing-quality claim; it is a decision point before the full run, not an RQ result.
  Wired into `run_gpu_experiments.sh` as the `characterize` stage, deliberately
  excluded from `all`.

  **Run this before the five-checkpoint suite.** If the raw-scale threshold spread
  comes back near 1.0, RQ1's premise does not hold as the proposal states it, and the
  framing needs revisiting before five checkpoints of compute go into answering it. It
  is also the first time any real checkpoint gets loaded, so it doubles as a smoke test
  of 4-bit loading and the entropy monitor under quantization — which constraint #2's
  note says still needs sanity-checking.

  Hypothesis worth checking it against: AdaDec (arXiv:2506.08980, Table VI) reports
  learned raw-entropy thresholds spanning 0.6153-1.9353 nats across eight checkpoints,
  but only 0.6153-0.7134 within the Qwen3 family — wide across families, tight across
  scale. Different task and a different fitting procedure, so treat it as a prediction
  to test, not evidence to cite.

### The post-hoc verification baseline (M2)

`experiments/selfcheck_baseline_halueval.py` + `eval/src/sense_eval/selfcheck.py` +
`configs/selfcheck.yaml`, added 2026-10-04. SelfCheckGPT-NLI: sample N stochastic
continuations per prompt, score the main greedy answer's inconsistency against them as
the mean probability that a sample contradicts each of its sentences, and flag above a
calibrated threshold.

This is the baseline the proposal's central framing depends on. "Neither always on, as
in retrieval, nor applied after the fact, as in post-hoc verification, but invoked
selectively during generation" is a claim *relative to* after-the-fact checking, and
until now nothing quantitative stood behind the second half of it.

**Run at matched budget.** The flag threshold is calibrated at the same quantile the
entropy gate uses, so both flag the same fraction of examples and their precision,
recall and AUROC are directly comparable — `detection` in the baseline's result file
against `routing_quality`/`routing_detection` in RQ1-RQ3. Comparing detectors at
different rates would confound signal quality with intervention rate. `GatePolicy` is
reused for the thresholding (it provides exactly what is wanted — a
quantile-calibrated threshold with the leakage guard and the refuse-if-uncalibrated
rule), so the signal differs between the two and the thresholding discipline does not.

**The cost side is measured, not asserted.** This method needs `n_samples + 1` full
generations per example; the entropy gate needs a logarithm over a distribution the
forward pass already produced. `mean_posthoc_overhead_ratio` reports that gap.

Two limitations that **favor SENSE** and therefore have to be stated in the results
discussion rather than left in a config comment: `n_samples` defaults to 10 where
Manakul et al. use 20, and the judge is this project's pinned
`deberta-v3-xsmall-mnli` rather than the DeBERTa-v3-large the paper used. Both weaken
the baseline. Raise `n_samples` if the budget allows; do not quietly lower it.

One justified exception to constraint #3 (hold decoding constant): the sample passes
necessarily sample, since that is the method. The **main** answer uses gate.yaml's
greedy config, identical to every other condition, so the thing being scored stays
comparable across the study. `sampling_decoding_cfg` changes only `do_sample` and
`temperature`, and a test pins that.

### Why the statistical treatment is bootstrap intervals, not multiple seeds

The writing guide §4.6 asks for "multiple seeds, report variance." That is the right
instinct for a stochastic pipeline and the wrong instrument for this one:
`configs/gate.yaml` sets `do_sample: false`, so re-running a checkpoint on the same
examples is byte-identical and seed-to-seed variance is exactly **zero**. Reporting
that zero as variance would claim a stability result the experiment never tested.

The variance that genuinely exists is sampling variance, from two sources:

1. **Calibration sampling** — the threshold is a quantile of a finite calibration
   draw. Another draw from the same model gives a different threshold, and everything
   downstream is a function of it. This is the dominant source.
2. **Evaluation sampling** — precision, recall, F1 and AUROC are estimated on a finite
   development split.

Both are addressable by resampling data already in hand, at **no extra GPU cost** — no
extra generation, no extra judge calls. `eval/src/sense_eval/bootstrap.py` provides
percentile intervals for each, and RQ1/RQ2 now report a threshold interval per gate, a
per-metric interval on every detection number, and — most importantly — a **paired**
interval on the transferred-vs-native difference. Paired because the two gates are
scored on the same examples, so their errors are correlated and independent resampling
would widen the interval and understate a real gap.

**An interval excluding zero on `routing_quality_delta_ci` is RQ1's and RQ2's
evidence.** An interval straddling zero means the data does not distinguish the two
gates, which on these research questions is a finding in its own right and must be
reported as one rather than presented as a null.

Say this in the methods section rather than letting a reader assume seeds were varied.
When the proposal's §5 gets its statistical-treatment paragraph (M3, document side),
this is the argument it should make.

**Document side, done 2026-10-04** — in
`/Users/udaysah/Documents/files/filess/SENSE_starred_paper_proposal_v5.docx` and
`SENSE_starred_paper_writing_guide.md`, edited in place under their original
filenames. The copy in the claude.ai project Files panel is **stale** and must be
re-uploaded by hand; that store cannot hold binary, so it was left alone rather than
overwritten with extracted text (which is how the v4-era "UTF-8 mislabeled as .docx"
problem started).

- **C2** — the abstract, §2.1, §3, §4.1, §6 and §7 now name token entropy as the
  gating signal. §6's cost argument is rewritten: the driver is the symbolic stream
  scaled by the routing rate, not the uncertainty estimate.
- **M1** — new §2.4, "Uncertainty-gated intervention", citing Varshney et al. [25],
  AdaDec [26] and UnCert-CoT [27], and narrowing the contribution claim to the three
  parts that survive them. Kuhn et al. added as [24] for the semantic-entropy
  definition.
- **M4** — §4.3 discloses the constraint set's single-domain coverage, that
  unresolved claims are recorded as unverified rather than guessed at, and that RQ3's
  latency probe is always-resolvable by design.
- **M5** — the real-time argument is retired from §5 and §6 and from the writing
  guide's §A.3 and §C.2.
- **M3 (document side)** — §5 gains the statistical-treatment paragraph making the
  bootstrap-not-seeds argument.
- **Minors done:** m1 (ref [1] published), m4 (Kuhn added), m5 (RQ wording aligned),
  m6 (calibration-set wording), m7 (hardware claim softened; this doc updated too),
  m8 (Table I expectation marked as a hypothesis).

**Still open:**

- **m2** — reference [10] still lacks author initials. They must come from the paper,
  not from memory.
- **m3** — the final citation renumbering pass. [22]–[27] are all appended out of
  order; the reference notes say so.
- **m9** — the AI Use Statement question in §8.2 is unresolved, and the log still has
  not been started.
- **Citation verification** — [24], [25] and [27] were identified through citations in
  other work; only [26] was read in full. [1]'s publication details come from secondary
  reference lists. All four need confirming against publisher records before
  submission, and the reference notes in the proposal say so.

**Verification status of the code changes above.** 115 model-free tests pass:
`eval/tests/test_routing_quality.py` (27, including hand-computed AUROC values, tie
handling, and the confidence-interval wiring), `eval/tests/test_bootstrap.py` (19,
including the paired-vs-independent width property and degenerate-resample
suppression), `eval/tests/test_selfcheck.py` (13, including premise/hypothesis
direction and contradiction-label resolution),
`experiments/tests/test_transfer_arms.py` (16, arm logic and interval wiring),
`experiments/tests/test_characterize_entropy_distributions.py` (34), and
`experiments/tests/test_selfcheck_baseline_halueval.py` (6). All run without loading a
model.
The torch-dependent suites — `services/neural/tests/test_entropy_monitor.py`'s two new
raw-scale tests and the updated RQ1/RQ2 end-to-end harness tests — **have not been
run**; they need the macOS dev environment. Run before trusting this branch:

```bash
cd services/neural && uv run pytest -q
cd ../../eval        && uv run pytest -q
cd ../experiments    && uv run pytest -q
```
