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
- **`SENSE_starred_paper_proposal_v4.docx`** — the written proposal document (v4).

Title updated 2026-08-10 to match the reference doc's adopted title (was "...Real-Time
Hallucination Mitigation", retired the same day per advisor feedback — see the
reference doc §3). RQ1–RQ3 already matched verbatim, no change needed there.

**Scope reconciled 2026-08-10** against the reference doc's §5 scoping (confirmed
with advisor the same day): five checkpoints across three families, 4-bit
quantization, HaluEval + FActScore in place of TruthfulQA, Z3-based symbolic backend
in place of SPARQL/Wikidata. Reflected below in "Non-negotiable scientific
constraints," "Design decisions," and "Current status." The code and configs built
under the retired scope (TruthfulQA data, SPARQL backend, Llama-3/Mistral-only,
bf16) still work and are not thrown away — see "Current status" for exactly what
still needs to move to match the reconciled decisions.

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

### Repository layout

```
sense/
├── CLAUDE.md
├── README.md
├── docker-compose.yml
├── .env.example              # HF_TOKEN placeholder
├── configs/                  # YAML: model, thresholds, dataset, splits
├── services/
│   ├── neural/               # Python: model loading, LogitsProcessor, entropy, probes
│   ├── symbolic/             # TBD backend behind an HTTP contract
│   └── orchestrator/         # Python: gate policy, routing, instrumentation
├── experiments/              # one script per table/figure in the paper
├── eval/                     # metrics: factuality + latency
├── data/                     # loaders and split index files (no large datasets)
└── results/                  # logged runs, metrics, plots
```

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
- Fixed seeds, recorded per run.
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
(§5, §7). The benchmark and symbolic-backend choices below are still marked
`PROPOSED` in the reference doc pending final advisor sign-off at the Wednesday
meeting — build against them in the meantime rather than stubbing further, per the
advisor's own general instruction, but don't treat them as more final than the
source document does.

- **Uncertainty signal: token entropy.** The sole online gating signal — the only
  option cheap enough to sit in the per-step decoding loop. Optionally aggregated
  over a span or combined with a semantic-uncertainty measure at the *definition*
  level (writing guide §3.3), but semantic entropy itself stays out of the online
  gate; if used at all, it's a single offline correlation study, not part of the
  pipeline.
- **Quantization: 4-bit, all five checkpoints** (2026-08-10, supersedes the earlier
  bf16-only plan). See "Non-negotiable scientific constraints" #2 — this is a hard
  constraint, not a convenience.
- **Merge-back policy: annotate-only first, replace as a later follow-on.** Run the
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
  benchmark the mechanism isn't built to catch. (Still `PROPOSED` pending advisor
  confirmation — advisor said "I'd choose any two" of TruthfulQA/HaluEval/FActScore;
  this is the recommended pair, not yet confirmed in person.)
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
  research project of its own and eat the 3-credit budget. (Also still `PROPOSED`
  pending advisor confirmation, though the advisor's own question — "what other
  structured verifiers are there?" — now has a concrete, citable answer.)
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
2026-07-26), then the **title and RQ wording** were reconciled to the current
five-checkpoint/HaluEval+FActScore/Z3 scope on 2026-08-10 — the code below has *not*
yet been migrated to match. Nothing here is thrown away or wrong; it's
pipeline-mechanics validation that needs re-pointing, not a redo.

**What works today, as built:** end-to-end on CPU against `sshleifer/tiny-gpt2` and
real TruthfulQA data (committed split: 327/163/327 calibration/development/test,
seed 42): token entropy monitor with no-op verification, generation latency
instrumentation (TTFT, inter-token, via `services/neural/src/sense_neural/latency.py`),
split machinery with a leakage guard, the entropy-gating policy that refuses to run
uncalibrated, the truncated-vs-exact entropy bounds check, a symbolic backend
(SPARQL/Wikidata, `services/symbolic`), and the gate-to-symbolic router
(`services/orchestrator`, annotate-only merge-back). All four required correctness
checks pass. All three research-question harnesses (`experiments/`) run end-to-end
on real data: RQ1 (fixed-threshold transfer), RQ2 (self-adaptive threshold), RQ3
(accuracy-latency trade-off — factuality scored via a placeholder lexical-containment
metric, `eval/src/sense_eval/factuality.py`, not the eventual judge). A RAG comparison
baseline (`services/rag`, FAISS + all-MiniLM-L6-v2 over a committed, curated
Wikipedia passage corpus tied to the TruthfulQA test questions) also ran end-to-end
on the full 327-example test split (`experiments/rag_baseline_truthful_qa.py`,
315/327 passages resolved, `factuality_accuracy_proxy` scored via the same
lexical-containment placeholder, accuracy-only — no latency instrumentation).

**What's needed to reach the reconciled scope** (see "Design decisions" for the
rationale behind each):
1. **Models** — add Qwen3 8B/4B/1.7B configs alongside Llama-3 8B Instruct and
   Mistral 7B Instruct v0.3 (`configs/model.yaml`); the GPU run that just completed
   used only the two-model bf16 setup and needs re-pinning to 4-bit across all five.
2. **Quantization** — re-pin the GPU configs from bf16 to 4-bit (constraint #2).
3. **Benchmark** — swap TruthfulQA data/splits for HaluEval + FActScore; TruthfulQA
   stays in the repo as the excluded-benchmark limitation citation, not deleted.
4. **Symbolic backend** — replace `services/symbolic`'s SPARQL/Wikidata client with
   Z3 constraint checking behind an atomic-claim decomposition front-end; keep the
   existing HTTP contract shape, keep SPARQL/Wikidata reachable as the swappable
   fallback rather than deleting it.
5. **RAG baseline** — re-point `services/rag`'s corpus at HaluEval/FActScore
   questions once that data is in place.
6. **Metrics** — `eval/src/sense_eval/factuality.py`'s placeholder lexical-containment
   scorer needs to report hallucination rate, task accuracy, and abstention rate
   together (design-decisions reporting requirement), not accuracy alone.

Not yet built regardless of scope: real-model GPU runs against the reconciled
five-checkpoint lineup (the completed Llama-3/Mistral GPU run predates the scope
reconciliation and used the retired two-model config; current results remain
pipeline-mechanics validation, not scientific findings, until re-run against the
current scope).
