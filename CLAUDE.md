# SENSE — Project Context

**SENSE: A Self-Adaptive Neuro-Symbolic Framework for Real-Time Hallucination Mitigation**
Self-adaptive ENtropy-aware Symbolic Engine

Master's Culminating Project (Starred Paper, CSCI699) · Computer Science MS · St. Cloud State University

---

## What this system does

An LLM generates normally. A monitor computes predictive uncertainty at each decoding
step. When uncertainty crosses a **per-model calibrated threshold**, the generation is
routed to a symbolic verification stream instead of being accepted as-is. The symbolic
result is merged back and generation resumes.

The research question is **not** "does entropy gating work." It is **does a gating policy
calibrated on one model transfer to a model from a different family.** Every design
decision serves that question.

### Research questions

- **RQ1** — Does a fixed entropy-gating threshold transfer across model families?
- **RQ2** — Does a self-adaptive threshold that recalibrates per model preserve routing
  quality where a fixed one does not?
- **RQ3** — What accuracy–latency trade-off does routing introduce, and does it hold on
  both models?

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

### 3. Decoding configuration is held constant

Same temperature, same `max_new_tokens`, same sampling strategy across all conditions
within an experiment. Entropy characterization runs use greedy decoding.

### 4. Entropy is not comparable across models in raw form

Llama-3 and Mistral have different tokenizers and different vocabulary sizes. Raw
entropy values are not directly comparable. Normalize by `ln(V)` or compare quantiles.
Read `V` from `tokenizer.vocab_size` at load time — never hard-code it.

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

- Keep model identity in config. Nothing hard-codes Llama-3 or Mistral.
- The test suite runs against a small CPU-friendly model so the full pipeline —
  entropy computation, gate decisions, routing, merge-back, instrumentation — is
  exercised on every change.
- Llama-3 and Mistral runs happen on the GPU environment only, using the identical
  code path.

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

The "real-time" claim in the title rests entirely on these numbers. Instrument from the
start; retrofitting timing after experiments run is how the claim ends up unsupported.

- Measure **per stage**: time-to-first-token, mean inter-token latency, gate evaluation
  cost, symbolic round-trip, merge-back.
- Discard warm-up runs. First-call kernel compilation otherwise dominates.
- Measure inside the container, not from the host.

---

## Design decisions

Decided 2026-07-26 as a working proposal, to unblock implementation. Not yet
confirmed by the actual project advisor — treat as provisional until that
confirmation happens, but build against them in the meantime rather than stubbing
further.

- **Uncertainty signal: token entropy.** The sole online gating signal — the only
  option cheap enough to sit in the per-step decoding loop without breaking the
  real-time claim. Full semantic entropy is out of scope for the gate itself; if used
  at all, it's a single offline correlation study, not part of the pipeline.
- **Merge-back policy: annotate-only first, replace as a later follow-on.** Run the
  RQ1–RQ3 routing/latency experiments on annotate-only so routing cost is measured
  without confounding it with a correction strategy. Add a replace condition only
  once the symbolic backend is stable, to get a mitigation-effect number without
  rushing it ahead of a working backend.
- **Benchmark: TruthfulQA.** Short-form QA suits per-token entropy (long free-form
  generation dilutes the signal), has an established factuality metric, and is the
  most-cited comparison point for hallucination-mitigation work. Do not also stand up
  a second benchmark — one done right beats two done partially.
- **Symbolic backend: SPARQL endpoint against a public KG (Wikidata).** Lightest
  option to stand up behind a stable HTTP contract, easiest to stub/mock for CPU
  testing, predictable round-trip latency for the instrumentation story. KG-Trie
  constrained decoding, a hand-authored logical constraint checker, and a structured
  verifier model were all considered and rejected as out of scope for this project's
  timeline (constrained decoding is a paper on its own; a verifier model blurs into
  semantic-entropy territory and adds its own latency).
- **Baseline RAG stack: FAISS + all-MiniLM-L6-v2 + a small Wikipedia passage subset.**
  In-memory vector store (no server), small CPU-friendly embedding model, modest
  reproducible corpus. This is a comparison baseline, not the contribution — capped
  effort on purpose.

Build against these decisions as if final. If the real advisor changes one, treat it
as a design change, not a correction — update this file and the affected code
together, don't leave the doc stale.

---

## Current status

Phase 0. Repo scaffolded, all-Python stack, design decisions above committed as a
working proposal pending real advisor confirmation. Working end-to-end on CPU
against `sshleifer/tiny-gpt2` and real TruthfulQA data (committed split:
327/163/327 calibration/development/test, seed 42): token entropy monitor with
no-op verification, generation latency instrumentation (TTFT, inter-token, via
`services/neural/src/sense_neural/latency.py`), split machinery with a leakage
guard, the entropy-gating policy that refuses to run uncalibrated, the
truncated-vs-exact entropy bounds check, a symbolic backend (SPARQL/Wikidata,
`services/symbolic`), and the gate-to-symbolic router (`services/orchestrator`,
annotate-only merge-back). All four required correctness checks pass. All three
research question harnesses (`experiments/`) run end-to-end on real data: RQ1
(fixed-threshold transfer), RQ2 (self-adaptive threshold), RQ3 (accuracy-latency
trade-off — factuality scored via a placeholder lexical-containment metric,
`eval/src/sense_eval/factuality.py`, not the eventual judge). A RAG comparison
baseline (`services/rag`, FAISS + all-MiniLM-L6-v2 over a committed, curated
Wikipedia passage corpus tied to the TruthfulQA test questions) also ran
end-to-end on the full 327-example test split (`experiments/rag_baseline_truthful_qa.py`,
315/327 passages resolved, `factuality_accuracy_proxy` scored via the same
lexical-containment placeholder, accuracy-only — no latency instrumentation, per
CLAUDE.md's design decisions). Not yet built: Llama-3/Mistral runs on GPU (the
current results are pipeline-mechanics validation on untrained tiny models, not
scientific findings).
