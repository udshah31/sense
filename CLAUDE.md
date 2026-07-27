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

### Language split — this is forced, not stylistic

| Component | Language | Why |
|---|---|---|
| Entropy monitor, model loading, probes | **Python** | Per-step logits and hidden states are only reachable in-process via Hugging Face Transformers |
| Gate policy, routing, symbolic interface, orchestration | **TypeScript** | Clean separation of the research contribution from the model plumbing |
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
│   └── orchestrator/         # TypeScript: gate policy, routing, instrumentation
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
- Pinned Python and Node dependencies.
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

## Open decisions — do not guess these

These are unresolved and pending advisor input. If a task requires one, stop and ask
rather than picking a default and proceeding.

- **Symbolic backend** — KG-Trie constrained decoding, SPARQL endpoint, logical
  constraint checker, or structured verifier.
- **Benchmark set** — TruthfulQA, HaluEval, FActScore-style, or knowledge-grounded QA.
- **Uncertainty signal** — token entropy, full sampled semantic entropy, or
  probe-based approximation. Full semantic entropy requires an additional NLI model
  in the stack.
- **Merge-back policy** — accept, replace, abstain, or annotate. Until decided, the
  harness annotates only, so routing cost is measured without confounding it with an
  arbitrary correction strategy.
- **Baseline RAG stack** — vector store, embedding model, corpus.

Where a component depends on an open decision, build against an interface and provide
a stub implementation clearly marked as such.

---

## Current status

Phase 0 — environment bootstrap. Nothing implemented yet.
