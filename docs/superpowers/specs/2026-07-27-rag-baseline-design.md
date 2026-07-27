# RAG Baseline — Design Spec

Date: 2026-07-27
Status: Approved for planning

## Purpose

SENSE's Phase-0 design decisions (CLAUDE.md) name a RAG stack — FAISS +
`all-MiniLM-L6-v2` + a small Wikipedia passage subset — as the comparison
baseline against the entropy-gated symbolic pipeline. This spec covers
building that baseline end-to-end and running it once on the TruthfulQA test
split, producing a factuality number alongside the RQ1–RQ3 results.

This is explicitly **not** the project's contribution. Per CLAUDE.md: "This is
a comparison baseline, not the contribution — capped effort on purpose." No
latency instrumentation, no gate integration, no second benchmark.

## Non-goals

- No latency measurement of any kind (TTFT, retrieval time, etc.). The
  "real-time" claim in the paper title rests on the gated pipeline (RQ3), not
  on this baseline. Adding instrumentation here would be scope creep into a
  component intentionally capped.
- No interaction with the entropy gate, symbolic backend, or orchestrator.
  RAG is a standalone generation condition, comparable in spirit to RQ3's
  "ungated" condition but with retrieved context prepended.
- No live network calls during the scored experiment run. Wikipedia is
  queried once, offline, to build a committed corpus file — not at
  experiment time.
- No second benchmark dataset. Runs on the same TruthfulQA test split
  (327 examples, seed 42) already committed for RQ1–RQ3.

## Architecture

### New package: `services/rag/`

Mirrors the existing `services/symbolic/` package structure (pyproject.toml,
`src/sense_rag/`, `tests/`).

**`sense_rag/index.py`** — loads a committed passages file, embeds each
passage with `all-MiniLM-L6-v2` (via `sentence-transformers`), and builds an
in-memory FAISS `IndexFlatIP` index over L2-normalized embeddings (cosine
similarity via inner product). No server process — matches the CLAUDE.md
decision that this is an in-memory vector store, not a service.

**`sense_rag/retrieve.py`** — `retrieve(query: str, k: int = 3) -> list[str]`.
Embeds the query with the same model, searches the FAISS index, returns the
top-k passage texts in ranked order.

### Corpus build script: `services/rag/scripts/build_corpus.py`

Run once, offline, not part of the experiment run path.

For each of the 327 TruthfulQA test-split questions (loaded via
`sense_data.truthful_qa.load_truthful_qa()` + `sense_data.splits.load_splits()`,
same fixed index order as RQ1–RQ3 use), calls the Wikipedia REST API search
endpoint with the question text, takes the top search hit, fetches its intro
extract, and writes one `{title, text}` passage per question to
`data/rag_corpus/passages.json`. That file is committed to the repo, so the
corpus is reproducible without re-hitting Wikipedia. (Some questions may
resolve to the same Wikipedia page; duplicate passages are fine — the corpus
does not need to be deduplicated beyond what naturally happens.)

### Config: `configs/rag.yaml`

Config-driven, following the `rq3.yaml` pattern — no parameters hard-coded
into the experiment script:

```yaml
model: cpu_test       # key into configs/model.yaml, same as rq3.yaml
top_k: 3
corpus_path: data/rag_corpus/passages.json
eval_split: test
```

### Experiment script: `experiments/rag_baseline_truthful_qa.py`

Follows the shape of `experiments/rq3_accuracy_latency_truthful_qa.py`
(`_common` helpers for model loading and config loading, same result-writing
convention).

For each example in the test split:
1. Retrieve top-`k` passages for the question via `sense_rag.retrieve`.
2. Build the prompt:
   ```
   Context: {p1}
   {p2}
   {p3}

   Question: {question}
   ```
3. Generate greedily with the same decoding config used across RQ1–RQ3
   (`configs/gate.yaml`'s `decoding` block) — same model (`tiny-gpt2`), same
   temperature/`max_new_tokens`/sampling strategy, per CLAUDE.md's decoding-
   configuration-held-constant rule.
4. Score the generated text with `sense_eval.factuality.lexical_containment_verdict`
   (the same placeholder proxy RQ3 uses — not a paper-grade judge).

Writes `results/rag_baseline_truthful_qa_{model}.json`: per-example records
(index, retrieved passage titles, factuality verdict, generated text) plus an
aggregate `factuality_accuracy_proxy` and metadata (model name, hf_repo,
revision, decoding config, corpus size) matching the metadata shape RQ1–RQ3
results already use.

## Data flow

```
TruthfulQA test split (327 examples, fixed index order)
        │
        ▼ (offline, one-time)
build_corpus.py ──> Wikipedia REST API ──> data/rag_corpus/passages.json (committed)
        │
        ▼ (experiment run)
passages.json ──embed (MiniLM)──> FAISS index (in-memory)
        │
question ──embed (MiniLM)──> retrieve(k=3) ──> context block
        │
context + question ──prompt──> tiny-gpt2 generate (greedy) ──> text
        │
text ──lexical_containment_verdict──> factuality label
        │
        ▼
results/rag_baseline_truthful_qa_tiny_gpt2.json
```

## Testing (CPU-only, per the dev-environment rule)

- **`build_corpus` parsing**: unit test with a mocked Wikipedia API response,
  asserting the `{title, text}` extraction is correct. No live network calls
  in tests.
- **`retrieve()`**: unit test against a small fixture corpus (3-5 known
  passages), asserting top-k ordering matches expected similarity ranking.
- **Integration test**: runs the full experiment script end-to-end against a
  2-3 example subset with a fixture corpus (not the full 327-example real
  corpus, and not live Wikipedia), asserting the script produces a
  well-formed results JSON with the expected fields.

## Error handling

- If the Wikipedia search API returns no hit for a question during corpus
  build, log the question and skip it — the corpus can have fewer than 327
  passages; the retriever still works over whatever was successfully built.
  This is a one-time, human-reviewed build step, not a runtime path that
  needs silent fallback logic.
- No retry/fallback logic needed in `retrieve()` or the experiment script
  itself, since they never touch the network — only the offline build script
  does, and its failure mode (skip + log) is enough for a capped-effort
  baseline.

## Open items for the implementation plan

None — this design is fully scoped: corpus source (Wikipedia REST API tied to
test-split topics), model (tiny-gpt2, matching RQ1–RQ3), prompt format (simple
prepended context block), top-k (3), and split (test only) were all decided
during brainstorming.
