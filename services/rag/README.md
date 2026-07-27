# RAG baseline

FAISS + `all-MiniLM-L6-v2` retrieval-augmented-generation comparison baseline
(see `../../CLAUDE.md`'s Phase-0 design decisions). This is a comparison
baseline, not the project's contribution — accuracy-only, no latency
instrumentation, no gate interaction.

## Modules

- `sense_rag.index` — builds an in-memory FAISS index from a committed
  passages file.
- `sense_rag.retrieve` — embeds a query and returns the top-k passage texts.

## Corpus

`data/rag_corpus/passages.json` is built once, offline, via
`scripts/build_corpus.py` (hits the Wikipedia REST API) and committed to the
repo. Neither `sense_rag.index` nor `sense_rag.retrieve` ever calls the
network — they only read the committed file.

## Setup

```
uv sync
```

## Tests

```
uv run pytest
```
