Symbolic verification backend: SPARQL endpoint against Wikidata (see `../../CLAUDE.md`'s
design decisions), behind a FastAPI HTTP contract.

## Endpoints

- `GET /health`
- `POST /entity_exists {label}` — does a labeled entity exist in Wikidata?
- `POST /verify_triple {subject_label, predicate_pid, object_label}` — does the
  triple hold? `verified` is tri-state (`true` / `false` / `null`) — `null` means
  "could not be checked" (a label didn't resolve, or Wikidata was unreachable), which
  is a different claim from "checked and found false" and must not be collapsed into
  it by anything consuming this response.

Every response includes `latency_ms`, measured inside this service, for the
orchestrator's symbolic-round-trip instrumentation (CLAUDE.md's latency requirements).

## Setup

```
uv sync
```

## Run

```
uv run uvicorn sense_symbolic.app:app --reload
```

## Tests

```
uv run pytest
```

Tests hit the real public Wikidata endpoints (no mocking — the point of this backend
is a live round-trip) using stable, definitional facts (a Nobel laureate's
occupation). The SPARQL query service occasionally enters an upstream
outage/aggressive-rate-limit state outside this project's control; `ask_triple`
retries once honoring `Retry-After`, and tests skip (rather than fail) if it's still
unavailable after that — that's a live-service signal, not a code defect.

## Known limitation

`search_entity_qid` resolves labels via Wikidata's `wbsearchentities` (exact
label match against English labels), not full entity linking / disambiguation.
Ambiguous or non-English labels may resolve to the wrong entity or fail to resolve.
Good enough for stable, well-known entities; revisit if the merge-back "replace"
condition needs it to be more robust.
