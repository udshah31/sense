"""Symbolic verification backend HTTP contract.

Two endpoints, both wrapping public Wikidata lookups:
  - POST /entity_exists  — does a labeled entity exist in Wikidata?
  - POST /verify_triple  — does a (subject, predicate, object) triple hold?

`verified` in the triple response is a tri-state (true / false / null), not boolean:
null means "could not be checked" (a label didn't resolve, or Wikidata was
unreachable) — that is not the same claim as "checked and found false", and merging
those would silently corrupt whatever merge-back policy consumes this response.

Every response includes `latency_ms`, measured inside this service, feeding the
orchestrator's symbolic-round-trip instrumentation (CLAUDE.md's latency
requirements).
"""

import time

from fastapi import FastAPI
from pydantic import BaseModel

from sense_symbolic.wikidata_client import WikidataUnavailableError, ask_triple, search_entity_qid

app = FastAPI(title="sense-symbolic")


class EntityExistsRequest(BaseModel):
    label: str


class EntityExistsResponse(BaseModel):
    label: str
    exists: bool
    qid: str | None
    latency_ms: float


class VerifyTripleRequest(BaseModel):
    subject_label: str
    predicate_pid: str  # Wikidata property ID, e.g. "P27" (country of citizenship)
    object_label: str


class VerifyTripleResponse(BaseModel):
    subject_label: str
    predicate_pid: str
    object_label: str
    subject_qid: str | None
    object_qid: str | None
    verified: bool | None  # None = could not be checked, not "checked and false"
    latency_ms: float


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/entity_exists", response_model=EntityExistsResponse)
async def entity_exists(request: EntityExistsRequest) -> EntityExistsResponse:
    start = time.perf_counter()
    try:
        qid = await search_entity_qid(request.label)
    except WikidataUnavailableError:
        qid = None
    latency_ms = (time.perf_counter() - start) * 1000

    return EntityExistsResponse(label=request.label, exists=qid is not None, qid=qid, latency_ms=latency_ms)


@app.post("/verify_triple", response_model=VerifyTripleResponse)
async def verify_triple(request: VerifyTripleRequest) -> VerifyTripleResponse:
    start = time.perf_counter()
    verified: bool | None = None
    subject_qid: str | None = None
    object_qid: str | None = None

    try:
        subject_qid = await search_entity_qid(request.subject_label)
        object_qid = await search_entity_qid(request.object_label)
        if subject_qid is not None and object_qid is not None:
            verified = await ask_triple(subject_qid, request.predicate_pid, object_qid)
    except WikidataUnavailableError:
        verified = None

    latency_ms = (time.perf_counter() - start) * 1000

    return VerifyTripleResponse(
        subject_label=request.subject_label,
        predicate_pid=request.predicate_pid,
        object_label=request.object_label,
        subject_qid=subject_qid,
        object_qid=object_qid,
        verified=verified,
        latency_ms=latency_ms,
    )
