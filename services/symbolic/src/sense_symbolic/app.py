"""Symbolic verification backend HTTP contract.

Two endpoints:
  - POST /entity_exists  — does a labeled entity exist in this backend's domain?
  - POST /verify_triple  — does a (subject, predicate, object) triple hold?

`verified` in the triple response is a tri-state (true / false / null), not boolean:
null means "could not be checked" (an entity/predicate didn't resolve, or the
backend was unreachable/underdetermined) — that is not the same claim as "checked
and found false", and merging those would silently corrupt whatever merge-back
policy consumes this response.

Every response includes `latency_ms`, measured inside this service, feeding the
orchestrator's symbolic-round-trip instrumentation (CLAUDE.md's latency
requirements).

Two backends implement this same contract, selected by `create_app(backend=...)` /
the SENSE_SYMBOLIC_BACKEND env var:
  - "z3" (default, 2026-08-13 scope reconciliation): decompose the triple into an
    AtomicClaim restricted to a fixed relation registry (decomposition.py), then
    autoformalize and check it against a fixed biographical fact base with Z3
    (z3_verifier.py). Local, deterministic, no network.
  - "sparql" (the original backend, kept as a fallback per CLAUDE.md's design
    decisions — not deleted, just no longer the default): live Wikidata lookups
    (wikidata_client.py).
Router.py and the rest of the orchestrator talk to either one identically — they
only ever see this HTTP contract, never which backend answered it.
"""

import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from sense_symbolic.decomposition import decompose_claim
from sense_symbolic.domain import resolve_entity as z3_resolve_entity
from sense_symbolic.wikidata_client import WikidataUnavailableError, ask_triple, search_entity_qid
from sense_symbolic.z3_verifier import verify_claim as z3_verify_claim

DEFAULT_BACKEND = "z3"
SUPPORTED_BACKENDS = frozenset({"z3", "sparql"})


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


def _validate_backend(backend: str) -> str:
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"unknown symbolic backend {backend!r} — expected one of {sorted(SUPPORTED_BACKENDS)}")
    return backend


def create_app(backend: str = DEFAULT_BACKEND) -> FastAPI:
    backend = _validate_backend(backend)
    app = FastAPI(title="sense-symbolic")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "backend": backend}

    @app.post("/entity_exists", response_model=EntityExistsResponse)
    async def entity_exists(request: EntityExistsRequest) -> EntityExistsResponse:
        start = time.perf_counter()

        if backend == "z3":
            qid = z3_resolve_entity(request.label)
        else:
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

        if backend == "z3":
            claim = decompose_claim(request.subject_label, request.predicate_pid, request.object_label)
            if claim is not None:
                subject_qid = z3_resolve_entity(claim.subject_label)
                if claim.relation_kind == "before_year":
                    object_qid = z3_resolve_entity(claim.object_label)
                verified = z3_verify_claim(claim)
        else:
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

    return app


def _default_backend() -> str:
    return _validate_backend(os.environ.get("SENSE_SYMBOLIC_BACKEND", DEFAULT_BACKEND))


app = create_app(_default_backend())
