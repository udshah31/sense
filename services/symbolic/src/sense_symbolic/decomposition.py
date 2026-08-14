"""Decomposition step: turn a (subject_label, predicate_pid, object_label) request
into an AtomicClaim, or refuse.

This runs BEFORE autoformalization (z3_verifier.py) by construction — the proposal's
stated reason for decomposition-first is that the dominant failure mode in
solver-backed NL verification is autoformalization error: a syntactically valid but
semantically wrong translation makes the solver verify the wrong proposition.
Restricting decomposition to a small fixed registry of relation kinds (domain.py's
RELATION_KINDS) is how that surface is narrowed here — a predicate this backend
doesn't recognize is refused at this stage, before any Z3 expression is built from
it, rather than guessed at.

The triple itself is already the atomic unit by construction of the HTTP interface
(services/symbolic/app.py's /verify_triple takes exactly one subject-predicate-object
claim per call) — decomposing a full generation into a sequence of such triples is
the caller's job (the merge-back/routing layer), matching FActScore's atomic-fact
decomposition granularity (CLAUDE.md's design decisions).
"""

from dataclasses import dataclass

from sense_symbolic.domain import RELATION_KINDS


@dataclass(frozen=True)
class AtomicClaim:
    subject_label: str
    relation_kind: str  # domain.RELATION_KINDS value, e.g. "birth_year", "nationality"
    object_label: str


def decompose_claim(subject_label: str, predicate_pid: str, object_label: str) -> AtomicClaim | None:
    """Returns None when predicate_pid isn't in the fixed relation registry —
    "not decomposable", refused before formalization, not silently coerced into
    the nearest known relation.
    """
    relation_kind = RELATION_KINDS.get(predicate_pid)
    if relation_kind is None:
        return None
    return AtomicClaim(subject_label=subject_label, relation_kind=relation_kind, object_label=object_label)
