"""Tests for the "sparql" backend — kept as a fallback (CLAUDE.md's design
decisions), not deleted when z3 became the default (2026-08-13 scope
reconciliation). Hits the real public Wikidata endpoints — no mocking, since the
whole point of this backend is a live symbolic verification round-trip. Facts
used here are stable/definitional (a Nobel laureate's occupation), not likely to
change.

See test_app_z3.py for the default backend's tests (local, deterministic, no
network) and test_app_backend_selection.py for the create_app()/env-var wiring
shared by both.
"""

import pytest
from fastapi.testclient import TestClient

from sense_symbolic.app import create_app
from sense_symbolic.wikidata_client import WikidataUnavailableError, search_entity_qid

client = TestClient(create_app("sparql"))

EINSTEIN = "Albert Einstein"
PHYSICIST = "physicist"
POLITICIAN = "politician"
OCCUPATION_PID = "P106"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "backend": "sparql"}


async def test_entity_exists_for_known_entity():
    # /entity_exists swallows WikidataUnavailableError into qid=None (same as a
    # genuine "not found") — indistinguishable from the response alone, so probe
    # the live service directly first to tell "service down" from "code broke".
    try:
        await search_entity_qid(EINSTEIN)
    except WikidataUnavailableError as e:
        pytest.skip(f"Wikidata search service unavailable: {e}")

    response = client.post("/entity_exists", json={"label": EINSTEIN})
    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is True
    assert body["qid"] == "Q937"
    assert body["latency_ms"] > 0


def test_entity_exists_for_unknown_entity():
    response = client.post("/entity_exists", json={"label": "xyzzy-not-a-real-entity-qqq123"})
    assert response.status_code == 200
    body = response.json()
    assert body["exists"] is False
    assert body["qid"] is None


def test_verify_triple_true_case():
    response = client.post(
        "/verify_triple",
        json={"subject_label": EINSTEIN, "predicate_pid": OCCUPATION_PID, "object_label": PHYSICIST},
    )
    assert response.status_code == 200
    body = response.json()
    # Skip before asserting subject_qid — a search-service outage leaves it None
    # too (not just verified), and that's "service down", not a code defect.
    if body["verified"] is None:
        pytest.skip("Wikidata SPARQL service unavailable (entity search or ASK query failed)")
    assert body["subject_qid"] == "Q937"
    assert body["latency_ms"] > 0
    assert body["verified"] is True


def test_verify_triple_false_case():
    response = client.post(
        "/verify_triple",
        json={"subject_label": EINSTEIN, "predicate_pid": OCCUPATION_PID, "object_label": POLITICIAN},
    )
    assert response.status_code == 200
    body = response.json()
    if body["verified"] is None:
        pytest.skip("Wikidata SPARQL service unavailable (entity search or ASK query failed)")
    assert body["verified"] is False


def test_verify_triple_unknown_subject_yields_null_not_false():
    response = client.post(
        "/verify_triple",
        json={
            "subject_label": "xyzzy-not-a-real-entity-qqq123",
            "predicate_pid": OCCUPATION_PID,
            "object_label": PHYSICIST,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verified"] is None  # unresolved subject, not "checked and false"
    assert body["subject_qid"] is None
