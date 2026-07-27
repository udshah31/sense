import pytest
from fastapi.testclient import TestClient

from sense_symbolic.app import app

client = TestClient(app)

EINSTEIN = "Albert Einstein"
PHYSICIST = "physicist"
POLITICIAN = "politician"
OCCUPATION_PID = "P106"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_entity_exists_for_known_entity():
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
    assert body["subject_qid"] == "Q937"
    assert body["latency_ms"] > 0
    if body["verified"] is None:
        pytest.skip("Wikidata SPARQL service unavailable (both entities resolved, ASK query failed)")
    assert body["verified"] is True


def test_verify_triple_false_case():
    response = client.post(
        "/verify_triple",
        json={"subject_label": EINSTEIN, "predicate_pid": OCCUPATION_PID, "object_label": POLITICIAN},
    )
    assert response.status_code == 200
    body = response.json()
    if body["verified"] is None and body["subject_qid"] is not None and body["object_qid"] is not None:
        pytest.skip("Wikidata SPARQL service unavailable (both entities resolved, ASK query failed)")
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
