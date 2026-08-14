"""Tests for the "z3" backend — the default (2026-08-13 scope reconciliation,
supersedes SPARQL/Wikidata as CLAUDE.md's default symbolic backend). Local,
deterministic fixed fact base (domain.py) — no network, no mocking needed.
"""

from fastapi.testclient import TestClient

from sense_symbolic.app import app, create_app

client = TestClient(create_app("z3"))

EINSTEIN = "Albert Einstein"
NEWTON = "Isaac Newton"
PHYSICIST = "physicist"
POLITICIAN = "politician"
OCCUPATION_PID = "P106"
BIRTH_YEAR_PID = "P569"
NATIONALITY_PID = "P27"
UNSUPPORTED_PID = "P9999999"  # not in domain.RELATION_KINDS
ARTHURS_MAGAZINE = "Arthur's Magazine"
FIRST_FOR_WOMEN = "First for Women"


def test_default_app_uses_z3_backend():
    response = TestClient(app).get("/health")
    assert response.json() == {"status": "ok", "backend": "z3"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "backend": "z3"}


def test_entity_exists_for_known_entity():
    response = client.post("/entity_exists", json={"label": EINSTEIN})
    body = response.json()
    assert body["exists"] is True
    assert body["qid"] == "albert einstein"


def test_entity_exists_is_case_insensitive():
    response = client.post("/entity_exists", json={"label": "ALBERT EINSTEIN"})
    assert response.json()["exists"] is True


def test_entity_exists_for_unknown_entity():
    response = client.post("/entity_exists", json={"label": "xyzzy-not-a-real-entity-qqq123"})
    body = response.json()
    assert body["exists"] is False
    assert body["qid"] is None


def test_verify_triple_occupation_true_case():
    response = client.post(
        "/verify_triple",
        json={"subject_label": EINSTEIN, "predicate_pid": OCCUPATION_PID, "object_label": PHYSICIST},
    )
    body = response.json()
    assert body["subject_qid"] == "albert einstein"
    assert body["verified"] is True
    assert body["latency_ms"] > 0


def test_verify_triple_occupation_false_case():
    response = client.post(
        "/verify_triple",
        json={"subject_label": EINSTEIN, "predicate_pid": OCCUPATION_PID, "object_label": POLITICIAN},
    )
    assert response.json()["verified"] is False


def test_verify_triple_birth_year_true_case_via_z3_entailment():
    response = client.post(
        "/verify_triple",
        json={"subject_label": NEWTON, "predicate_pid": BIRTH_YEAR_PID, "object_label": "1643"},
    )
    assert response.json()["verified"] is True


def test_verify_triple_birth_year_false_case_via_z3_entailment():
    response = client.post(
        "/verify_triple",
        json={"subject_label": NEWTON, "predicate_pid": BIRTH_YEAR_PID, "object_label": "1900"},
    )
    assert response.json()["verified"] is False


def test_verify_triple_birth_year_unparsable_object_yields_null():
    response = client.post(
        "/verify_triple",
        json={"subject_label": NEWTON, "predicate_pid": BIRTH_YEAR_PID, "object_label": "a long time ago"},
    )
    assert response.json()["verified"] is None


def test_verify_triple_before_year_true_case_via_z3_entailment():
    # Reproduces the HaluEval qa-style claim this relation kind was added for:
    # "Arthur's Magazine (1844) started before First for Women (1989)."
    response = client.post(
        "/verify_triple",
        json={
            "subject_label": ARTHURS_MAGAZINE,
            "predicate_pid": "BEFORE_YEAR",
            "object_label": FIRST_FOR_WOMEN,
        },
    )
    body = response.json()
    assert body["subject_qid"] == "arthur's magazine"
    assert body["object_qid"] == "first for women"
    assert body["verified"] is True


def test_verify_triple_before_year_false_case_via_z3_entailment():
    response = client.post(
        "/verify_triple",
        json={
            "subject_label": FIRST_FOR_WOMEN,
            "predicate_pid": "BEFORE_YEAR",
            "object_label": ARTHURS_MAGAZINE,
        },
    )
    assert response.json()["verified"] is False


def test_verify_triple_unknown_subject_yields_null_not_false():
    response = client.post(
        "/verify_triple",
        json={
            "subject_label": "xyzzy-not-a-real-entity-qqq123",
            "predicate_pid": OCCUPATION_PID,
            "object_label": PHYSICIST,
        },
    )
    body = response.json()
    assert body["verified"] is None
    assert body["subject_qid"] is None


def test_verify_triple_unsupported_predicate_yields_null_not_a_guess():
    # decomposition.py refuses to build an AtomicClaim for a predicate outside the
    # fixed relation registry — this must not fall through to any default guess.
    response = client.post(
        "/verify_triple",
        json={"subject_label": EINSTEIN, "predicate_pid": UNSUPPORTED_PID, "object_label": PHYSICIST},
    )
    body = response.json()
    assert body["verified"] is None
    assert body["subject_qid"] is None  # decomposition failed before entity resolution even ran


def test_verify_triple_nationality_true_case():
    response = client.post(
        "/verify_triple",
        json={"subject_label": EINSTEIN, "predicate_pid": NATIONALITY_PID, "object_label": "German"},
    )
    assert response.json()["verified"] is True
