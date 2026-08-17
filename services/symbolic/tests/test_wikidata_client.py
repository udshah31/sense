"""Tests against the real public Wikidata endpoints — no mocking, since the whole
point of this backend is a live symbolic verification round-trip. Facts used here
are stable/definitional (a Nobel laureate's occupation), not likely to change.

The SPARQL query service (query.wikidata.org) occasionally enters an upstream
outage/aggressive-rate-limit state outside this project's control (observed during
development: "Aggressively rate-limiting to 1 req/min - this rule was created
during active wdqs outage"). ask_triple already retries once honoring Retry-After;
if it's still unavailable after that, these tests skip rather than fail, since that
is a live-service availability signal, not a code defect — WikidataUnavailableError
is exactly the tri-state distinction the backend is built to make (see app.py).
"""

import pytest

from sense_symbolic.wikidata_client import WikidataUnavailableError, ask_triple, search_entity_qid

EINSTEIN_QID = "Q937"
PHYSICIST_QID = "Q169470"
POLITICIAN_QID = "Q82955"
OCCUPATION_PID = "P106"


async def test_search_entity_qid_finds_known_entity():
    try:
        qid = await search_entity_qid("Albert Einstein")
    except WikidataUnavailableError as e:
        pytest.skip(f"Wikidata search service unavailable: {e}")
    assert qid == EINSTEIN_QID


async def test_search_entity_qid_returns_none_for_nonsense_label():
    try:
        qid = await search_entity_qid("xyzzy-not-a-real-entity-qqq123")
    except WikidataUnavailableError as e:
        pytest.skip(f"Wikidata search service unavailable: {e}")
    assert qid is None


async def test_ask_triple_true_case():
    try:
        assert await ask_triple(EINSTEIN_QID, OCCUPATION_PID, PHYSICIST_QID) is True
    except WikidataUnavailableError as e:
        pytest.skip(f"Wikidata SPARQL service unavailable: {e}")


async def test_ask_triple_false_case():
    try:
        assert await ask_triple(EINSTEIN_QID, OCCUPATION_PID, POLITICIAN_QID) is False
    except WikidataUnavailableError as e:
        pytest.skip(f"Wikidata SPARQL service unavailable: {e}")
