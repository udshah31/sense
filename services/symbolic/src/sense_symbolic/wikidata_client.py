"""Thin client for the two public Wikidata HTTP endpoints this backend uses:
wbsearchentities (label -> QID resolution) and the SPARQL query service (ASK
queries for triple verification).
"""

import asyncio

import httpx

SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"

# Both are public, unauthenticated, rate-limited endpoints. A short timeout keeps a
# slow/unavailable Wikidata from stalling the gated pipeline's symbolic round-trip.
DEFAULT_TIMEOUT_SECONDS = 5.0

# Wikidata's user-agent policy (https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy)
# blocks generic/default user agents with a 403 — a descriptive one identifying the
# project is required.
REQUEST_HEADERS = {"User-Agent": "sense-symbolic/0.1 (CSCI699 research project; no contact URL yet)"}

# Wikidata occasionally rate-limits (429) with a Retry-After header, especially
# during upstream query-service incidents. One retry, honoring Retry-After up to a
# cap, keeps the backend resilient without letting a single request stall the
# pipeline indefinitely.
MAX_RETRY_AFTER_SECONDS = 70.0


class WikidataUnavailableError(Exception):
    """Raised when a Wikidata request fails or times out. Distinct from a normal
    'not found' or 'not verified' result — this means the backend itself couldn't
    be reached, not that the claim is false."""


async def _get_with_retry(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    try:
        response = await client.get(url, **kwargs)
        response.raise_for_status()
        return response
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 429:
            raise WikidataUnavailableError(str(e)) from e
        retry_after = min(float(e.response.headers.get("Retry-After", 1)), MAX_RETRY_AFTER_SECONDS)
        await asyncio.sleep(retry_after)
        try:
            response = await client.get(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPError as retry_error:
            raise WikidataUnavailableError(str(retry_error)) from retry_error
    except httpx.HTTPError as e:
        raise WikidataUnavailableError(str(e)) from e


async def search_entity_qid(label: str, client: httpx.AsyncClient | None = None) -> str | None:
    """Resolve a label to a Wikidata QID via exact-match search. Returns None if no
    entity with that label is found — not an error, just "unknown to Wikidata".
    """
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
        "limit": 5,
    }
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        response = await _get_with_retry(client, SEARCH_URL, params=params, headers=REQUEST_HEADERS)
    finally:
        if owns_client:
            await client.aclose()

    results = response.json().get("search", [])
    for result in results:
        if result.get("label", "").lower() == label.lower():
            return result["id"]
    return results[0]["id"] if results else None


async def ask_triple(
    subject_qid: str, predicate_pid: str, object_qid: str, client: httpx.AsyncClient | None = None
) -> bool:
    """SPARQL ASK: does wd:subject_qid wdt:predicate_pid wd:object_qid hold?"""
    query = f"ASK {{ wd:{subject_qid} wdt:{predicate_pid} wd:{object_qid} . }}"
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        response = await _get_with_retry(
            client,
            SPARQL_URL,
            params={"query": query},
            headers={**REQUEST_HEADERS, "Accept": "application/sparql-results+json"},
        )
    finally:
        if owns_client:
            await client.aclose()

    return bool(response.json()["boolean"])
