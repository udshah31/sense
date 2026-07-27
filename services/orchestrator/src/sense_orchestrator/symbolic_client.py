"""HTTP client for the symbolic verification backend (services/symbolic).

Talks to it only over HTTP, per the architecture split — no in-process import of
sense_symbolic at runtime (it is a dev/test-only dependency, used solely to give
router tests a real in-process response without a live server).
"""

import httpx

DEFAULT_TIMEOUT_SECONDS = 10.0


class SymbolicBackendError(Exception):
    """Raised when the symbolic backend request itself fails (network/HTTP error) —
    distinct from a successful response whose `verified` field is null."""


async def verify_triple(
    base_url: str,
    subject_label: str,
    predicate_pid: str,
    object_label: str,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Call POST {base_url}/verify_triple and return its JSON body verbatim."""
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)
    try:
        response = await client.post(
            f"{base_url}/verify_triple",
            json={
                "subject_label": subject_label,
                "predicate_pid": predicate_pid,
                "object_label": object_label,
            },
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise SymbolicBackendError(str(e)) from e
    finally:
        if owns_client:
            await client.aclose()
