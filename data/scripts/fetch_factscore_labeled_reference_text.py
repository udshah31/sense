"""One-time generation of ../fixtures/factscore_labeled/wikipedia_reference_text.json
— the reference text sense_data.factscore_labeled.load_factscore_labeled pairs with
each of the FActScore paper's 183 labeled entities.

The labeled release itself (InstructGPT.jsonl/ChatGPT.jsonl/PerplexityAI.jsonl, see
that module's docstring for provenance) carries no reference text — the original
paper verified against a 20GB `enwiki-20230401.db` snapshot too large to fetch or
commit here. This fetches each topic's current Wikipedia plain-text extract via the
public MediaWiki API instead, as a documented, honest substitute — see the loader
module's docstring for the resulting limitation (2026 snapshot, not 2023).

Live, rate-limited fetch (backoff: the API returns 429 past roughly one request per
second unauthenticated) — run only to regenerate the committed JSON files, same
one-time-generation contract as generate_factscore_splits.py. Every other part of
the project reads the committed output, never calls this at run time.

Usage: uv run python scripts/fetch_factscore_labeled_reference_text.py
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from sense_data.factscore_labeled import MODELS

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "factscore_labeled"
REFERENCE_TEXT_PATH = FIXTURES_DIR / "wikipedia_reference_text.json"
FAILURES_PATH = FIXTURES_DIR / "wikipedia_fetch_failures.json"

USER_AGENT = "SENSE-research-project/1.0 (academic use; St. Cloud State University CSCI699)"
REQUEST_DELAY_SECONDS = 1.0
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 3.0


def _topics_from_labeled_files() -> list[str]:
    topics = set()
    for model in MODELS:
        for line in (FIXTURES_DIR / f"{model}.jsonl").read_text().splitlines():
            if line.strip():
                topics.add(json.loads(line)["topic"])
    return sorted(topics)


def _fetch_extract(topic: str) -> str | None:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "titles": topic, "prop": "extracts", "explaintext": 1, "format": "json", "redirects": 1}
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.load(response)
            page = next(iter(data["query"]["pages"].values()))
            if "missing" in page or not page.get("extract"):
                return None
            return page["extract"]
        except urllib.error.HTTPError:
            if attempt == MAX_ATTEMPTS - 1:
                return None
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
    return None


def main() -> None:
    topics = _topics_from_labeled_files()
    results: dict[str, str] = {}
    failures: list[str] = []

    for topic in topics:
        extract = _fetch_extract(topic)
        if extract is None:
            failures.append(topic)
        else:
            results[topic] = extract
        time.sleep(REQUEST_DELAY_SECONDS)

    REFERENCE_TEXT_PATH.write_text(json.dumps({t: {"extract": e} for t, e in results.items()}, indent=2))
    FAILURES_PATH.write_text(json.dumps(failures, indent=2))
    print(f"resolved {len(results)}/{len(topics)} topics, {len(failures)} failed: {failures}")


if __name__ == "__main__":
    main()
