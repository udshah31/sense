"""One-time, offline corpus build: queries the Wikipedia REST API for each
TruthfulQA test-split question and writes the resolved passages to
data/rag_corpus/passages.json, committed to the repo.

Not part of any automated test-suite execution path or experiment run path —
sense_rag.index and sense_rag.retrieve only ever read the committed output file.
Run by hand: `uv run python scripts/build_corpus.py`.
"""

import json
import sys
import time
from pathlib import Path

import httpx

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_HEADERS = {"User-Agent": "sense-rag/0.1 (CSCI699 research project; no contact URL yet)"}

REPO_ROOT = Path(__file__).parent.parent.parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "rag_corpus" / "passages.json"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "truthful_qa.json"


def fetch_passage(question: str, client: httpx.Client) -> dict | None:
    """Search Wikipedia for `question`, return the top hit's {title, text}
    intro extract, or None if nothing resolved."""
    for attempt in range(6):
        response = client.get(
            WIKIPEDIA_API_URL,
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": question,
                "gsrlimit": 1,
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
            },
            headers=REQUEST_HEADERS,
        )
        if response.status_code == 429:
            wait = float(response.headers.get("retry-after", 2 * (attempt + 1)))
            time.sleep(wait)
            continue
        response.raise_for_status()
        break
    else:
        response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    extract = page.get("extract")
    if not extract:
        return None

    return {"title": page["title"], "text": extract}


def build_corpus(questions: list[str], client: httpx.Client) -> list[dict]:
    """Resolve each question to a passage, skipping (and logging) any that
    don't resolve. Returns the list of resolved {title, text} passages."""
    passages = []
    for question in questions:
        passage = fetch_passage(question, client)
        if passage is None:
            print(f"skipping question with no Wikipedia hit: {question!r}", file=sys.stderr)
            continue
        passages.append(passage)
        time.sleep(0.2)
    return passages


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "data" / "src"))
    from sense_data.splits import load_splits
    from sense_data.truthful_qa import load_truthful_qa

    examples = load_truthful_qa()
    splits = load_splits(SPLIT_PATH)
    questions = [examples[i].question for i in splits.test]

    # Force IPv4: this network environment's IPv6 route to Wikipedia's edge is
    # rate-limited far more aggressively than IPv4 (confirmed via curl -6 vs -4),
    # which otherwise causes long 429/retry-after stalls or an eventual raise.
    transport = httpx.HTTPTransport(local_address="0.0.0.0")
    with httpx.Client(timeout=10.0, transport=transport) as client:
        passages = build_corpus(questions, client)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(json.dumps(passages, indent=2))
    print(f"wrote {len(passages)} passages (of {len(questions)} questions) to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
