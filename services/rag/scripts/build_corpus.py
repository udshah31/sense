"""One-time, offline corpus build: queries the Wikipedia REST API for each
HaluEval test-split question and writes the resolved passages to
data/rag_corpus/passages.json, committed to the repo.

Re-pointed from TruthfulQA to HaluEval (2026-08-13 scope reconciliation) — same
retrieval architecture (live Wikipedia search resolved once, offline FAISS lookup
at run time), only the question source changed. HaluEval's own `knowledge` field
would also work as a corpus directly, but going through Wikipedia search keeps
the baseline an honest "can retrieval find *a* relevant passage" test rather than
handing the model the exact source passage HaluEval's questions were written
from, which would trivially inflate its accuracy relative to the gated pipeline.

Not part of any automated test-suite execution path or experiment run path —
sense_rag.index and sense_rag.retrieve only ever read the committed output file.
Run by hand: `uv run python scripts/build_corpus.py [split]` (default `test`; the
gated-retrieval condition evaluates on `development`, so it needs
`build_corpus.py development` -> data/rag_corpus/passages_development.json).
"""

import json
import sys
import time
from pathlib import Path

import httpx

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_HEADERS = {"User-Agent": "sense-rag/0.1 (CSCI699 research project; no contact URL yet)"}

REPO_ROOT = Path(__file__).parent.parent.parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "rag_corpus"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "halueval.json"


def fetch_passage(question: str, client: httpx.Client) -> dict | None:
    """Search Wikipedia for `question`, return the top hit's {title, text}
    intro extract, or None if nothing resolved."""
    for attempt in range(6):
        try:
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
        except httpx.TransportError:
            # A timeout/connection reset used to abort the whole ~1,000-question build
            # (it did, after 150 questions); retry with backoff like a 429. Only the
            # last attempt's failure propagates.
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))
            continue
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
    for n, question in enumerate(questions, 1):
        if n % 50 == 0:
            print(f"resolved {len(passages)} passages from {n - 1}/{len(questions)} questions", file=sys.stderr, flush=True)
        passage = fetch_passage(question, client)
        if passage is None:
            print(f"skipping question with no Wikipedia hit: {question!r}", file=sys.stderr)
            continue
        passages.append(passage)
        time.sleep(0.2)
    return passages


def corpus_path(split: str) -> Path:
    """`test` keeps the original passages.json (the always-on RAG baseline reads it);
    any other split gets its own file so the two corpora can never overwrite each other."""
    return CORPUS_DIR / ("passages.json" if split == "test" else f"passages_{split}.json")


def main(split: str = "test") -> None:
    sys.path.insert(0, str(REPO_ROOT / "data" / "src"))
    from sense_data.halueval import load_halueval
    from sense_data.splits import load_per_checkpoint_splits

    examples = load_halueval()
    splits = load_per_checkpoint_splits(SPLIT_PATH)
    questions = [examples[i].question for i in getattr(splits, split)]

    # Force IPv4: this network environment's IPv6 route to Wikipedia's edge is
    # rate-limited far more aggressively than IPv4 (confirmed via curl -6 vs -4),
    # which otherwise causes long 429/retry-after stalls or an eventual raise.
    transport = httpx.HTTPTransport(local_address="0.0.0.0")
    with httpx.Client(timeout=10.0, transport=transport) as client:
        passages = build_corpus(questions, client)

    path = corpus_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(passages, indent=2))
    print(f"wrote {len(passages)} passages (of {len(questions)} questions) to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
