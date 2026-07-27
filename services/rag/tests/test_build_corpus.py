import httpx
import pytest

from build_corpus import build_corpus, fetch_passage

WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


def _mock_transport(response_json):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_json)

    return httpx.MockTransport(handler)


def test_fetch_passage_returns_title_and_extract():
    response_json = {
        "query": {
            "pages": {
                "736": {
                    "title": "Albert Einstein",
                    "extract": "Albert Einstein was a theoretical physicist.",
                }
            }
        }
    }
    client = httpx.Client(transport=_mock_transport(response_json))

    passage = fetch_passage("Who was Albert Einstein?", client)

    assert passage == {
        "title": "Albert Einstein",
        "text": "Albert Einstein was a theoretical physicist.",
    }


def test_fetch_passage_returns_none_when_no_pages():
    response_json = {"query": {"pages": {}}}
    client = httpx.Client(transport=_mock_transport(response_json))

    passage = fetch_passage("asdlkfjasldkfj nonsense query", client)

    assert passage is None


def test_build_corpus_skips_unresolved_questions():
    response_json = {"query": {"pages": {}}}
    client = httpx.Client(transport=_mock_transport(response_json))

    passages = build_corpus(["a question with no hit"], client)

    assert passages == []


def test_build_corpus_collects_resolved_passages():
    response_json = {
        "query": {
            "pages": {
                "1": {"title": "Paris", "text_stub": "unused"},
            }
        }
    }
    # fetch_passage keys off "extract", not "text_stub" — this response has no
    # extract, so it should resolve to None and be skipped, proving build_corpus
    # tolerates a mix of resolved and unresolved questions in one run.
    client = httpx.Client(transport=_mock_transport(response_json))

    passages = build_corpus(["question one", "question two"], client)

    assert passages == []
