import json

from sense_rag.index import PassageIndex

PASSAGES = [
    {"title": "Albert Einstein", "text": "Albert Einstein was a theoretical physicist known for relativity."},
    {"title": "Paris", "text": "Paris is the capital city of France."},
    {"title": "Photosynthesis", "text": "Photosynthesis converts light energy into chemical energy in plants."},
]


def test_from_passages_search_returns_most_similar_text_first():
    index = PassageIndex.from_passages(PASSAGES)

    results = index.search("Who discovered the theory of relativity?", k=1)

    assert results == ["Albert Einstein was a theoretical physicist known for relativity."]


def test_search_respects_k():
    index = PassageIndex.from_passages(PASSAGES)

    results = index.search("science topics", k=2)

    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_from_file_loads_committed_passages_json(tmp_path):
    passages_path = tmp_path / "passages.json"
    passages_path.write_text(json.dumps(PASSAGES))

    index = PassageIndex.from_file(passages_path)

    assert index.passages == PASSAGES
    results = index.search("capital of France", k=1)
    assert results == ["Paris is the capital city of France."]
