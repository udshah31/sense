from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve

PASSAGES = [
    {"title": "Albert Einstein", "text": "Albert Einstein was a theoretical physicist known for relativity."},
    {"title": "Paris", "text": "Paris is the capital city of France."},
    {"title": "Photosynthesis", "text": "Photosynthesis converts light energy into chemical energy in plants."},
]


def test_retrieve_returns_default_top_3():
    index = PassageIndex.from_passages(PASSAGES)

    results = retrieve(index, "Tell me about science")

    assert len(results) == 3


def test_retrieve_respects_explicit_k():
    index = PassageIndex.from_passages(PASSAGES)

    results = retrieve(index, "Who was Einstein?", k=1)

    assert results == ["Albert Einstein was a theoretical physicist known for relativity."]
