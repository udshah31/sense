import json

import pytest

from rag_baseline_truthful_qa import build_prompt, run_experiment


def test_build_prompt_formats_context_and_question():
    prompt = build_prompt(["Paris is the capital of France.", "France is in Europe."], "What is the capital of France?")

    assert "Paris is the capital of France." in prompt
    assert "France is in Europe." in prompt
    assert "What is the capital of France?" in prompt
    assert prompt.index("Paris is the capital of France.") < prompt.index("What is the capital of France?")


@pytest.mark.asyncio
async def test_run_experiment_end_to_end_on_fixture_corpus(tmp_path, monkeypatch):
    fixture_passages = [
        {"title": "Fixture", "text": "This is a fixture passage about a fixture topic."},
    ]
    corpus_path = tmp_path / "passages.json"
    corpus_path.write_text(json.dumps(fixture_passages))

    config = {
        "models": {"cpu_test": {"hf_repo": "sshleifer/tiny-gpt2", "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"}},
        "gate": {"decoding": {"do_sample": False, "max_new_tokens": 5}},
        "rag": {"model": "cpu_test", "top_k": 1, "corpus_path": str(corpus_path), "eval_split": "test", "n_eval_examples": 2},
    }

    result = await run_experiment(config)

    assert result["n_eval_examples"] == 2
    assert 0.0 <= result["factuality_accuracy_proxy"] <= 1.0
    assert len(result["per_example"]) == 2
    for record in result["per_example"]:
        assert "retrieved_passages_preview" in record
        assert "factuality" in record
