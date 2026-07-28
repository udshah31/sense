import json

import pytest
from transformers import AutoTokenizer

from rag_baseline_truthful_qa import build_prompt, build_prompt_within_budget, run_experiment

TINY_GPT2_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"


def test_build_prompt_formats_context_and_question():
    prompt = build_prompt(["Paris is the capital of France.", "France is in Europe."], "What is the capital of France?")

    assert "Paris is the capital of France." in prompt
    assert "France is in Europe." in prompt
    assert "What is the capital of France?" in prompt
    assert prompt.index("Paris is the capital of France.") < prompt.index("What is the capital of France?")


@pytest.mark.asyncio
async def test_run_experiment_end_to_end_on_fixture_corpus(tmp_path):
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
        assert "prompt_truncated" in record
        assert "n_passages_used" in record
    assert "verdict_counts" in result
    assert sum(result["verdict_counts"].values()) == len(result["per_example"])
    assert "n_examples_with_truncated_prompt" in result


@pytest.mark.asyncio
async def test_run_experiment_does_not_crash_on_overlong_prompt(tmp_path):
    # A repeated long sentence forces the tokenized prompt well past tiny-gpt2's
    # 1024-token position-embedding limit, reproducing the crash observed in the
    # full 327-example run (IndexError: index out of range in self).
    long_sentence = "This is a very long fixture sentence about a fixture topic used to overflow the context window. "
    fixture_passages = [
        {"title": "Fixture", "text": long_sentence * 60},
        {"title": "Fixture2", "text": long_sentence * 60},
        {"title": "Fixture3", "text": long_sentence * 60},
    ]
    corpus_path = tmp_path / "passages.json"
    corpus_path.write_text(json.dumps(fixture_passages))

    config = {
        "models": {"cpu_test": {"hf_repo": "sshleifer/tiny-gpt2", "revision": TINY_GPT2_REVISION}},
        "gate": {"decoding": {"do_sample": False, "max_new_tokens": 5}},
        "rag": {"model": "cpu_test", "top_k": 3, "corpus_path": str(corpus_path), "eval_split": "test", "n_eval_examples": 1},
    }

    result = await run_experiment(config)

    assert result["n_eval_examples"] == 1
    assert len(result["per_example"]) == 1
    record = result["per_example"][0]
    # Only the highest-ranked passage(s) that fit should survive; with 3 huge
    # passages far exceeding budget, expect fewer than top_k retained and the
    # truncation to be recorded.
    assert record["n_passages_used"] < 3
    assert record["prompt_truncated"] is True


def test_build_prompt_within_budget_keeps_highest_ranked_passages_that_fit():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", revision=TINY_GPT2_REVISION)

    passages = [
        "The Eiffel Tower is in Paris.",  # rank 1, most relevant
        "filler token " * 500,  # rank 2, too big to fit alongside rank 1 in a small budget
        "filler token " * 500,  # rank 3
    ]
    question = "Where is the Eiffel Tower?"

    prompt, n_passages_used, dropped = build_prompt_within_budget(passages, question, tokenizer, max_tokens=50)

    assert n_passages_used == 1
    assert dropped is True
    assert "The Eiffel Tower is in Paris." in prompt
    assert question in prompt


def test_build_prompt_within_budget_keeps_all_passages_when_they_fit():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", revision=TINY_GPT2_REVISION)

    passages = ["Paris is the capital of France.", "France is in Europe."]
    question = "What is the capital of France?"

    prompt, n_passages_used, dropped = build_prompt_within_budget(passages, question, tokenizer, max_tokens=1000)

    assert n_passages_used == 2
    assert dropped is False
    for passage in passages:
        assert passage in prompt


@pytest.mark.asyncio
async def test_run_experiment_handles_zero_examples_without_zero_division(tmp_path):
    fixture_passages = [{"title": "Fixture", "text": "A fixture passage."}]
    corpus_path = tmp_path / "passages.json"
    corpus_path.write_text(json.dumps(fixture_passages))

    config = {
        "models": {"cpu_test": {"hf_repo": "sshleifer/tiny-gpt2", "revision": TINY_GPT2_REVISION}},
        "gate": {"decoding": {"do_sample": False, "max_new_tokens": 5}},
        "rag": {"model": "cpu_test", "top_k": 1, "corpus_path": str(corpus_path), "eval_split": "test", "n_eval_examples": 0},
    }

    result = await run_experiment(config)

    assert result["n_eval_examples"] == 0
    assert result["factuality_accuracy_proxy"] is None
    assert result["verdict_counts"] == {"correct": 0, "incorrect": 0, "unknown": 0}
    assert result["n_examples_with_truncated_prompt"] == 0


def test_truncation_side_left_preserves_question_text():
    # Left-truncation must cut off the (long) context, not the question that is
    # appended last in the prompt, or the model would generate against a prompt
    # missing the thing it's supposed to answer.
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", revision=TINY_GPT2_REVISION)
    tokenizer.truncation_side = "left"

    long_context = "filler token " * 2000
    question = "What is the capital of France?"
    prompt = build_prompt([long_context], question)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=100)
    decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    assert decoded.rstrip().endswith(question) or question in decoded[-len(question) - 10 :]
