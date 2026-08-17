import pytest

from sense_eval.nli_judge import entailment_scores, load_nli_model


def test_load_nli_model_requires_pinned_revision():
    with pytest.raises(ValueError, match="revision"):
        load_nli_model("cliang1453/deberta-v3-xsmall-mnli", "")


def test_entailment_scores_true_pair_scores_high_entailment(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    scores = entailment_scores(
        model, tokenizer, "Paris is the capital of France.", "France's capital is Paris."
    )

    assert set(scores.keys()) == {"entailment", "neutral", "contradiction"}
    assert scores["entailment"] > 0.9
    assert abs(sum(scores.values()) - 1.0) < 1e-4


def test_entailment_scores_false_pair_scores_high_contradiction(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    scores = entailment_scores(
        model, tokenizer, "Paris is the capital of France.", "Berlin is the capital of France."
    )

    assert scores["contradiction"] > 0.9


from sense_eval.nli_judge import split_into_atomic_claims


def test_split_into_atomic_claims_splits_on_sentence_boundaries():
    claims = split_into_atomic_claims(
        "Albert Einstein was a physicist. He was born in 1879. He developed relativity."
    )

    assert claims == [
        "Albert Einstein was a physicist.",
        "He was born in 1879.",
        "He developed relativity.",
    ]


def test_split_into_atomic_claims_drops_empty_fragments():
    claims = split_into_atomic_claims("One sentence.   \n\n  Another sentence.")

    assert claims == ["One sentence.", "Another sentence."]


def test_split_into_atomic_claims_handles_empty_text():
    assert split_into_atomic_claims("") == []
    assert split_into_atomic_claims("   ") == []
