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


from sense_eval.nli_judge import nli_verdict_short_answer


def test_nli_verdict_short_answer_correct_when_generated_entails_right_answer(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict = nli_verdict_short_answer(
        model,
        tokenizer,
        generated_text="The watermelon seeds pass through your digestive system without effect.",
        right_answer="Nothing happens, the seeds just pass through your digestive system.",
        hallucinated_answer="You die if you eat watermelon seeds.",
        entailment_threshold=0.7,
    )

    assert verdict.label == "correct"


def test_nli_verdict_short_answer_incorrect_when_generated_entails_hallucinated_answer(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict = nli_verdict_short_answer(
        model,
        tokenizer,
        generated_text="You die if you eat watermelon seeds.",
        right_answer="Nothing happens, the seeds just pass through your digestive system.",
        hallucinated_answer="You die if you eat watermelon seeds.",
        entailment_threshold=0.7,
    )

    assert verdict.label == "incorrect"


def test_nli_verdict_short_answer_unknown_when_neither_entailed(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict = nli_verdict_short_answer(
        model,
        tokenizer,
        generated_text="Watermelons are a delicious summer fruit.",
        right_answer="Nothing happens, the seeds just pass through your digestive system.",
        hallucinated_answer="You die if you eat watermelon seeds.",
        entailment_threshold=0.7,
    )

    assert verdict.label == "unknown"


from sense_eval.nli_judge import factscore_style_verdict


def test_factscore_style_verdict_correct_when_all_claims_supported(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict, detail = factscore_style_verdict(
        model,
        tokenizer,
        generated_text="Albert Einstein was a theoretical physicist. He was born in Germany.",
        reference_text=(
            "Albert Einstein was a German-born theoretical physicist, widely "
            "acknowledged to be one of the greatest physicists of all time."
        ),
        claim_supported_threshold=0.5,
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )

    assert verdict.label == "correct"
    assert detail.supported_fraction == 1.0
    assert len(detail.claims) == 2
    assert len(detail.claim_entailment_scores) == 2


def test_factscore_style_verdict_incorrect_when_no_claims_supported(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict, detail = factscore_style_verdict(
        model,
        tokenizer,
        generated_text="Albert Einstein was a professional basketball player.",
        reference_text=(
            "Albert Einstein was a German-born theoretical physicist, widely "
            "acknowledged to be one of the greatest physicists of all time."
        ),
        claim_supported_threshold=0.5,
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )

    assert verdict.label == "incorrect"
    assert detail.supported_fraction == 0.0


def test_factscore_style_verdict_unknown_when_no_claims_to_score(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer

    verdict, detail = factscore_style_verdict(
        model,
        tokenizer,
        generated_text="   ",
        reference_text="Albert Einstein was a German-born theoretical physicist.",
        claim_supported_threshold=0.5,
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )

    assert verdict.label == "unknown"
    assert detail.claims == ()
    assert detail.supported_fraction is None
