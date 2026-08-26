import pytest

from sense_eval.nli_judge import entailment_scores, entailment_scores_batch, load_nli_model


def test_load_nli_model_requires_pinned_revision():
    with pytest.raises(ValueError, match="revision"):
        load_nli_model("cliang1453/deberta-v3-xsmall-mnli", "")


def test_load_nli_model_pins_tokenizer_max_length_to_model_position_limit(nli_model_and_tokenizer):
    """Regression test: an unpinned tokenizer.model_max_length silently
    disables truncation=True everywhere (the tokenizer logs "no maximum
    length is provided... default to no truncation"), which let a full
    Wikipedia-article premise blow up batched-inference memory. See
    load_nli_model's docstring."""
    model, tokenizer = nli_model_and_tokenizer
    assert tokenizer.model_max_length == model.config.max_position_embeddings
    assert tokenizer.model_max_length < 10_000  # sanity: not the huge default sentinel


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


def test_entailment_scores_batch_matches_unbatched_scores(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer
    pairs = [
        ("Paris is the capital of France.", "France's capital is Paris."),
        ("Paris is the capital of France.", "Berlin is the capital of France."),
    ]

    batched = entailment_scores_batch(model, tokenizer, pairs, batch_size=2)
    unbatched = [entailment_scores(model, tokenizer, premise, hypothesis) for premise, hypothesis in pairs]

    assert len(batched) == 2
    for b, u in zip(batched, unbatched):
        assert set(b.keys()) == set(u.keys())
        for label in b:
            assert abs(b[label] - u[label]) < 1e-4


def test_entailment_scores_batch_splits_across_multiple_batches(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer
    pairs = [("Paris is the capital of France.", "France's capital is Paris.")] * 5

    results = entailment_scores_batch(model, tokenizer, pairs, batch_size=2)

    assert len(results) == 5


def test_entailment_scores_batch_handles_empty_input(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer
    assert entailment_scores_batch(model, tokenizer, []) == []


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


from sense_eval.nli_judge import best_claim_entailment_batch, select_candidate_reference_sentences


def test_select_candidate_reference_sentences_returns_all_when_fewer_than_top_k():
    sentences = ["Sentence A.", "Sentence B."]
    assert select_candidate_reference_sentences("claim", sentences, top_k=3) == sentences


def test_select_candidate_reference_sentences_ranks_by_word_overlap():
    sentences = [
        "The weather today is sunny and warm.",
        "Taral Hicks was born in New York City in 1974.",
        "Many actresses have appeared in films.",
    ]
    candidates = select_candidate_reference_sentences(
        "Taral Hicks was born in 1974.", sentences, top_k=1
    )
    assert candidates == ["Taral Hicks was born in New York City in 1974."]


def test_best_claim_entailment_batch_picks_the_best_matching_sentence(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer
    reference_sentences = [
        "The weather today is sunny and warm.",
        "Albert Einstein was a German-born theoretical physicist.",
        "Many scientists have won the Nobel Prize.",
    ]

    results = best_claim_entailment_batch(model, tokenizer, [("Albert Einstein was a physicist.", reference_sentences)])

    assert len(results) == 1
    score, best_sentence = results[0]
    assert score > 0.5
    assert best_sentence == "Albert Einstein was a German-born theoretical physicist."


def test_best_claim_entailment_batch_returns_zero_for_empty_reference(nli_model_and_tokenizer):
    model, tokenizer = nli_model_and_tokenizer
    results = best_claim_entailment_batch(model, tokenizer, [("Some claim.", [])])
    assert results == [(0.0, None)]


def test_factscore_style_verdict_retrieves_relevant_sentence_from_long_reference(nli_model_and_tokenizer):
    """Regression test for the real finding that motivated retrieval:
    scoring a claim against an entire multi-paragraph reference (instead of
    the specific supporting sentence within it) badly degrades this small
    NLI model's entailment score, even when the supporting sentence is
    present and un-truncated. Mirrors the real Taral Hicks case documented
    in factscore_style_verdict's docstring, with a synthetic long reference
    so the test doesn't depend on network access to Wikipedia."""
    model, tokenizer = nli_model_and_tokenizer

    filler = " ".join(f"This is unrelated filler sentence number {i} about various topics." for i in range(20))
    reference_text = "Taral Hicks is an American actress and R&B singer. " + filler

    verdict, detail = factscore_style_verdict(
        model,
        tokenizer,
        generated_text="Taral Hicks is an American.",
        reference_text=reference_text,
        claim_supported_threshold=0.5,
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )

    assert verdict.label == "correct"
    assert detail.claim_entailment_scores[0] > 0.5
    assert detail.best_reference_sentences[0] == "Taral Hicks is an American actress and R&B singer."


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
