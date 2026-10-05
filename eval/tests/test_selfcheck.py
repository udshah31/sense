"""Tests for the SelfCheckGPT-NLI baseline scorer.

The contradiction scorer is injected, so the aggregation is tested with hand-computed
values and no model loading.
"""

import pytest

from sense_eval.nli_judge import ContradictionLabelError, contradiction_label
from sense_eval.selfcheck import SelfCheckScore, selfcheck_inconsistency


def constant_scorer(value):
    return lambda pairs: [value] * len(pairs)


def sequence_scorer(values):
    """Returns `values` in order; asserts it was called with the expected pair count."""
    def score(pairs):
        assert len(pairs) == len(values), f"expected {len(values)} pairs, got {len(pairs)}"
        return list(values)
    return score


def test_single_sentence_single_sample_is_the_pair_score():
    result = selfcheck_inconsistency(["Paris is in France."], ["Paris is in France."], constant_scorer(0.02))
    assert result.score == pytest.approx(0.02)
    assert result.n_sentences == 1 and result.n_samples == 1


def test_score_is_the_mean_over_samples_then_over_sentences():
    # 2 sentences x 3 samples. Pair order is sentence-major, sample-minor.
    # Sentence 1: (0.0, 0.0, 0.6) -> 0.2 ; Sentence 2: (1.0, 0.8, 0.6) -> 0.8
    # Passage: (0.2 + 0.8) / 2 = 0.5
    scores = [0.0, 0.0, 0.6, 1.0, 0.8, 0.6]
    result = selfcheck_inconsistency(["a", "b"], ["s1", "s2", "s3"], sequence_scorer(scores))

    assert result.per_sentence_scores == pytest.approx((0.2, 0.8))
    assert result.score == pytest.approx(0.5)


def test_pairs_are_built_sample_as_premise_sentence_as_hypothesis():
    """SelfCheckGPT-NLI asks whether a SAMPLE contradicts the main answer's sentence,
    so the sample must be the premise. Inverting this measures something else."""
    captured = {}

    def score(pairs):
        captured["pairs"] = list(pairs)
        return [0.5] * len(pairs)

    selfcheck_inconsistency(["MAIN"], ["SAMPLE_A", "SAMPLE_B"], score)
    assert captured["pairs"] == [("SAMPLE_A", "MAIN"), ("SAMPLE_B", "MAIN")]


def test_consistent_answer_scores_near_zero_and_inconsistent_near_one():
    consistent = selfcheck_inconsistency(["x"], ["s"] * 5, constant_scorer(0.01))
    inconsistent = selfcheck_inconsistency(["x"], ["s"] * 5, constant_scorer(0.97))
    assert consistent.score < inconsistent.score
    assert consistent.score == pytest.approx(0.01)
    assert inconsistent.score == pytest.approx(0.97)


def test_no_sentences_gives_none_not_zero():
    # An empty generation is not "perfectly consistent" — it is unscorable.
    result = selfcheck_inconsistency([], ["s1", "s2"], constant_scorer(0.5))
    assert result.score is None
    assert result.per_sentence_scores == ()


def test_no_samples_gives_none_not_zero():
    result = selfcheck_inconsistency(["a"], [], constant_scorer(0.5))
    assert result.score is None


def test_a_scorer_returning_the_wrong_count_raises():
    with pytest.raises(ValueError, match="one score per pair"):
        selfcheck_inconsistency(["a", "b"], ["s"], lambda pairs: [0.1])


def test_as_dict_shape_is_stable():
    result = SelfCheckScore(score=0.4, per_sentence_scores=(0.2, 0.6), n_sentences=2, n_samples=3)
    assert result.as_dict() == {
        "selfcheck_inconsistency": 0.4,
        "per_sentence_inconsistency": [0.2, 0.6],
        "n_sentences": 2,
        "n_samples": 3,
    }


# --- contradiction label resolution ---------------------------------------------


class _FakeConfig:
    def __init__(self, labels):
        self.id2label = dict(enumerate(labels))


class _FakeModel:
    def __init__(self, labels):
        self.config = _FakeConfig(labels)


@pytest.mark.parametrize("labels,expected", [
    (["ENTAILMENT", "NEUTRAL", "CONTRADICTION"], "CONTRADICTION"),
    (["entailment", "neutral", "contradiction"], "contradiction"),
    (["contradiction", "neutral", "entailment"], "contradiction"),
])
def test_contradiction_label_is_resolved_by_name_not_index(labels, expected):
    assert contradiction_label(_FakeModel(labels)) == expected


def test_opaque_label_names_raise_rather_than_guessing_an_index():
    with pytest.raises(ContradictionLabelError, match="map this model's labels explicitly"):
        contradiction_label(_FakeModel(["LABEL_0", "LABEL_1", "LABEL_2"]))


def test_ambiguous_labels_raise():
    with pytest.raises(ContradictionLabelError):
        contradiction_label(_FakeModel(["contradiction", "contradiction_strong", "entailment"]))
