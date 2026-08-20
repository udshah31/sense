from calibrate_nli_short_answer_threshold import (
    _tri_state_verdict,
    accuracy_at_threshold,
    build_calibration_pool,
    compute_calibration_scores,
    pick_best_threshold,
    split_fit_holdout,
    sweep_thresholds,
)
from sense_data.halueval import HaluEvalExample
from sense_data.splits import PerCheckpointSplitIndices
from sense_eval.nli_judge import load_nli_model, nli_verdict_short_answer

NLI_MODEL_CFG = {
    "hf_repo": "cliang1453/deberta-v3-xsmall-mnli",
    "revision": "d1ca70f9ece4d8afd33015893a69df9a6e45a672",
}


def test_build_calibration_pool_unions_and_dedupes_all_checkpoints():
    splits = PerCheckpointSplitIndices(
        development=[100],
        test=[200],
        calibration={"a": [1, 3, 5], "b": [2, 4, 6]},
    )
    assert build_calibration_pool(splits) == [1, 2, 3, 4, 5, 6]


def test_split_fit_holdout_is_deterministic_and_disjoint():
    pool = list(range(100))
    fit_a, holdout_a = split_fit_holdout(pool)
    fit_b, holdout_b = split_fit_holdout(pool)

    assert fit_a == fit_b
    assert holdout_a == holdout_b
    assert set(fit_a) & set(holdout_a) == set()
    assert set(fit_a) | set(holdout_a) == set(pool)
    assert len(fit_a) == 80
    assert len(holdout_a) == 20


def test_pick_best_threshold_prefers_higher_accuracy():
    sweep = [
        {"threshold": 0.5, "verdict_accuracy": 0.6, "unknown_rate": 0.1},
        {"threshold": 0.7, "verdict_accuracy": 0.9, "unknown_rate": 0.05},
    ]
    assert pick_best_threshold(sweep)["threshold"] == 0.7


def test_pick_best_threshold_breaks_ties_by_lower_unknown_rate():
    sweep = [
        {"threshold": 0.5, "verdict_accuracy": 0.8, "unknown_rate": 0.2},
        {"threshold": 0.7, "verdict_accuracy": 0.8, "unknown_rate": 0.05},
    ]
    assert pick_best_threshold(sweep)["threshold"] == 0.7


def test_pick_best_threshold_breaks_remaining_ties_by_lower_threshold():
    sweep = [
        {"threshold": 0.7, "verdict_accuracy": 0.8, "unknown_rate": 0.1},
        {"threshold": 0.5, "verdict_accuracy": 0.8, "unknown_rate": 0.1},
    ]
    assert pick_best_threshold(sweep)["threshold"] == 0.5


def test_tri_state_verdict_matches_correct_case():
    assert _tri_state_verdict(right_entailment=0.9, wrong_entailment=0.1, threshold=0.5) == "correct"


def test_tri_state_verdict_matches_incorrect_case():
    assert _tri_state_verdict(right_entailment=0.1, wrong_entailment=0.9, threshold=0.5) == "incorrect"


def test_tri_state_verdict_matches_unknown_when_neither_clears():
    assert _tri_state_verdict(right_entailment=0.1, wrong_entailment=0.1, threshold=0.5) == "unknown"


def test_tri_state_verdict_matches_unknown_when_both_clear():
    assert _tri_state_verdict(right_entailment=0.9, wrong_entailment=0.9, threshold=0.5) == "unknown"


def test_tri_state_verdict_does_not_drift_from_nli_verdict_short_answer():
    """_tri_state_verdict duplicates nli_verdict_short_answer's tri-state contract
    on precomputed scores for calibration-sweep performance (see module
    docstring) — this guards the duplication against silently drifting apart by
    checking both agree on a real example through the real NLI model."""
    model, tokenizer = load_nli_model(NLI_MODEL_CFG["hf_repo"], NLI_MODEL_CFG["revision"])
    right_answer = "Paris is the capital of France."
    hallucinated_answer = "Berlin is the capital of France."

    for generated_text in (right_answer, hallucinated_answer):
        real_verdict = nli_verdict_short_answer(
            model, tokenizer, generated_text, right_answer, hallucinated_answer, entailment_threshold=0.5
        )

        scores = compute_calibration_scores(
            model,
            tokenizer,
            [
                HaluEvalExample(
                    index=0, knowledge="", question="q", right_answer=right_answer,
                    hallucinated_answer=hallucinated_answer,
                )
            ],
            [0],
        )[0]
        precomputed_right_entailment = scores["right_vs_right"] if generated_text == right_answer else scores["wrong_vs_right"]
        precomputed_wrong_entailment = scores["right_vs_wrong"] if generated_text == right_answer else scores["wrong_vs_wrong"]

        assert _tri_state_verdict(precomputed_right_entailment, precomputed_wrong_entailment, 0.5) == real_verdict.label


def test_compute_calibration_scores_returns_four_scores_per_example():
    model, tokenizer = load_nli_model(NLI_MODEL_CFG["hf_repo"], NLI_MODEL_CFG["revision"])
    examples = [
        HaluEvalExample(
            index=0,
            knowledge="",
            question="What is the capital of France?",
            right_answer="Paris is the capital of France.",
            hallucinated_answer="Berlin is the capital of France.",
        )
    ]

    scores = compute_calibration_scores(model, tokenizer, examples, [0])

    assert len(scores) == 1
    row = scores[0]
    assert row["index"] == 0
    for key in ("right_vs_right", "right_vs_wrong", "wrong_vs_right", "wrong_vs_wrong"):
        assert 0.0 <= row[key] <= 1.0
    # self-entailment should score near-maximal — the honest limitation the
    # module docstring names, not a bug.
    assert row["right_vs_right"] > 0.9
    assert row["wrong_vs_wrong"] > 0.9


def test_accuracy_at_threshold_scores_cached_examples():
    cached_scores = [
        {"index": 0, "right_vs_right": 0.95, "right_vs_wrong": 0.05, "wrong_vs_right": 0.05, "wrong_vs_wrong": 0.95},
    ]

    metrics = accuracy_at_threshold(cached_scores, threshold=0.5)

    assert metrics == {"threshold": 0.5, "n_cases": 2, "verdict_accuracy": 1.0, "unknown_rate": 0.0}


def test_sweep_thresholds_returns_one_result_per_candidate_in_order():
    cached_scores = [
        {"index": 0, "right_vs_right": 0.95, "right_vs_wrong": 0.05, "wrong_vs_right": 0.05, "wrong_vs_wrong": 0.95},
    ]

    results = sweep_thresholds(cached_scores, [0.3, 0.7])

    assert [r["threshold"] for r in results] == [0.3, 0.7]
