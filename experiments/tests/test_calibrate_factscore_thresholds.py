from calibrate_factscore_thresholds import (
    _ground_truth_label,
    accuracy_at_claim_threshold,
    accuracy_at_fraction_thresholds,
    build_claim_cases,
    build_fraction_cases,
    human_supported_fraction,
    pick_best_claim_threshold,
    pick_best_fraction_thresholds,
    split_fit_holdout,
    sweep_claim_thresholds,
    sweep_fraction_thresholds,
    usable_topics,
)
from sense_data.factscore_labeled import FActScoreLabeledExample, LabeledAtomicFact
from sense_eval.nli_judge import entailment_scores, load_nli_model

NLI_MODEL_CFG = {
    "hf_repo": "cliang1453/deberta-v3-xsmall-mnli",
    "revision": "d1ca70f9ece4d8afd33015893a69df9a6e45a672",
}


def _example(topic, model="InstructGPT", generated_text="", facts=(), reference_text="ref"):
    return FActScoreLabeledExample(
        model=model,
        topic=topic,
        generated_text=generated_text,
        human_atomic_facts=tuple(facts),
        reference_text=reference_text,
    )


def test_usable_topics_excludes_examples_with_no_reference_text():
    examples = [
        _example("A", reference_text="has a ref"),
        _example("B", reference_text=None),
    ]
    assert usable_topics(examples) == ["A"]


def test_split_fit_holdout_is_deterministic_and_disjoint():
    topics = [f"topic-{i}" for i in range(50)]
    fit_a, holdout_a = split_fit_holdout(topics)
    fit_b, holdout_b = split_fit_holdout(topics)

    assert fit_a == fit_b
    assert holdout_a == holdout_b
    assert set(fit_a) & set(holdout_a) == set()
    assert set(fit_a) | set(holdout_a) == set(topics)
    assert len(fit_a) == 40


def test_build_claim_cases_excludes_ir_facts_and_unusable_topics():
    examples = [
        _example(
            "A",
            reference_text="ref-a",
            facts=[
                LabeledAtomicFact(text="supported fact", label="S"),
                LabeledAtomicFact(text="unsupported fact", label="NS"),
                LabeledAtomicFact(text="irrelevant fact", label="IR"),
            ],
        ),
        _example("B", reference_text=None, facts=[LabeledAtomicFact(text="x", label="S")]),
    ]

    cases = build_claim_cases(examples, {"A"})

    assert len(cases) == 2
    assert {c["label"] for c in cases} == {"S", "NS"}
    assert all(c["reference_text"] == "ref-a" for c in cases)


def test_accuracy_at_claim_threshold_scores_binary_classification():
    cached = [
        {"label": "S", "entailment": 0.9},
        {"label": "S", "entailment": 0.1},  # false negative
        {"label": "NS", "entailment": 0.1},
        {"label": "NS", "entailment": 0.9},  # false positive
    ]

    metrics = accuracy_at_claim_threshold(cached, threshold=0.5)

    assert metrics["n_cases"] == 4
    assert metrics["verdict_accuracy"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["false_positive_rate"] == 0.5


def test_sweep_claim_thresholds_returns_one_result_per_candidate():
    cached = [{"label": "S", "entailment": 0.9}]
    results = sweep_claim_thresholds(cached, [0.3, 0.7])
    assert [r["threshold"] for r in results] == [0.3, 0.7]


def test_pick_best_claim_threshold_prefers_higher_accuracy_then_lower_threshold():
    sweep = [
        {"threshold": 0.7, "verdict_accuracy": 0.9},
        {"threshold": 0.3, "verdict_accuracy": 0.9},
        {"threshold": 0.5, "verdict_accuracy": 0.6},
    ]
    assert pick_best_claim_threshold(sweep)["threshold"] == 0.3


def test_human_supported_fraction_excludes_irrelevant_facts():
    example = _example(
        "A",
        facts=[
            LabeledAtomicFact(text="a", label="S"),
            LabeledAtomicFact(text="b", label="S"),
            LabeledAtomicFact(text="c", label="NS"),
            LabeledAtomicFact(text="d", label="IR"),
        ],
    )
    assert human_supported_fraction(example) == 2 / 3


def test_human_supported_fraction_is_none_with_no_s_or_ns_facts():
    example = _example("A", facts=[LabeledAtomicFact(text="d", label="IR")])
    assert human_supported_fraction(example) is None


def test_ground_truth_label_thresholds():
    assert _ground_truth_label(0.9) == "correct"
    assert _ground_truth_label(0.1) == "incorrect"
    assert _ground_truth_label(0.5) == "ambiguous"


def test_accuracy_at_fraction_thresholds_excludes_ambiguous_ground_truth():
    cached = [
        {"human_supported_fraction": 0.9, "our_supported_fraction": 0.9},  # correct, predicted correct
        {"human_supported_fraction": 0.1, "our_supported_fraction": 0.9},  # incorrect, predicted correct (wrong)
        {"human_supported_fraction": 0.5, "our_supported_fraction": 0.9},  # ambiguous, excluded
    ]

    metrics = accuracy_at_fraction_thresholds(cached, fraction_correct_threshold=0.8, fraction_incorrect_threshold=0.2)

    assert metrics["n_definite_ground_truth_cases"] == 2
    assert metrics["verdict_accuracy"] == 0.5


def test_accuracy_at_fraction_thresholds_handles_no_definite_cases():
    metrics = accuracy_at_fraction_thresholds(
        [{"human_supported_fraction": 0.5, "our_supported_fraction": 0.5}],
        fraction_correct_threshold=0.8,
        fraction_incorrect_threshold=0.2,
    )
    assert metrics["n_definite_ground_truth_cases"] == 0
    assert metrics["verdict_accuracy"] is None
    assert metrics["unknown_rate"] is None


def test_sweep_fraction_thresholds_only_includes_incorrect_below_correct():
    cached = [{"human_supported_fraction": 0.9, "our_supported_fraction": 0.9}]
    results = sweep_fraction_thresholds(cached, [0.2, 0.5, 0.8])
    for r in results:
        assert r["fraction_incorrect_threshold"] < r["fraction_correct_threshold"]


def test_pick_best_fraction_thresholds_prefers_higher_accuracy():
    sweep = [
        {"fraction_correct_threshold": 0.8, "fraction_incorrect_threshold": 0.2, "verdict_accuracy": 0.6, "unknown_rate": 0.1},
        {"fraction_correct_threshold": 0.7, "fraction_incorrect_threshold": 0.3, "verdict_accuracy": 0.9, "unknown_rate": 0.05},
    ]
    best = pick_best_fraction_thresholds(sweep)
    assert (best["fraction_correct_threshold"], best["fraction_incorrect_threshold"]) == (0.7, 0.3)


def test_build_fraction_cases_skips_examples_with_no_claims_or_no_sns_facts():
    model, tokenizer = load_nli_model(NLI_MODEL_CFG["hf_repo"], NLI_MODEL_CFG["revision"])
    examples = [
        _example("A", reference_text="Albert Einstein was a physicist.", generated_text="", facts=[LabeledAtomicFact(text="x", label="S")]),
        _example("B", reference_text=None, generated_text="Something.", facts=[LabeledAtomicFact(text="x", label="S")]),
        _example("C", reference_text="ref", generated_text="Something happened.", facts=[LabeledAtomicFact(text="x", label="IR")]),
        _example(
            "D",
            reference_text="Albert Einstein was a theoretical physicist.",
            generated_text="Albert Einstein was a physicist.",
            facts=[LabeledAtomicFact(text="x", label="S"), LabeledAtomicFact(text="y", label="NS")],
        ),
    ]

    cases = build_fraction_cases(model, tokenizer, examples, {"A", "B", "C", "D"}, claim_supported_threshold=0.5)

    assert len(cases) == 1
    assert cases[0]["topic"] == "D"
    assert cases[0]["human_supported_fraction"] == 0.5
    assert 0.0 <= cases[0]["our_supported_fraction"] <= 1.0


def test_claim_threshold_accuracy_matches_real_entailment_scores():
    """Sanity check against the real model: a clearly-supported fact should
    score high entailment against a matching reference, a clearly-
    unsupported one should score low."""
    model, tokenizer = load_nli_model(NLI_MODEL_CFG["hf_repo"], NLI_MODEL_CFG["revision"])
    reference = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity."

    supported = entailment_scores(model, tokenizer, reference, "Albert Einstein was a physicist.")["entailment"]
    unsupported = entailment_scores(model, tokenizer, reference, "Albert Einstein was a professional chef.")[
        "entailment"
    ]

    assert supported > unsupported
