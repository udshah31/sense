"""Tests for routing-quality detection metrics.

AUROC is checked against hand-computed values rather than another library, so the
rank-sum implementation (including its tie handling) is verified rather than assumed.
"""

import pytest

from sense_eval.routing_quality import (
    RoutingOutcome,
    UnknownFactualityLabelError,
    _auroc,
    routing_detection_metrics,
    routing_quality_delta,
)


# --- AUROC ---------------------------------------------------------------------


def test_auroc_perfect_separation():
    # Every positive scores above every negative.
    assert _auroc([0.8, 0.9, 1.0], [0.1, 0.2, 0.3]) == 1.0


def test_auroc_perfectly_inverted():
    assert _auroc([0.1, 0.2], [0.8, 0.9]) == 0.0


def test_auroc_all_identical_scores_is_one_half():
    # Complete ties must land exactly on chance, not on a sort-order artifact.
    assert _auroc([0.5, 0.5, 0.5], [0.5, 0.5]) == 0.5


def test_auroc_hand_computed_mixed_case():
    # pos=[3,1], neg=[2,0]. Pairwise: (3>2), (3>0), (1<2), (1>0) -> 3 of 4.
    assert _auroc([3.0, 1.0], [2.0, 0.0]) == 0.75


def test_auroc_single_cross_class_tie_counts_as_half():
    # One positive and one negative at the same score: exactly 0.5.
    assert _auroc([1.0], [1.0]) == 0.5


def test_auroc_is_none_when_a_class_is_empty():
    # Undefined, so it must be absent rather than defaulted to 0.5.
    assert _auroc([], [0.1, 0.2]) is None
    assert _auroc([0.1, 0.2], []) is None


# --- detection metrics ---------------------------------------------------------


def test_detection_metrics_counts_and_rates():
    outcomes = [
        RoutingOutcome(entropy=0.9, routed=True, factuality="incorrect"),   # TP
        RoutingOutcome(entropy=0.8, routed=True, factuality="incorrect"),   # TP
        RoutingOutcome(entropy=0.7, routed=True, factuality="correct"),     # FP
        RoutingOutcome(entropy=0.2, routed=False, factuality="incorrect"),  # FN
        RoutingOutcome(entropy=0.1, routed=False, factuality="correct"),    # TN
        RoutingOutcome(entropy=0.1, routed=False, factuality="correct"),    # TN
    ]
    m = routing_detection_metrics(outcomes)

    assert (m["n_true_positive"], m["n_false_positive"]) == (2, 1)
    assert (m["n_false_negative"], m["n_true_negative"]) == (1, 2)
    assert m["routed_hallucination_precision"] == pytest.approx(2 / 3)
    assert m["hallucination_recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    assert m["routing_rate"] == pytest.approx(0.5)


def test_unknown_verdicts_are_excluded_from_detection_but_counted():
    outcomes = [
        RoutingOutcome(entropy=0.9, routed=True, factuality="incorrect"),
        RoutingOutcome(entropy=0.1, routed=False, factuality="correct"),
        RoutingOutcome(entropy=0.9, routed=True, factuality="unknown"),
        RoutingOutcome(entropy=0.1, routed=False, factuality="unknown"),
    ]
    m = routing_detection_metrics(outcomes)

    assert m["n_outcomes"] == 4
    assert m["n_scored"] == 2
    assert m["n_unknown_excluded"] == 2
    # Unknowns must not inflate either class: one TP and one TN only.
    assert (m["n_true_positive"], m["n_true_negative"]) == (1, 1)
    assert (m["n_false_positive"], m["n_false_negative"]) == (0, 0)
    # Routing rate is a cost figure and DOES include unknowns in its denominator.
    assert m["routing_rate"] == pytest.approx(0.5)


def test_rates_are_none_not_zero_when_denominator_is_empty():
    # Nothing routed: precision has no denominator. 0.0 would be a fabricated number.
    outcomes = [
        RoutingOutcome(entropy=0.1, routed=False, factuality="correct"),
        RoutingOutcome(entropy=0.2, routed=False, factuality="incorrect"),
    ]
    m = routing_detection_metrics(outcomes)

    assert m["routed_hallucination_precision"] is None
    assert m["f1"] is None
    assert m["hallucination_recall"] == 0.0  # real measurement: one miss, zero catches
    assert m["routing_rate"] == 0.0


def test_auroc_reported_over_the_scored_subset_only():
    outcomes = [
        RoutingOutcome(entropy=0.9, routed=True, factuality="incorrect"),
        RoutingOutcome(entropy=0.1, routed=False, factuality="correct"),
        # An unknown at a score that would drag AUROC down if it were counted.
        RoutingOutcome(entropy=0.0, routed=False, factuality="unknown"),
    ]
    assert routing_detection_metrics(outcomes)["entropy_auroc"] == 1.0


def test_auroc_absent_when_every_scored_example_shares_one_class():
    outcomes = [
        RoutingOutcome(entropy=0.9, routed=True, factuality="incorrect"),
        RoutingOutcome(entropy=0.8, routed=True, factuality="incorrect"),
    ]
    assert routing_detection_metrics(outcomes)["entropy_auroc"] is None


def test_empty_outcomes_yields_none_rates_without_raising():
    m = routing_detection_metrics([])
    assert m["n_outcomes"] == 0
    assert m["routing_rate"] is None
    assert m["entropy_auroc"] is None


def test_unrecognized_factuality_label_raises():
    with pytest.raises(UnknownFactualityLabelError, match="hallucinated"):
        routing_detection_metrics(
            [RoutingOutcome(entropy=0.5, routed=True, factuality="hallucinated")]
        )


# --- delta ---------------------------------------------------------------------


def test_delta_is_native_minus_transferred():
    transferred = {"f1": 0.4, "hallucination_recall": 0.3, "routed_hallucination_precision": 0.5,
                   "specificity": 0.6, "balanced_accuracy": 0.45, "routing_rate": 0.2}
    native = {"f1": 0.7, "hallucination_recall": 0.6, "routed_hallucination_precision": 0.8,
              "specificity": 0.9, "balanced_accuracy": 0.75, "routing_rate": 0.1}

    d = routing_quality_delta(transferred, native)
    assert d["delta_f1"] == pytest.approx(0.3)
    assert d["delta_routing_rate"] == pytest.approx(-0.1)


def test_delta_propagates_none_rather_than_treating_it_as_zero():
    transferred = {"f1": None, "hallucination_recall": 0.3}
    native = {"f1": 0.7, "hallucination_recall": 0.6}
    d = routing_quality_delta(transferred, native)

    assert d["delta_f1"] is None
    assert d["delta_hallucination_recall"] == pytest.approx(0.3)


def test_delta_does_not_difference_auroc():
    # Threshold-free, so both gates see the same distribution on the same model and
    # the delta would always be zero by construction — a misleading number to report.
    d = routing_quality_delta({"f1": 0.4}, {"f1": 0.7})
    assert "delta_entropy_auroc" not in d


# --- confidence intervals ------------------------------------------------------

from sense_eval.routing_quality import (  # noqa: E402
    CI_METRICS,
    DELTA_CI_METRICS,
    detection_metrics_with_ci,
    routing_quality_delta_with_ci,
)

CI_KWARGS = {"n_resamples": 200, "confidence": 0.95, "seed": 11}


def _outcomes(spec):
    """spec: list of (entropy, routed, factuality)."""
    return [RoutingOutcome(entropy=e, routed=r, factuality=f) for e, r, f in spec]


SEPARATED = _outcomes([
    (0.95, True, "incorrect"), (0.90, True, "incorrect"), (0.85, True, "incorrect"),
    (0.20, False, "correct"), (0.15, False, "correct"), (0.10, False, "correct"),
    (0.05, False, "correct"), (0.30, False, "correct"),
])
# Same examples, same labels, a gate that fires on the wrong half.
INVERTED = _outcomes([
    (0.95, False, "incorrect"), (0.90, False, "incorrect"), (0.85, False, "incorrect"),
    (0.20, True, "correct"), (0.15, True, "correct"), (0.10, True, "correct"),
    (0.05, True, "correct"), (0.30, True, "correct"),
])


def test_ci_metrics_preserve_the_point_estimates():
    with_ci = detection_metrics_with_ci(SEPARATED, **CI_KWARGS)
    plain = routing_detection_metrics(SEPARATED)
    for key in plain:
        assert with_ci[key] == plain[key], key


def test_every_ci_metric_gets_an_interval_bracketing_its_point():
    with_ci = detection_metrics_with_ci(SEPARATED, **CI_KWARGS)
    for key in CI_METRICS:
        interval = with_ci[f"{key}_ci"]
        assert interval["point"] == with_ci[key]
        assert interval["n_resamples"] == 200
        if interval["ci_low"] is not None:
            assert interval["ci_low"] <= interval["point"] <= interval["ci_high"]


def test_ci_is_reproducible_for_a_given_seed():
    a = detection_metrics_with_ci(SEPARATED, **CI_KWARGS)["f1_ci"]
    b = detection_metrics_with_ci(SEPARATED, **CI_KWARGS)["f1_ci"]
    assert a == b


def test_empty_outcomes_yield_intervals_with_no_endpoints():
    with_ci = detection_metrics_with_ci([], **CI_KWARGS)
    for key in CI_METRICS:
        interval = with_ci[f"{key}_ci"]
        assert interval["ci_low"] is None and interval["ci_high"] is None


def test_a_perfect_gate_versus_an_inverted_one_gives_a_delta_interval_excluding_zero():
    """The shape RQ1 and RQ2 are looking for: a gap larger than sampling noise."""
    delta = routing_quality_delta_with_ci(INVERTED, SEPARATED, **CI_KWARGS)
    f1 = delta["delta_f1_ci"]
    assert f1["point"] > 0
    assert f1["ci_low"] > 0, "a perfect-vs-inverted gate gap should clear zero"


def test_identical_gates_give_a_delta_of_exactly_zero():
    delta = routing_quality_delta_with_ci(SEPARATED, SEPARATED, **CI_KWARGS)
    for key in DELTA_CI_METRICS:
        interval = delta[f"delta_{key}_ci"]
        if interval["point"] is not None:
            assert interval["point"] == 0.0
            assert interval["ci_low"] == 0.0 and interval["ci_high"] == 0.0


def test_delta_ci_excludes_auroc_because_its_delta_is_zero_by_construction():
    delta = routing_quality_delta_with_ci(INVERTED, SEPARATED, **CI_KWARGS)
    assert "delta_entropy_auroc_ci" not in delta
    assert "entropy_auroc" not in DELTA_CI_METRICS


def test_misaligned_outcome_lists_raise():
    with pytest.raises(ValueError, match="index-aligned"):
        routing_quality_delta_with_ci(SEPARATED, SEPARATED[:-1], **CI_KWARGS)


def test_count_keys_do_not_get_intervals():
    # Counts describe this sample, not an estimate of a population quantity.
    with_ci = detection_metrics_with_ci(SEPARATED, **CI_KWARGS)
    for key in ("n_true_positive", "n_false_positive", "n_outcomes", "n_scored"):
        assert f"{key}_ci" not in with_ci


def test_f1_is_zero_not_none_when_a_gate_catches_nothing_it_should():
    """Regression: f1 previously returned None when precision and recall were both a
    legitimate 0.0, which hid total gate failure and made the delta against a working
    gate unreportable. Both are real measurements here — five examples were routed and
    none were hallucinations; three hallucinations existed and none were caught."""
    metrics = routing_detection_metrics(INVERTED)
    assert metrics["routed_hallucination_precision"] == 0.0
    assert metrics["hallucination_recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_f1_stays_none_when_a_component_is_unmeasurable():
    # Nothing routed: precision has no denominator, so f1 cannot be formed.
    nothing_routed = _outcomes([(0.1, False, "correct"), (0.2, False, "incorrect")])
    metrics = routing_detection_metrics(nothing_routed)
    assert metrics["routed_hallucination_precision"] is None
    assert metrics["f1"] is None
