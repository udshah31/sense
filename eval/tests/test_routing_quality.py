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
