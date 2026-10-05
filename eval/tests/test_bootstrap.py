"""Tests for the bootstrap confidence intervals.

Checked against known-answer cases and against invariants (reproducibility, coverage
direction, degenerate handling) rather than against another library, so the
implementation is verified rather than assumed.
"""

import pytest

from sense_eval.bootstrap import (
    ConfidenceInterval,
    MAX_DEGENERATE_FRACTION,
    interval_from_estimates,
    paired_delta_bootstrap,
    percentile_bootstrap,
)


def mean(values):
    return sum(values) / len(values) if values else None


# --- percentile bootstrap --------------------------------------------------------


def test_point_estimate_is_the_statistic_on_the_full_sample():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci = percentile_bootstrap(data, mean, n_resamples=200, seed=1)
    assert ci.point == pytest.approx(3.0)


def test_interval_brackets_the_point_estimate():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    ci = percentile_bootstrap(data, mean, n_resamples=500, seed=1)
    assert ci.low <= ci.point <= ci.high


def test_interval_is_reproducible_for_a_given_seed():
    data = [0.1, 0.5, 0.9, 0.2, 0.7]
    a = percentile_bootstrap(data, mean, n_resamples=200, seed=7)
    b = percentile_bootstrap(data, mean, n_resamples=200, seed=7)
    assert (a.low, a.high) == (b.low, b.high)


def test_different_seeds_give_different_intervals():
    data = [0.1, 0.5, 0.9, 0.2, 0.7]
    a = percentile_bootstrap(data, mean, n_resamples=200, seed=7)
    b = percentile_bootstrap(data, mean, n_resamples=200, seed=8)
    assert (a.low, a.high) != (b.low, b.high)


def test_zero_variance_data_gives_a_zero_width_interval():
    # Every resample of a constant sample is the same constant. A zero-width interval
    # here is a true statement about the data, not a defaulted one.
    ci = percentile_bootstrap([2.0] * 20, mean, n_resamples=100, seed=1)
    assert ci.point == ci.low == ci.high == 2.0


def test_wider_data_gives_a_wider_interval():
    tight = percentile_bootstrap([5.0, 5.1, 4.9, 5.0, 5.05], mean, n_resamples=500, seed=3)
    loose = percentile_bootstrap([0.0, 10.0, 5.0, 1.0, 9.0], mean, n_resamples=500, seed=3)
    assert (loose.high - loose.low) > (tight.high - tight.low)


def test_higher_confidence_gives_a_wider_interval():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    narrow = percentile_bootstrap(data, mean, n_resamples=500, confidence=0.80, seed=2)
    wide = percentile_bootstrap(data, mean, n_resamples=500, confidence=0.99, seed=2)
    assert (wide.high - wide.low) >= (narrow.high - narrow.low)


def test_empty_data_reports_no_interval_rather_than_raising():
    ci = percentile_bootstrap([], mean, n_resamples=100, seed=1)
    assert ci.point is None and ci.low is None and ci.high is None
    assert ci.n_degenerate == 100


def test_a_statistic_undefined_on_the_full_sample_reports_no_interval():
    ci = percentile_bootstrap([1.0, 2.0], lambda values: None, n_resamples=100, seed=1)
    assert ci.point is None and ci.low is None and ci.high is None


def test_too_many_degenerate_resamples_suppresses_the_interval():
    """An interval built from a surviving minority would describe a different
    population than the point estimate, so it is withheld rather than reported."""
    calls = {"n": 0}

    def mostly_undefined(values):
        # Defined on the first call (the point estimate) and then almost never.
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 or calls["n"] % 50 == 0 else None

    ci = percentile_bootstrap([1.0, 2.0, 3.0], mostly_undefined, n_resamples=100, seed=1)
    assert ci.point == 1.0
    assert ci.low is None and ci.high is None
    assert ci.n_degenerate > MAX_DEGENERATE_FRACTION * 100


def test_seed_is_a_required_keyword():
    with pytest.raises(TypeError):
        percentile_bootstrap([1.0, 2.0], mean, n_resamples=10)  # no seed


# --- paired delta ----------------------------------------------------------------


def test_paired_delta_point_is_b_minus_a():
    a = [1.0, 1.0, 1.0, 1.0]
    b = [3.0, 3.0, 3.0, 3.0]
    ci = paired_delta_bootstrap(a, b, mean, n_resamples=100, seed=1)
    assert ci.point == pytest.approx(2.0)


def test_paired_delta_of_identical_sequences_is_exactly_zero_with_zero_width():
    data = [0.1, 0.9, 0.4, 0.7]
    ci = paired_delta_bootstrap(data, data, mean, n_resamples=200, seed=1)
    assert ci.point == 0.0
    assert ci.low == 0.0 and ci.high == 0.0


def test_a_constant_offset_gives_an_interval_excluding_zero():
    # b is a's values plus 5 everywhere, so the difference is 5 in every resample —
    # the signature of an effect far larger than sampling noise.
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [v + 5.0 for v in a]
    ci = paired_delta_bootstrap(a, b, mean, n_resamples=300, seed=1)
    assert ci.low > 0.0


def test_pairing_narrows_the_interval_versus_resampling_independently():
    """The point of paired resampling: correlated conditions get a tighter, correct
    interval. Here b tracks a closely, so the paired difference barely varies while
    each sequence's own mean varies a lot."""
    a = [1.0, 5.0, 9.0, 13.0, 17.0]
    b = [v + 1.0 for v in a]

    paired = paired_delta_bootstrap(a, b, mean, n_resamples=500, seed=4)
    independent_a = percentile_bootstrap(a, mean, n_resamples=500, seed=4)
    independent_b = percentile_bootstrap(b, mean, n_resamples=500, seed=5)
    unpaired_width = (independent_b.high - independent_b.low) + (independent_a.high - independent_a.low)

    assert (paired.high - paired.low) < unpaired_width


def test_misaligned_sequences_raise():
    with pytest.raises(ValueError, match="index-aligned"):
        paired_delta_bootstrap([1.0, 2.0], [1.0], mean, n_resamples=10, seed=1)


# --- interval_from_estimates -----------------------------------------------------


def test_interval_from_estimates_uses_percentile_endpoints():
    estimates = [float(i) for i in range(101)]  # 0..100
    ci = interval_from_estimates(50.0, estimates, n_resamples=101, confidence=0.90)
    # 90% interval -> 5th and 95th percentiles of 0..100.
    assert ci.low == pytest.approx(5.0)
    assert ci.high == pytest.approx(95.0)


def test_interval_from_estimates_counts_the_shortfall_as_degenerate():
    ci = interval_from_estimates(1.0, [1.0] * 95, n_resamples=100, confidence=0.95)
    assert ci.n_degenerate == 5
    assert ci.low is not None  # 5% is within the tolerated fraction


def test_as_dict_shape_is_stable():
    ci = ConfidenceInterval(point=0.5, low=0.4, high=0.6, n_resamples=10, confidence=0.95, n_degenerate=0)
    assert ci.as_dict() == {
        "point": 0.5,
        "ci_low": 0.4,
        "ci_high": 0.6,
        "ci_confidence": 0.95,
        "n_resamples": 10,
        "n_degenerate_resamples": 0,
    }
