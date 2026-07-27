import math

import pytest

from sense_orchestrator.truncated_entropy import truncated_entropy_bounds


def test_lower_never_exceeds_upper():
    lower, upper = truncated_entropy_bounds([0.5, 0.2, 0.1], vocab_size=1000)
    assert lower <= upper


def test_uniform_full_distribution_bounds_collapse_to_true_entropy():
    v = 100
    uniform = [1.0 / v] * v
    lower, upper = truncated_entropy_bounds(uniform, vocab_size=v)
    true_entropy = math.log(v)
    assert lower == pytest.approx(true_entropy, abs=1e-9)
    assert upper == pytest.approx(true_entropy, abs=1e-9)


def test_rejects_empty_topk():
    with pytest.raises(ValueError):
        truncated_entropy_bounds([], vocab_size=100)


def test_rejects_topk_longer_than_vocab():
    with pytest.raises(ValueError):
        truncated_entropy_bounds([0.5, 0.5, 0.1], vocab_size=2)


def test_near_zero_remaining_mass_gives_tight_bounds():
    # top-k already accounts for ~all probability mass
    topk = [0.5, 0.3, 0.1999999999]
    lower, upper = truncated_entropy_bounds(topk, vocab_size=50000)
    assert upper - lower < 1e-6
