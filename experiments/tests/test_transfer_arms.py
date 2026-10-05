"""Pure-logic tests for the RQ1/RQ2 transfer arms.

No models and no generation: transfer_arm and adaptive_arm take already-computed
ExampleSignal lists, so the arm assembly, scale selection, threshold transfer,
agreement computation and routing-quality delta can all be checked directly and in
milliseconds. The slower end-to-end tests in test_transfer_threshold_halueval.py and
test_adaptive_threshold_halueval.py cover the generation path with tiny CPU models.

Added 2026-10-04 alongside the dual-arm change (proposal-v5 review issues C1 and C3,
docs/research/sense-proposal-v5-review.md).
"""
import pytest

from sense_data.splits import SplitIndices
from _common import ExampleSignal, entropies_on_scale
from transfer_threshold_halueval import transfer_arm
from adaptive_threshold_halueval import adaptive_arm


def signals(normalized, raw, texts=None):
    texts = texts or [""] * len(normalized)
    return [ExampleSignal(normalized_entropy=n, raw_entropy=r, generated_text=t)
            for n, r, t in zip(normalized, raw, texts)]


# Raw entropies are deliberately on a different scale from normalized ones, as they
# would be for two tokenizers of different vocabulary size.
SOURCE_CAL = signals([0.10, 0.20, 0.30, 0.40, 0.90], [1.0, 2.0, 3.0, 4.0, 9.0])
TARGET_CAL = signals([0.12, 0.22, 0.32, 0.42, 0.95], [0.6, 1.1, 1.6, 2.1, 4.8])
TARGET_EVAL = signals([0.05, 0.50, 0.95, 0.15, 0.85], [0.3, 2.5, 4.8, 0.8, 4.3])
LABELS = ["correct", "incorrect", "incorrect", "correct", "unknown"]

VIEW = SplitIndices(calibration=[0, 1, 2, 3, 4], development=[5, 6, 7, 8, 9], test=[10, 11, 12])

# Small on purpose: these tests check that intervals are produced and wired, not
# that they are tight. The bootstrap itself is tested in eval/tests/test_bootstrap.py.
BOOTSTRAP = {"n_resamples": 50, "confidence": 0.95, "seed": 42}


def test_entropies_on_scale_selects_the_right_scale():
    assert entropies_on_scale(SOURCE_CAL, "normalized") == [0.10, 0.20, 0.30, 0.40, 0.90]
    assert entropies_on_scale(SOURCE_CAL, "raw") == [1.0, 2.0, 3.0, 4.0, 9.0]


def test_unknown_scale_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown entropy scale"):
        entropies_on_scale(SOURCE_CAL, "normalised")  # British spelling typo


def _arm(scale):
    return transfer_arm(scale, SOURCE_CAL, TARGET_CAL, TARGET_EVAL, VIEW, VIEW,
                        0.9, "src", "tgt", LABELS, BOOTSTRAP)


@pytest.mark.parametrize("scale", ["normalized", "raw"])
def test_transfer_arm_shape_and_invariants(scale):
    arm = _arm(scale)
    assert arm["entropy_scale"] == scale
    # The transferred threshold IS the source's native threshold, never refit.
    assert arm["transferred_threshold"] == arm["source_native_threshold"]
    assert 0.0 <= arm["transfer_agreement_rate"] <= 1.0
    for key in ("transferred_gate", "native_gate"):
        m = arm[key]
        assert m["n_outcomes"] == 5
        # One "unknown" label must be excluded from detection, counted separately.
        assert m["n_scored"] == 4 and m["n_unknown_excluded"] == 1
    assert "delta_f1" in arm["routing_quality_delta"]


def test_the_two_arms_transfer_different_numbers():
    # If normalization and raw produced the same threshold the second arm would be
    # measuring nothing. This is the whole point of issue C3's fix.
    assert _arm("raw")["transferred_threshold"] != _arm("normalized")["transferred_threshold"]


def test_raw_arm_threshold_is_on_the_raw_scale():
    # Source raw calibration is [1..9]; its 0.9 quantile must land in that range,
    # not in the [0, 1] normalized range.
    assert _arm("raw")["source_native_threshold"] > 1.0


def test_transferred_and_native_gates_see_identical_labels():
    # Both gates score against the same generation, so their scored/unknown counts
    # must match exactly; only the routing decisions may differ.
    arm = _arm("normalized")
    t, n = arm["transferred_gate"], arm["native_gate"]
    assert (t["n_scored"], t["n_unknown_excluded"]) == (n["n_scored"], n["n_unknown_excluded"])
    # AUROC is threshold-free: identical for both gates on the same model.
    assert t["entropy_auroc"] == n["entropy_auroc"]


def test_agreement_is_one_when_thresholds_coincide():
    # Transferring a model's threshold to itself must agree perfectly — a sanity
    # check that the agreement computation is wired to the decisions it claims.
    arm = transfer_arm("normalized", TARGET_CAL, TARGET_CAL, TARGET_EVAL, VIEW, VIEW,
                       0.9, "tgt", "tgt", LABELS, BOOTSTRAP)
    assert arm["transfer_agreement_rate"] == 1.0
    assert arm["routing_quality_delta"]["delta_f1"] in (0.0, None)


def test_split_leakage_cannot_be_constructed_let_alone_calibrated_on():
    """Defense in depth, verified at the layer that actually enforces it.

    The arm path derives `calibration_indices` from the same view it validates
    against, so the only way to leak test data into a threshold would be a view whose
    own calibration list overlaps its test list — and SplitIndices refuses to exist in
    that state. GatePolicy.calibrate's assert_no_test_leakage is the second layer,
    covering callers that pass indices from somewhere other than the view.
    """
    from sense_data.splits import assert_no_test_leakage, LeakageError

    with pytest.raises(ValueError, match="pairwise disjoint"):
        SplitIndices(calibration=[10, 11, 12], development=[5, 6], test=[10, 11, 12])

    # Second layer: indices from elsewhere that happen to include test rows.
    with pytest.raises(LeakageError):
        assert_no_test_leakage([4, 10], VIEW)


@pytest.mark.parametrize("scale", ["normalized", "raw"])
def test_adaptive_arm_shape_and_fidelity(scale):
    arm = adaptive_arm(scale, SOURCE_CAL, TARGET_CAL, TARGET_EVAL, VIEW, VIEW,
                       0.9, "src", "tgt", LABELS, BOOTSTRAP)
    assert arm["entropy_scale"] == scale
    assert arm["expected_routing_rate"] == pytest.approx(0.1)
    assert arm["fixed_calibration_fidelity_gap"] >= 0.0
    assert arm["adaptive_calibration_fidelity_gap"] >= 0.0
    for key in ("fixed_gate", "adaptive_gate"):
        assert arm[key]["n_outcomes"] == 5
    assert "delta_f1" in arm["routing_quality_delta"]
    # The adaptive threshold is fit on the target's own calibration split, using the
    # same linear-interpolation quantile GatePolicy uses. Computed here independently
    # rather than read back from the implementation.
    values = sorted(entropies_on_scale(TARGET_CAL, scale))
    position = 0.9 * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    expected = values[lower] + (values[upper] - values[lower]) * (position - lower)
    assert arm["adaptive_threshold"] == pytest.approx(expected)


# --- confidence intervals are wired through the arms ---------------------------


@pytest.mark.parametrize("scale", ["normalized", "raw"])
def test_transfer_arm_reports_threshold_and_delta_intervals(scale):
    arm = _arm(scale)

    for key in ("source_threshold_ci", "target_native_threshold_ci"):
        interval = arm[key]
        assert interval["n_resamples"] == 50
        if interval["ci_low"] is not None:
            assert interval["ci_low"] <= interval["point"] <= interval["ci_high"]

    # The threshold interval's point must be the threshold actually used.
    assert arm["source_threshold_ci"]["point"] == arm["source_native_threshold"]

    # Paired delta intervals, which are the primary evidence for RQ1.
    assert "delta_f1_ci" in arm["routing_quality_delta_ci"]
    assert "delta_entropy_auroc_ci" not in arm["routing_quality_delta_ci"]

    # Per-metric intervals on each gate.
    for gate_key in ("transferred_gate", "native_gate"):
        assert "f1_ci" in arm[gate_key]
        assert "entropy_auroc_ci" in arm[gate_key]


def test_self_transfer_gives_zero_width_or_honestly_suppressed_delta_intervals():
    """Transferring a model's threshold to itself: both gates make identical decisions,
    so every paired difference is exactly zero.

    Where an interval forms it must have zero width. Where it does not, it must have
    been suppressed for a stated reason — more than MAX_DEGENERATE_FRACTION of
    resamples left the metric undefined — rather than silently dropped. With this
    5-example fixture routing a single example, roughly a third of resamples draw no
    routed example at all, so `routed_hallucination_precision` genuinely cannot be
    estimated and the module declines to invent a number for it. That is the behavior
    under test, not a shortcoming of it.
    """
    from sense_eval.bootstrap import MAX_DEGENERATE_FRACTION

    arm = transfer_arm("normalized", TARGET_CAL, TARGET_CAL, TARGET_EVAL, VIEW, VIEW,
                       0.9, "tgt", "tgt", LABELS, BOOTSTRAP)

    saw_an_interval = False
    for key, interval in arm["routing_quality_delta_ci"].items():
        if interval["point"] is not None:
            assert interval["point"] == 0.0, key

        if interval["ci_low"] is not None:
            saw_an_interval = True
            assert interval["ci_low"] == 0.0 and interval["ci_high"] == 0.0, key
        else:
            assert (
                interval["point"] is None
                or interval["n_degenerate_resamples"] > MAX_DEGENERATE_FRACTION * interval["n_resamples"]
            ), f"{key}: interval absent without a stated reason"

    assert saw_an_interval, "every delta interval was suppressed — fixture is too small to test anything"


@pytest.mark.parametrize("scale", ["normalized", "raw"])
def test_adaptive_arm_reports_both_threshold_intervals(scale):
    arm = adaptive_arm(scale, SOURCE_CAL, TARGET_CAL, TARGET_EVAL, VIEW, VIEW,
                       0.9, "src", "tgt", LABELS, BOOTSTRAP)
    assert arm["fixed_threshold_ci"]["point"] == arm["fixed_threshold"]
    assert arm["adaptive_threshold_ci"]["point"] == arm["adaptive_threshold"]
    assert "delta_f1_ci" in arm["routing_quality_delta_ci"]
