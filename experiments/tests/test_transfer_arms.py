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


def test_entropies_on_scale_selects_the_right_scale():
    assert entropies_on_scale(SOURCE_CAL, "normalized") == [0.10, 0.20, 0.30, 0.40, 0.90]
    assert entropies_on_scale(SOURCE_CAL, "raw") == [1.0, 2.0, 3.0, 4.0, 9.0]


def test_unknown_scale_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown entropy scale"):
        entropies_on_scale(SOURCE_CAL, "normalised")  # British spelling typo


def _arm(scale):
    return transfer_arm(scale, SOURCE_CAL, TARGET_CAL, TARGET_EVAL, VIEW, VIEW,
                        0.9, "src", "tgt", LABELS)


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
                       0.9, "tgt", "tgt", LABELS)
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
                       0.9, "src", "tgt", LABELS)
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
