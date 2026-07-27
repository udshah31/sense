import pytest

from sense_data.splits import generate_splits
from sense_orchestrator.gate import GatePolicy, UncalibratedGateError


@pytest.fixture
def splits():
    return generate_splits(n=100, calibration_frac=0.5, development_frac=0.25, test_frac=0.25, seed=0)


def test_decide_raises_when_uncalibrated():
    gate = GatePolicy()
    assert not gate.is_calibrated
    with pytest.raises(UncalibratedGateError):
        gate.decide(0.5)


def test_threshold_raises_when_uncalibrated():
    gate = GatePolicy()
    with pytest.raises(UncalibratedGateError):
        _ = gate.threshold


def test_calibrate_on_calibration_indices_succeeds(splits):
    gate = GatePolicy()
    entropies = [i / 100 for i in range(len(splits.calibration))]
    threshold = gate.calibrate(
        calibration_entropies=entropies,
        calibration_indices=splits.calibration,
        splits=splits,
        quantile=0.9,
    )
    assert gate.is_calibrated
    assert gate.threshold == threshold
    assert gate.calibration_source == "native"


def test_calibrate_raises_on_test_index_leakage(splits):
    gate = GatePolicy()
    leaking_indices = splits.calibration[:-1] + [splits.test[0]]
    entropies = [0.5] * len(leaking_indices)
    with pytest.raises(Exception):  # LeakageError, re-exported via sense_data.splits
        gate.calibrate(
            calibration_entropies=entropies,
            calibration_indices=leaking_indices,
            splits=splits,
            quantile=0.9,
        )
    assert not gate.is_calibrated  # failed calibration must not leave a threshold set


def test_calibrate_rejects_invalid_quantile(splits):
    gate = GatePolicy()
    with pytest.raises(ValueError):
        gate.calibrate(
            calibration_entropies=[0.1, 0.2],
            calibration_indices=splits.calibration[:2],
            splits=splits,
            quantile=1.5,
        )


def test_decide_routes_above_threshold_only():
    gate = GatePolicy()
    gate.set_threshold(0.7, source="transferred-from-llama3")
    assert gate.decide(0.9) is True
    assert gate.decide(0.7) is True  # boundary: >= threshold routes
    assert gate.decide(0.5) is False


def test_set_threshold_requires_explicit_source():
    gate = GatePolicy()
    gate.set_threshold(0.6, source="native")
    assert gate.calibration_source == "native"
    assert gate.is_calibrated
