"""Split leakage test, required by CLAUDE.md: must fail if calibration code reads
test indices.
"""

import pytest

from sense_data.splits import LeakageError, assert_no_test_leakage, generate_splits


@pytest.fixture
def splits():
    return generate_splits(n=100, calibration_frac=0.5, development_frac=0.25, test_frac=0.25, seed=0)


def test_calibration_reading_only_calibration_indices_is_allowed(splits):
    assert_no_test_leakage(splits.calibration, splits)  # must not raise


def test_calibration_reading_development_indices_is_allowed(splits):
    assert_no_test_leakage(splits.development, splits)  # must not raise


def test_calibration_reading_any_test_index_raises(splits):
    leaking_access = [splits.calibration[0], splits.test[0]]
    with pytest.raises(LeakageError):
        assert_no_test_leakage(leaking_access, splits)


def test_calibration_reading_all_test_indices_raises(splits):
    with pytest.raises(LeakageError):
        assert_no_test_leakage(splits.test, splits)
