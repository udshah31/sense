import pytest

from sense_data.splits import SplitIndices, generate_splits, load_splits, save_splits


def test_generate_splits_is_deterministic():
    a = generate_splits(n=200, calibration_frac=0.5, development_frac=0.25, test_frac=0.25, seed=42)
    b = generate_splits(n=200, calibration_frac=0.5, development_frac=0.25, test_frac=0.25, seed=42)
    assert a == b


def test_generate_splits_different_seed_differs():
    a = generate_splits(n=200, calibration_frac=0.5, development_frac=0.25, test_frac=0.25, seed=1)
    b = generate_splits(n=200, calibration_frac=0.5, development_frac=0.25, test_frac=0.25, seed=2)
    assert a != b


def test_splits_are_disjoint_and_cover_all_indices():
    n = 500
    splits = generate_splits(n=n, calibration_frac=0.5, development_frac=0.3, test_frac=0.2, seed=7)
    cal, dev, test = set(splits.calibration), set(splits.development), set(splits.test)

    assert not (cal & dev)
    assert not (cal & test)
    assert not (dev & test)
    assert cal | dev | test == set(range(n))


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        generate_splits(n=100, calibration_frac=0.5, development_frac=0.3, test_frac=0.3, seed=0)


def test_split_indices_rejects_overlapping_sets():
    with pytest.raises(ValueError):
        SplitIndices(calibration=[1, 2, 3], development=[3, 4], test=[5])


def test_save_and_load_round_trip(tmp_path):
    splits = generate_splits(n=100, calibration_frac=0.6, development_frac=0.2, test_frac=0.2, seed=3)
    path = tmp_path / "splits" / "toy_dataset.json"
    save_splits(splits, path)

    loaded = load_splits(path)
    assert loaded == splits
