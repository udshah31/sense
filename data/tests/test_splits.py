import pytest

from sense_data.splits import (
    LeakageError,
    PerCheckpointSplitIndices,
    SplitIndices,
    assert_no_test_leakage,
    generate_per_checkpoint_splits,
    generate_splits,
    load_per_checkpoint_splits,
    load_splits,
    save_per_checkpoint_splits,
    save_splits,
)

CHECKPOINTS = ["llama3", "mistral", "qwen3_8b", "qwen3_4b", "qwen3_1_7b"]


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


def test_generate_per_checkpoint_splits_is_deterministic():
    a = generate_per_checkpoint_splits(
        n=10_000, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.1,
        development_frac=0.1, test_frac=0.3, seed=42,
    )
    b = generate_per_checkpoint_splits(
        n=10_000, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.1,
        development_frac=0.1, test_frac=0.3, seed=42,
    )
    assert a == b


def test_generate_per_checkpoint_splits_has_one_calibration_set_per_checkpoint():
    splits = generate_per_checkpoint_splits(
        n=10_000, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.1,
        development_frac=0.1, test_frac=0.3, seed=42,
    )
    assert set(splits.calibration.keys()) == set(CHECKPOINTS)
    for key in CHECKPOINTS:
        assert len(splits.calibration[key]) == 1000


def test_generate_per_checkpoint_splits_are_pairwise_disjoint():
    splits = generate_per_checkpoint_splits(
        n=10_000, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.1,
        development_frac=0.1, test_frac=0.3, seed=42,
    )
    dev, test = set(splits.development), set(splits.test)
    assert not (dev & test)

    cal_sets = list(splits.calibration.values())
    for i, a in enumerate(cal_sets):
        assert not (set(a) & dev)
        assert not (set(a) & test)
        for b in cal_sets[i + 1 :]:
            assert not (set(a) & set(b))


def test_generate_per_checkpoint_splits_rejects_insufficient_pool():
    with pytest.raises(ValueError):
        generate_per_checkpoint_splits(
            n=100, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.3,
            development_frac=0.1, test_frac=0.3, seed=0,
        )


def test_generate_per_checkpoint_splits_rejects_dev_test_leaving_no_room():
    with pytest.raises(ValueError):
        generate_per_checkpoint_splits(
            n=100, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.01,
            development_frac=0.5, test_frac=0.5, seed=0,
        )


def test_per_checkpoint_split_indices_rejects_overlap_between_checkpoints():
    with pytest.raises(ValueError):
        PerCheckpointSplitIndices(
            development=[0], test=[1], calibration={"llama3": [2, 3], "mistral": [3, 4]}
        )


def test_per_checkpoint_split_indices_rejects_calibration_overlapping_test():
    with pytest.raises(ValueError):
        PerCheckpointSplitIndices(development=[0], test=[1], calibration={"llama3": [1]})


def test_for_checkpoint_returns_ordinary_split_indices():
    splits = generate_per_checkpoint_splits(
        n=10_000, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.1,
        development_frac=0.1, test_frac=0.3, seed=42,
    )
    view = splits.for_checkpoint("llama3")
    assert isinstance(view, SplitIndices)
    assert view.calibration == splits.calibration["llama3"]
    assert view.development == splits.development
    assert view.test == splits.test


def test_per_checkpoint_save_and_load_round_trip(tmp_path):
    splits = generate_per_checkpoint_splits(
        n=1000, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.1,
        development_frac=0.1, test_frac=0.3, seed=42,
    )
    path = tmp_path / "splits" / "toy_dataset.json"
    save_per_checkpoint_splits(splits, path)

    loaded = load_per_checkpoint_splits(path)
    assert loaded == splits


def test_assert_no_test_leakage_via_for_checkpoint_view_raises_on_test_index():
    splits = generate_per_checkpoint_splits(
        n=1000, checkpoint_keys=CHECKPOINTS, calibration_frac_per_checkpoint=0.1,
        development_frac=0.1, test_frac=0.3, seed=42,
    )
    view = splits.for_checkpoint("llama3")
    leaking_indices = view.calibration[:-1] + [view.test[0]]

    with pytest.raises(LeakageError):
        assert_no_test_leakage(leaking_indices, view)
