"""The committed HaluEval split file must stay consistent with the dataset it
partitions and with configs/model.yaml's GPU checkpoint list: development/test/
each checkpoint's calibration set pairwise disjoint, every index in range, and one
calibration set per currently-configured GPU checkpoint (CLAUDE.md non-negotiable
constraint #1's per-checkpoint held-out calibration requirement)."""

from pathlib import Path

import yaml

from sense_data.halueval import load_halueval
from sense_data.splits import load_per_checkpoint_splits

SPLIT_PATH = Path(__file__).parent.parent / "splits" / "halueval.json"
MODEL_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "model.yaml"


def test_split_file_exists():
    assert SPLIT_PATH.exists(), "run scripts/generate_halueval_splits.py to create it"


def test_split_file_indices_are_within_dataset_range():
    splits = load_per_checkpoint_splits(SPLIT_PATH)
    n = len(load_halueval())

    all_indices = set(splits.development) | set(splits.test)
    for calibration in splits.calibration.values():
        all_indices |= set(calibration)
    assert all_indices <= set(range(n))


def test_split_file_has_one_calibration_set_per_gpu_checkpoint():
    splits = load_per_checkpoint_splits(SPLIT_PATH)
    model_cfg = yaml.safe_load(MODEL_CONFIG_PATH.read_text())
    gpu_checkpoint_keys = {key for key in model_cfg["models"] if not key.startswith("cpu_test")}

    assert set(splits.calibration.keys()) == gpu_checkpoint_keys


def test_split_file_is_pairwise_disjoint():
    splits = load_per_checkpoint_splits(SPLIT_PATH)  # __post_init__ already enforces this
    assert splits is not None
