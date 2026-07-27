"""The committed TruthfulQA split file must stay consistent with the dataset it
partitions: disjoint, covers every index, and matches the dataset's current length."""

from pathlib import Path

from sense_data.splits import load_splits
from sense_data.truthful_qa import load_truthful_qa

SPLIT_PATH = Path(__file__).parent.parent / "splits" / "truthful_qa.json"


def test_split_file_exists():
    assert SPLIT_PATH.exists(), "run scripts/generate_truthful_qa_splits.py to create it"


def test_split_file_covers_full_dataset():
    splits = load_splits(SPLIT_PATH)
    n = len(load_truthful_qa())

    all_indices = set(splits.calibration) | set(splits.development) | set(splits.test)
    assert all_indices == set(range(n))


def test_split_file_is_disjoint():
    splits = load_splits(SPLIT_PATH)  # SplitIndices.__post_init__ already enforces this
    assert splits is not None
