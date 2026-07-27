"""TruthfulQA loader (generation config).

TruthfulQA ships only a single 817-example "validation" split — there is no
train/test distinction upstream. This project's calibration/development/test
partition (see splits.py) is defined entirely on top of it, using the fixed index
order returned here. Do not re-fetch or re-order the dataset elsewhere: any change to
ordering invalidates the committed split index files.
"""

from dataclasses import dataclass

from datasets import load_dataset

DATASET_REPO = "truthfulqa/truthful_qa"
DATASET_CONFIG = "generation"
DATASET_SPLIT = "validation"


@dataclass(frozen=True)
class TruthfulQAExample:
    index: int
    question: str
    best_answer: str
    correct_answers: tuple[str, ...]
    incorrect_answers: tuple[str, ...]
    category: str


def load_truthful_qa() -> list[TruthfulQAExample]:
    """Load TruthfulQA in the project's fixed index order.

    Index `i` here is the same index used by split index lists generated over this
    dataset (see splits.generate_splits) — this function's ordering is the contract
    those lists depend on.
    """
    dataset = load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT)
    return [
        TruthfulQAExample(
            index=i,
            question=row["question"],
            best_answer=row["best_answer"],
            correct_answers=tuple(row["correct_answers"]),
            incorrect_answers=tuple(row["incorrect_answers"]),
            category=row["category"],
        )
        for i, row in enumerate(dataset)
    ]
