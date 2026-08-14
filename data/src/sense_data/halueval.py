"""HaluEval loader (QA hallucination-detection config).

HaluEval ships four task configs (qa, dialogue, summarization, general) totaling
~35k samples (CLAUDE.md's stated volume); this project uses the `qa` config
(10,000 rows) since it matches the generation-based, per-example question/answer
shape the entropy gate and RQ1-RQ3 harnesses already work with — each row pairs a
source `knowledge` passage with a `right_answer` and a model-authored
`hallucinated_answer` for the same question.

Do not re-fetch or re-order the dataset elsewhere: any change to ordering
invalidates the committed split index files (see splits.py and
scripts/generate_halueval_splits.py).
"""

from dataclasses import dataclass

from datasets import load_dataset

DATASET_REPO = "pminervini/HaluEval"
DATASET_CONFIG = "qa"
DATASET_SPLIT = "data"
DATASET_REVISION = "12a856119f03975a94509091e8cada3e6be6ead7"  # pinned 2026-08-13, main branch HEAD


@dataclass(frozen=True)
class HaluEvalExample:
    index: int
    knowledge: str
    question: str
    right_answer: str
    hallucinated_answer: str


def load_halueval() -> list[HaluEvalExample]:
    """Load HaluEval's qa config in the project's fixed index order.

    Index `i` here is the same index used by split index lists generated over this
    dataset (see splits.generate_per_checkpoint_splits) — this function's ordering
    is the contract those lists depend on.
    """
    dataset = load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
    return [
        HaluEvalExample(
            index=i,
            knowledge=row["knowledge"],
            question=row["question"],
            right_answer=row["right_answer"],
            hallucinated_answer=row["hallucinated_answer"],
        )
        for i, row in enumerate(dataset)
    ]
