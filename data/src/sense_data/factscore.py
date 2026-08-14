"""FActScore loader (biography-generation benchmark).

500 entities, each with generation prompts at three lengths (one-sentence,
~100-word, full biography) plus reference Wikipedia text for the entity — the
atomic-fact decomposition front-end shared with the Z3 symbolic backend
(CLAUDE.md's design decisions) operates on generations produced from these
prompts, checked against the reference text.

Do not re-fetch or re-order the dataset elsewhere: any change to ordering
invalidates the committed split index file (see splits.py and
scripts/generate_factscore_splits.py).
"""

from dataclasses import dataclass

from datasets import load_dataset

DATASET_REPO = "dskar/FActScore"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_REVISION = "991ea43a24a2b6abb73949ba84102494d1e0a58f"  # pinned 2026-08-13, main branch HEAD


@dataclass(frozen=True)
class FActScoreExample:
    index: int
    entity: str
    one_fact_prompt: str
    factscore_prompt: str
    hundredw_prompt: str
    around_100: str
    wikipedia_text: str


def load_factscore() -> list[FActScoreExample]:
    """Load FActScore in the project's fixed index order.

    Index `i` here is the same index used by the split index list generated over
    this dataset (see scripts/generate_factscore_splits.py) — this function's
    ordering is the contract that list depends on.
    """
    dataset = load_dataset(DATASET_REPO, DATASET_CONFIG, split=DATASET_SPLIT, revision=DATASET_REVISION)
    return [
        FActScoreExample(
            index=i,
            entity=row["entity"],
            one_fact_prompt=row["one_fact_prompt"],
            factscore_prompt=row["factscore_prompt"],
            hundredw_prompt=row["hundredw_prompt"],
            around_100=row["around_100"],
            wikipedia_text=row["wikipedia_text"],
        )
        for i, row in enumerate(dataset)
    ]
