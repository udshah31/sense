"""Fixed calibration/development/test split index lists.

Splits are defined by index lists generated once and committed to the repo (see
CLAUDE.md's data split discipline), not recomputed at runtime. `generate_splits` is
the one-time generation step; `load_splits`/`save_splits` read and write the committed
JSON files that are the actual source of truth for every experiment run afterward.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path


class LeakageError(Exception):
    """Raised when a code path reads test-split indices during calibration."""


@dataclass(frozen=True)
class SplitIndices:
    calibration: list[int] = field(default_factory=list)
    development: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)

    def __post_init__(self):
        cal, dev, test = set(self.calibration), set(self.development), set(self.test)
        if cal & dev or cal & test or dev & test:
            raise ValueError("split index sets must be pairwise disjoint")


def generate_splits(
    n: int,
    calibration_frac: float,
    development_frac: float,
    test_frac: float,
    seed: int,
) -> SplitIndices:
    """Deterministically partition indices [0, n) into three disjoint splits.

    This is the one-time generation step. Its output is meant to be written to disk
    with `save_splits` and committed — downstream code must load the committed file
    via `load_splits`, never call this at experiment run time.
    """
    if abs((calibration_frac + development_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_cal = round(n * calibration_frac)
    n_dev = round(n * development_frac)

    calibration = sorted(indices[:n_cal])
    development = sorted(indices[n_cal : n_cal + n_dev])
    test = sorted(indices[n_cal + n_dev :])

    return SplitIndices(calibration=calibration, development=development, test=test)


def save_splits(splits: SplitIndices, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "calibration": splits.calibration,
                "development": splits.development,
                "test": splits.test,
            },
            indent=2,
        )
    )


def load_splits(path: Path) -> SplitIndices:
    data = json.loads(Path(path).read_text())
    return SplitIndices(
        calibration=data["calibration"],
        development=data["development"],
        test=data["test"],
    )


def assert_no_test_leakage(accessed_indices, splits: SplitIndices) -> None:
    """Guard for calibration code paths: raise if any accessed index is in the test split.

    Call this wherever threshold-fitting code receives a set of indices, passing the
    indices it is about to read, to catch leakage immediately rather than letting it
    produce a clean-looking but invalid result.
    """
    leaked = set(accessed_indices) & set(splits.test)
    if leaked:
        raise LeakageError(
            f"calibration code attempted to read {len(leaked)} test-split index(es): "
            f"{sorted(leaked)[:10]}{'...' if len(leaked) > 10 else ''}"
        )
