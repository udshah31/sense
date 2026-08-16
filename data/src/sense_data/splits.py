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


@dataclass(frozen=True)
class PerCheckpointSplitIndices:
    """Like SplitIndices, but calibration is a distinct index list per checkpoint
    while development/test stay shared across all of them.

    Only worth doing when the dataset is large enough to afford it — HaluEval's
    ~10k-row qa config is (CLAUDE.md non-negotiable constraint #1's "genuine
    held-out calibration splits per checkpoint" requirement, 2026-08-10 scope
    reconciliation). Giving every one of the five GPU checkpoints its own
    calibration subset means a threshold fit for one checkpoint was never fit on
    rows another checkpoint's threshold was also fit on, without eating into the
    shared development/test sets or shrinking any individual checkpoint's
    calibration sample size the way five-way reuse of one small pool would.
    """

    development: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)
    calibration: dict[str, list[int]] = field(default_factory=dict)

    def __post_init__(self):
        dev, test = set(self.development), set(self.test)
        if dev & test:
            raise ValueError("development/test index sets must be disjoint")
        for key, cal in self.calibration.items():
            cal_set = set(cal)
            if cal_set & dev:
                raise ValueError(f"calibration split for {key!r} overlaps the development split")
            if cal_set & test:
                raise ValueError(f"calibration split for {key!r} overlaps the test split")
        cal_sets = list(self.calibration.values())
        for i, a in enumerate(cal_sets):
            for b in cal_sets[i + 1 :]:
                if set(a) & set(b):
                    raise ValueError("calibration splits for different checkpoints must be pairwise disjoint")

    def for_checkpoint(self, checkpoint: str) -> SplitIndices:
        """View as an ordinary SplitIndices for one checkpoint's calibration set,
        so calibration/leakage code (assert_no_test_leakage, GatePolicy.calibrate)
        doesn't need to know about the per-checkpoint shape.
        """
        return SplitIndices(calibration=self.calibration[checkpoint], development=self.development, test=self.test)


def generate_per_checkpoint_splits(
    n: int,
    checkpoint_keys,
    calibration_frac_per_checkpoint: float,
    development_frac: float,
    test_frac: float,
    seed: int,
) -> PerCheckpointSplitIndices:
    """Deterministically partition indices [0, n) into a shared development split,
    a shared test split, and one disjoint calibration subset per checkpoint.

    This is the one-time generation step, same contract as generate_splits: write
    the output with save_per_checkpoint_splits and commit it, never call this at
    experiment run time.
    """
    if development_frac + test_frac >= 1.0:
        raise ValueError("development_frac + test_frac must leave room for at least one calibration split")

    checkpoint_keys = list(checkpoint_keys)
    if not checkpoint_keys:
        raise ValueError("checkpoint_keys must be non-empty")

    remaining_frac = 1.0 - development_frac - test_frac
    needed_frac = calibration_frac_per_checkpoint * len(checkpoint_keys)
    if needed_frac > remaining_frac + 1e-9:
        raise ValueError(
            f"{len(checkpoint_keys)} checkpoints at calibration_frac_per_checkpoint="
            f"{calibration_frac_per_checkpoint} need {needed_frac:.3f} of the dataset, but only "
            f"{remaining_frac:.3f} remains after development_frac + test_frac"
        )

    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_dev = round(n * development_frac)
    n_test = round(n * test_frac)
    development = sorted(indices[:n_dev])
    test = sorted(indices[n_dev : n_dev + n_test])

    pool = indices[n_dev + n_test :]
    n_cal = round(n * calibration_frac_per_checkpoint)

    calibration = {}
    offset = 0
    for key in checkpoint_keys:
        chunk = pool[offset : offset + n_cal]
        if len(chunk) < n_cal:
            raise ValueError(f"not enough remaining examples to build a full calibration split for {key!r}")
        calibration[key] = sorted(chunk)
        offset += n_cal

    return PerCheckpointSplitIndices(development=development, test=test, calibration=calibration)


def save_per_checkpoint_splits(splits: PerCheckpointSplitIndices, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "development": splits.development,
                "test": splits.test,
                "calibration": splits.calibration,
            },
            indent=2,
        )
    )


def load_per_checkpoint_splits(path: Path) -> PerCheckpointSplitIndices:
    data = json.loads(Path(path).read_text())
    return PerCheckpointSplitIndices(
        development=data["development"],
        test=data["test"],
        calibration=data["calibration"],
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
