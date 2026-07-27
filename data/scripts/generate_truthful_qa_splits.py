"""One-time generation of the committed calibration/development/test split for
TruthfulQA. Run this only to regenerate ../splits/truthful_qa.json — every other
part of the project must load that committed file via splits.load_splits, never
call generate_splits on this dataset directly.

Usage: uv run python scripts/generate_truthful_qa_splits.py
"""

from pathlib import Path

from sense_data.splits import generate_splits, save_splits
from sense_data.truthful_qa import load_truthful_qa

# Fixed seed, recorded here per CLAUDE.md's reproducibility requirements. Fractions:
# calibration gets the largest reliable share since threshold fitting is the most
# leakage-sensitive step; test held large enough (~40%) for a stable final factuality/
# latency report; development is intentionally the smallest, since it's only for
# engineering iteration, not a reported number.
SEED = 42
CALIBRATION_FRAC = 0.4
DEVELOPMENT_FRAC = 0.2
TEST_FRAC = 0.4

OUTPUT_PATH = Path(__file__).parent.parent / "splits" / "truthful_qa.json"


def main() -> None:
    examples = load_truthful_qa()
    splits = generate_splits(
        n=len(examples),
        calibration_frac=CALIBRATION_FRAC,
        development_frac=DEVELOPMENT_FRAC,
        test_frac=TEST_FRAC,
        seed=SEED,
    )
    save_splits(splits, OUTPUT_PATH)
    print(
        f"wrote {OUTPUT_PATH}: "
        f"{len(splits.calibration)} calibration, "
        f"{len(splits.development)} development, "
        f"{len(splits.test)} test "
        f"(n={len(examples)}, seed={SEED})"
    )


if __name__ == "__main__":
    main()
