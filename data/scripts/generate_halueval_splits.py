"""One-time generation of the committed HaluEval split file: a shared
development/test split plus a disjoint calibration subset per GPU checkpoint.
Run this only to regenerate ../splits/halueval.json — every other part of the
project must load that committed file via splits.load_per_checkpoint_splits,
never call generate_per_checkpoint_splits on this dataset directly.

Usage: uv run python scripts/generate_halueval_splits.py
"""

from pathlib import Path

import yaml

from sense_data.halueval import load_halueval
from sense_data.splits import generate_per_checkpoint_splits, save_per_checkpoint_splits

# Fixed seed, recorded here per CLAUDE.md's reproducibility requirements. Fractions:
# each of the five GPU checkpoints gets its own 10%-of-dataset calibration subset
# (1,000 rows) — affordable only because HaluEval's qa config (10,000 rows) is large
# enough to carve out five disjoint calibration slices plus a shared development and
# test split without any of them going thin. Test held largest (30%) for a stable
# final report; development smallest (10%), since it's only for engineering
# iteration, not a reported number.
SEED = 42
DEVELOPMENT_FRAC = 0.1
TEST_FRAC = 0.3
CALIBRATION_FRAC_PER_CHECKPOINT = 0.1

REPO_ROOT = Path(__file__).parent.parent.parent
OUTPUT_PATH = Path(__file__).parent.parent / "splits" / "halueval.json"


def gpu_checkpoint_keys() -> list[str]:
    """The five GPU checkpoints from configs/model.yaml — cpu_test/cpu_test_transfer_target
    are CPU-only stand-ins and are excluded, matching experiments/_common.py's
    GPU_CHECKPOINT_KEYS.
    """
    model_cfg = yaml.safe_load((REPO_ROOT / "configs" / "model.yaml").read_text())
    return [key for key in model_cfg["models"] if not key.startswith("cpu_test")]


def main() -> None:
    examples = load_halueval()
    checkpoint_keys = gpu_checkpoint_keys()
    splits = generate_per_checkpoint_splits(
        n=len(examples),
        checkpoint_keys=checkpoint_keys,
        calibration_frac_per_checkpoint=CALIBRATION_FRAC_PER_CHECKPOINT,
        development_frac=DEVELOPMENT_FRAC,
        test_frac=TEST_FRAC,
        seed=SEED,
    )
    save_per_checkpoint_splits(splits, OUTPUT_PATH)
    cal_sizes = {key: len(indices) for key, indices in splits.calibration.items()}
    print(
        f"wrote {OUTPUT_PATH}: {len(splits.development)} development, {len(splits.test)} test, "
        f"calibration per checkpoint: {cal_sizes} (n={len(examples)}, seed={SEED})"
    )


if __name__ == "__main__":
    main()
