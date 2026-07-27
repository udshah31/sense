Python. Fixed calibration/development/test split index lists and loaders.

Splits are generated once via `generate_splits`, written to disk with `save_splits`,
and committed to the repo — not recomputed at runtime (CLAUDE.md's data split
discipline). Downstream code loads them with `load_splits` and must never touch
`.test` indices during threshold fitting; guard calibration code paths with
`assert_no_test_leakage`.

No benchmark dataset is chosen yet (open decision, see `../CLAUDE.md`), so no real
split files are committed here yet — only the generic machinery, exercised by
synthetic indices in `tests/`.

## Setup

```
uv sync
```

## Tests

```
uv run pytest
```

`tests/test_leakage.py` is the split leakage test required by CLAUDE.md: it fails if
calibration code reads test indices.
