Python. Fixed calibration/development/test split index lists and loaders.

Splits are generated once via `generate_splits`, written to disk with `save_splits`,
and committed to the repo — not recomputed at runtime (CLAUDE.md's data split
discipline). Downstream code loads them with `load_splits` and must never touch
`.test` indices during threshold fitting; guard calibration code paths with
`assert_no_test_leakage`.

Benchmark: TruthfulQA (`truthful_qa.py`, generation config, 817 examples — see
`../CLAUDE.md`'s design decisions). Its committed split is `splits/truthful_qa.json`,
generated once by `scripts/generate_truthful_qa_splits.py` (seed 42, 40/20/40
calibration/development/test). Regenerating that script's output is a deliberate,
rare action — everything else loads the committed file via `splits.load_splits`.

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
