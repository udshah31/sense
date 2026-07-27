Python. Model loading, LogitsProcessor-based entropy monitor, hidden-state probes. Must run
against a small CPU-friendly model for tests; GPU-only for Llama-3/Mistral runs.

## Setup

Dependencies are pinned in `pyproject.toml` / `uv.lock`, managed with [uv](https://docs.astral.sh/uv/).

```
uv sync
```

## Tests

```
uv run pytest
```

`tests/test_environment.py` is a network-free smoke test confirming the env (torch,
transformers, numpy, the `sense_neural` package) is importable and CPU tensor ops work.
Model-dependent tests (entropy monitor no-op check, truncated-vs-exact entropy bounds)
belong here too, running against a tiny CPU-friendly model per the project's testing rule
in `../../CLAUDE.md`.
