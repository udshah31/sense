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

`latency.py`'s `generate_with_latency` measures time-to-first-token and mean
inter-token latency via `TextIteratorStreamer` running `generate()` in a background
thread while timestamping each decoded piece as it arrives — the only way token
arrival time is directly observable, since `generate()` otherwise blocks until done.
`tests/test_latency.py` confirms the streaming path used for measurement produces
byte-identical output to plain `generate()`, and that attaching the entropy monitor
during latency measurement is still a no-op.
