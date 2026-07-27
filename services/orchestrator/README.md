Python. Gate policy, routing, symbolic interface, instrumentation. Talks to the neural
service only over HTTP — no in-process imports from services/neural — to keep the
routing logic decoupled from model plumbing.

## Setup

```
uv sync
```

## Tests

```
uv run pytest
```

`gate.py`'s `GatePolicy` has no default threshold and refuses to run (`decide` raises
`UncalibratedGateError`) until explicitly calibrated via `calibrate()` (fit a quantile
from calibration-split entropies, leakage-checked against `sense_data.splits`) or
`set_threshold()` (an already-fit threshold, e.g. transferred from another model for
the RQ1 fixed-threshold-transfer experiment).

`truncated_entropy.py` bounds true token entropy from a truncated top-k logprob
response (the vLLM/OpenAI-compatible serving path). `tests/test_truncated_entropy.py`
is the required correctness check that the exact in-process value (via `sense-neural`,
a test-only dependency — the orchestrator never imports it at runtime) falls within
those bounds.

`router.py`'s `route_and_annotate` ties the gate to the symbolic backend: evaluates
`GatePolicy.decide`, and if it fires, calls `symbolic_client.verify_triple` over HTTP
(base URL from `../configs/symbolic.yaml`) and attaches the response as an
annotation — merge-back is annotate-only for now (CLAUDE.md's design decisions), so
this never modifies generation output. Records per-stage latency (gate evaluation,
symbolic round-trip, total). A symbolic-backend failure is recorded in
`symbolic_error`, not raised, so a flaky external dependency can't take down
generation; an uncalibrated gate still raises, since that's a hard failure by design.
`tests/test_router.py` exercises this against the real `sense-symbolic` app
in-process via ASGI transport (test-only dependency — no live network hop, no
mocking of the router's own logic).
