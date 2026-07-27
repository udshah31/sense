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
