One script per table/figure in the paper.

## Setup

```
uv sync
```

## calibrate_gate_truthful_qa.py

Wires `GatePolicy` against real TruthfulQA calibration entropies: for each example
in the committed calibration split, greedily generates a short answer with
`TokenEntropyMonitor` attached and reduces its per-token trace to a mean-entropy
scalar, then fits the gate threshold on those scalars (leakage-guarded against the
committed split). Model identity, quantile, and decoding config all come from
`../configs/model.yaml` / `../configs/gate.yaml` — swapping `active` in `model.yaml`
to `llama3`/`mistral` on the GPU environment runs the identical code path.

```
uv run python calibrate_gate_truthful_qa.py
```

Writes `../results/gate_calibration_truthful_qa_<model_name>.json`.

## transfer_threshold_truthful_qa.py

RQ1 harness: calibrates natively on the source model, calibrates a separate native
baseline on the target model (comparison only, never applied), then applies the
source's threshold directly to the target model via `GatePolicy.set_threshold` and
compares its routing decisions against the target's own native gate. Evaluated on
the **development** split only — test stays untouched until the final reported
numbers. Model pair and eval split come from `../configs/rq1.yaml`;
`cpu_test_transfer_target` in `../configs/model.yaml` is a second CPU-testable model
from a different family (GPT-NeoX, 1024-token vocab) standing in for "a model from a
different family" the way llama3/mistral do on the GPU environment.

```
uv run python transfer_threshold_truthful_qa.py
```

Writes `../results/rq1_transfer_truthful_qa_<source>_to_<target>.json`.

## Tests

```
uv run pytest
```

Runs the same wiring on a small subset (for speed) plus a leakage-refusal check,
rather than the full 327-example calibration split.
