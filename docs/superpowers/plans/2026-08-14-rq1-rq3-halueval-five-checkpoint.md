# RQ1/RQ2/RQ3 → HaluEval, Five-Checkpoint Scope Implementation Plan

> **Status: COMPLETE.** All 8 tasks implemented, reviewed, and merged to `master` via
> PR #1 (`cc0f02b`, 2026-08-16). Final review escalated two bugs in
> `generate_with_latency` to Critical (missing Qwen3 non-thinking chat template,
> missing `.to(model.device)` for 4-bit models) — fixed and verified green in CI
> before merge. See `git log --oneline` from `cc0f02b` for the full commit trail.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-point the RQ1 (fixed-threshold transfer), RQ2 (self-adaptive threshold), and RQ3 (accuracy-latency trade-off) experiment harnesses from TruthfulQA/two-model to HaluEval/five-checkpoint, matching the 2026-08-10 scope reconciliation already reflected in `configs/model.yaml`, `data/splits/halueval.json`, and CLAUDE.md — without touching TruthfulQA's loader/data (kept for the excluded-benchmark discussion) or dragging FActScore into these three RQs (it has no per-checkpoint calibration split and belongs to the symbolic backend's decomposition work, not gate calibration).

**Architecture:** Each harness gains a per-checkpoint calibration view via `PerCheckpointSplitIndices.for_checkpoint(name)` (already implemented in `data/src/sense_data/splits.py`) instead of one shared `SplitIndices.calibration`. RQ1/RQ2 move from a single hardcoded source/target pair to a `transfer_pairs` list in their YAML config, looping over pairs and writing one result file per pair. RQ3 moves from a single hardcoded model to a `models` list, looping and writing one result file per model. Models are loaded and explicitly freed (`del` + `torch.cuda.empty_cache()`) around each pair/model iteration so peak GPU memory never holds more than two checkpoints at once, regardless of how many pairs/models the config lists.

**Tech Stack:** Python 3.13, pytest, HF `datasets`/`transformers`, existing `sense_data`/`sense_orchestrator`/`sense_eval` packages — no new dependencies.

## Global Constraints

- Never fit a threshold on anything but a model's own calibration split (CLAUDE.md's data-split discipline) — every calibration call in this plan uses `splits.for_checkpoint(<that model's name>).calibration`, never a shared or another checkpoint's list.
- Quantization stays 4-bit for every GPU checkpoint (already true in `configs/model.yaml`; `load_model_registry()` already asserts it — do not weaken that check).
- TruthfulQA's loader (`data/src/sense_data/truthful_qa.py`), its committed split (`data/splits/truthful_qa.json`), and `experiments/calibrate_gate_truthful_qa.py` are NOT touched by this plan — they stay for the excluded-benchmark discussion.
- FActScore is NOT wired into RQ1/RQ2/RQ3 by this plan — out of scope, see architecture note above.
- No silent subsampling: if a config caps `n_eval_examples` below the available split size, the harness must print that it's doing so (RQ3 already does this — preserve it).
- Every result written still goes through `write_results`, which enforces `task_accuracy`/`hallucination_rate`/`abstention_rate` being reported together (only applies to RQ3, which reports factuality; RQ1/RQ2 report routing agreement/fidelity, not factuality, and must NOT gain those three keys).

---

## Task 1: Shared HaluEval loading helper in `experiments/_common.py`

**Files:**
- Modify: `experiments/_common.py`
- Modify: `experiments/rag_baseline_halueval.py` (dedupe its local copy of this logic)
- Test: `experiments/tests/test_common.py`

**Interfaces:**
- Produces: `load_halueval_examples_and_splits() -> tuple[list[HaluEvalExample], PerCheckpointSplitIndices]` in `experiments/_common.py`, used by every task after this one.

- [x] **Step 1: Write the failing test**

Add to `experiments/tests/test_common.py`:

```python
from _common import load_halueval_examples_and_splits
from sense_data.halueval import HaluEvalExample
from sense_data.splits import PerCheckpointSplitIndices


def test_load_halueval_examples_and_splits_returns_real_data():
    examples, splits = load_halueval_examples_and_splits()

    assert len(examples) == 10_000
    assert isinstance(examples[0], HaluEvalExample)
    assert isinstance(splits, PerCheckpointSplitIndices)
    assert len(splits.development) > 0
    assert len(splits.test) > 0
    assert set(splits.calibration.keys()) == {"llama3", "mistral", "qwen3_8b", "qwen3_4b", "qwen3_1_7b"}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_common.py::test_load_halueval_examples_and_splits_returns_real_data -v`
Expected: FAIL with `ImportError: cannot import name 'load_halueval_examples_and_splits'`

- [x] **Step 3: Implement the helper**

In `experiments/_common.py`, add near the existing `load_examples_and_splits`:

```python
from sense_data.halueval import HaluEvalExample, load_halueval
from sense_data.splits import PerCheckpointSplitIndices, load_per_checkpoint_splits
```

(add to the existing `from sense_data.splits import ...` line rather than a second import line)

```python
HALUEVAL_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "halueval.json"


def load_halueval_examples_and_splits() -> tuple[list[HaluEvalExample], PerCheckpointSplitIndices]:
    return load_halueval(), load_per_checkpoint_splits(HALUEVAL_SPLIT_PATH)
```

- [x] **Step 4: Run test to verify it passes**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_common.py::test_load_halueval_examples_and_splits_returns_real_data -v`
Expected: PASS (hits the real HF Hub — HaluEval is public, no token needed)

- [x] **Step 5: Dedupe `rag_baseline_halueval.py`**

`experiments/rag_baseline_halueval.py` currently opens with (verified against the current file — these are the exact lines to change, nothing else in the file references `load_halueval`, `load_per_checkpoint_splits`, or `HALUEVAL_SPLIT_PATH`):

```python
from _common import REPO_ROOT, load_model, load_model_registry, load_yaml_config, write_results
from sense_data.halueval import load_halueval
from sense_data.splits import load_per_checkpoint_splits
from sense_eval.factuality import (
    PLACEHOLDER_FACTUALITY_METRIC_LABEL,
    FactualityVerdict,
    lexical_containment_verdict,
    summarize_factuality,
)
from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve

HALUEVAL_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "halueval.json"


def load_examples_and_splits():
    return load_halueval(), load_per_checkpoint_splits(HALUEVAL_SPLIT_PATH)
```

Replace it with:

```python
from _common import REPO_ROOT, load_halueval_examples_and_splits as load_examples_and_splits
from _common import load_model, load_model_registry, load_yaml_config, write_results
from sense_eval.factuality import (
    PLACEHOLDER_FACTUALITY_METRIC_LABEL,
    FactualityVerdict,
    lexical_containment_verdict,
    summarize_factuality,
)
from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve
```

This drops the `sense_data.halueval`/`sense_data.splits` imports, the `HALUEVAL_SPLIT_PATH` constant, and the local `load_examples_and_splits` function body entirely. Keep `REPO_ROOT` — it's still used later in the file at `PassageIndex.from_file(REPO_ROOT / config["rag"]["corpus_path"])`. Everywhere else in the file that calls `load_examples_and_splits()` keeps working unchanged, since the imported name is identical.

Note: `from _common import REPO_ROOT, load_halueval_examples_and_splits as load_examples_and_splits` — check this parses; if the `as` inside a multi-name `from X import a, b as c` line reads awkwardly, split it onto its own line instead:

```python
from _common import REPO_ROOT, load_model, load_model_registry, load_yaml_config, write_results
from _common import load_halueval_examples_and_splits as load_examples_and_splits
```

Either form is fine — pick whichever the linter/formatter in use doesn't flag.

- [x] **Step 6: Run the RAG baseline test suite to confirm the dedupe didn't break anything**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_rag_baseline_halueval.py -v`
Expected: PASS (same tests as before — this step only moved code, didn't change behavior)

- [x] **Step 7: Commit**

```bash
git add experiments/_common.py experiments/rag_baseline_halueval.py experiments/tests/test_common.py
git commit -m "Add shared load_halueval_examples_and_splits helper, dedupe RAG baseline"
```

---

## Task 2: RQ1 → `configs/rq1.yaml` multi-pair shape

**Files:**
- Modify: `configs/rq1.yaml`

**Interfaces:**
- Produces: `config["rq1"]["transfer_pairs"]` (list of `{source_model, target_model}` dicts) and `config["rq1"]["eval_split"]` (string), consumed by Task 3.

- [x] **Step 1: Replace the file contents**

```yaml
# RQ1 — does a fixed entropy-gating threshold, calibrated on one model, transfer to
# a model from a different family or a different scale? Both axes named in
# CLAUDE.md get tested: three cross-family anchors (llama3, mistral, qwen3_8b) and
# the Qwen3 scale ladder (qwen3_8b -> qwen3_4b -> qwen3_1_7b, largest-to-smallest —
# the direction a self-adaptive threshold would need to beat a fixed one on).
#
# Each entry is one directed source->target transfer run; the harness writes one
# result file per pair (rq1_transfer_halueval_<source>_to_<target>.json). This is
# five pairs, not the full 20-pair matrix across five checkpoints — deliberately
# contained scope (CLAUDE.md: "the symbolic backend can quietly become a research
# project of its own" applies just as much to blowing up the transfer matrix).
# Add reverse-direction pairs here if the advisor wants fuller coverage; don't add
# them silently.
transfer_pairs:
  - {source_model: llama3, target_model: mistral}
  - {source_model: llama3, target_model: qwen3_8b}
  - {source_model: mistral, target_model: qwen3_8b}
  - {source_model: qwen3_8b, target_model: qwen3_4b}
  - {source_model: qwen3_8b, target_model: qwen3_1_7b}

# Transfer is evaluated on the development split, never test — test is touched once,
# at the end, for the numbers that go in the paper (CLAUDE.md's data split discipline).
eval_split: development
```

- [x] **Step 2: Verify it parses**

Run: `python3 -c "import yaml; c = yaml.safe_load(open('configs/rq1.yaml')); print(c['transfer_pairs']); print(c['eval_split'])"`
Expected: prints the 5-item list and `development`

- [x] **Step 3: Commit**

```bash
git add configs/rq1.yaml
git commit -m "Reshape RQ1 config to a five-pair transfer list over HaluEval"
```

---

## Task 3: RQ1 harness → `experiments/transfer_threshold_halueval.py`

**Files:**
- Create: `experiments/transfer_threshold_halueval.py`
- Delete: `experiments/transfer_threshold_truthful_qa.py`
- Create: `experiments/tests/test_transfer_threshold_halueval.py` (rename from `test_transfer_threshold_truthful_qa.py`, update imports/docstring only)
- Delete: `experiments/tests/test_transfer_threshold_truthful_qa.py`

**Interfaces:**
- Consumes: `load_model_registry()`, `load_yaml_config()`, `load_model()`, `calibration_entropies()`, `write_results()` from `experiments/_common.py`; `load_halueval_examples_and_splits()` from Task 1; `GatePolicy` from `sense_orchestrator.gate`; `PerCheckpointSplitIndices.for_checkpoint(name) -> SplitIndices` from `sense_data.splits`.
- Produces: `run_pair(models_registry, examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name) -> dict` and `run() -> list[dict]`, both importable by tests.

- [x] **Step 1: Write the failing test**

Create `experiments/tests/test_transfer_threshold_halueval.py`:

```python
"""Integration test for the RQ1 transfer harness, on a small subset for speed —
transfer_threshold_halueval.py itself runs the full committed per-checkpoint
splits. Uses tiny CPU-testable models standing in for real checkpoints (CLAUDE.md:
every component must be testable on CPU with a tiny model) — the real five-
checkpoint run only happens on the GPU environment.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import calibration_entropies
from sense_data.splits import generate_splits
from sense_data.truthful_qa import load_truthful_qa
from sense_orchestrator.gate import GatePolicy
from transfer_threshold_halueval import run_pair

SOURCE_MODEL = "sshleifer/tiny-gpt2"
TARGET_MODEL = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 5}
SOURCE_MODEL_CFG = {"hf_repo": SOURCE_MODEL}
TARGET_MODEL_CFG = {"hf_repo": TARGET_MODEL}


@pytest.fixture(scope="module")
def source():
    return AutoModelForCausalLM.from_pretrained(SOURCE_MODEL), AutoTokenizer.from_pretrained(SOURCE_MODEL)


@pytest.fixture(scope="module")
def target():
    return AutoModelForCausalLM.from_pretrained(TARGET_MODEL), AutoTokenizer.from_pretrained(TARGET_MODEL)


@pytest.fixture(scope="module")
def small_splits():
    examples = load_truthful_qa()
    return generate_splits(n=len(examples), calibration_frac=0.02, development_frac=0.02, test_frac=0.96, seed=0)


def test_transfer_harness_runs_end_to_end(source, target, small_splits):
    source_model, source_tokenizer = source
    target_model, target_tokenizer = target
    examples = load_truthful_qa()

    cal_subset = small_splits.calibration[:5]
    dev_subset = small_splits.development[:5]

    source_entropies = calibration_entropies(
        source_model, source_tokenizer, examples, cal_subset, DECODING_CFG, SOURCE_MODEL_CFG
    )
    source_gate = GatePolicy()
    source_threshold = source_gate.calibrate(
        calibration_entropies=source_entropies,
        calibration_indices=cal_subset,
        splits=small_splits,
        quantile=0.9,
        source="source",
    )

    target_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, cal_subset, DECODING_CFG, TARGET_MODEL_CFG
    )
    target_native_gate = GatePolicy()
    target_native_gate.calibrate(
        calibration_entropies=target_entropies,
        calibration_indices=cal_subset,
        splits=small_splits,
        quantile=0.9,
        source="target",
    )

    transferred_gate = GatePolicy()
    transferred_gate.set_threshold(source_threshold, source="transferred-from-source")

    eval_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, dev_subset, DECODING_CFG, TARGET_MODEL_CFG
    )

    transferred_decisions = [transferred_gate.decide(h) for h in eval_entropies]
    native_decisions = [target_native_gate.decide(h) for h in eval_entropies]

    assert len(transferred_decisions) == len(dev_subset)
    assert all(isinstance(d, bool) for d in transferred_decisions)
    assert all(isinstance(d, bool) for d in native_decisions)
    assert transferred_gate.calibration_source == "transferred-from-source"


def test_transferred_threshold_equals_source_native_threshold(source, small_splits):
    source_model, source_tokenizer = source
    examples = load_truthful_qa()
    cal_subset = small_splits.calibration[:5]

    entropies = calibration_entropies(
        source_model, source_tokenizer, examples, cal_subset, DECODING_CFG, SOURCE_MODEL_CFG
    )
    source_gate = GatePolicy()
    threshold = source_gate.calibrate(
        calibration_entropies=entropies,
        calibration_indices=cal_subset,
        splits=small_splits,
        quantile=0.9,
        source="source",
    )

    transferred_gate = GatePolicy()
    transferred_gate.set_threshold(threshold, source="transferred-from-source")
    assert transferred_gate.threshold == threshold


def test_run_pair_end_to_end_on_tiny_models_via_halueval_shaped_splits():
    """Exercises run_pair itself (not just the underlying primitives above),
    against a tiny per-checkpoint-shaped split built from real HaluEval data —
    the actual code path the real five-checkpoint run uses, just with cpu_test
    stand-ins and a tiny slice for speed.
    """
    from sense_data.halueval import load_halueval
    from sense_data.splits import PerCheckpointSplitIndices

    examples = load_halueval()[:50]
    splits = PerCheckpointSplitIndices(
        development=list(range(40, 50)),
        test=list(range(30, 40)),
        calibration={"source": list(range(0, 10)), "target": list(range(10, 20))},
    )
    models_registry = {"source": SOURCE_MODEL_CFG, "target": TARGET_MODEL_CFG}

    result = run_pair(
        models_registry, examples, splits, DECODING_CFG, quantile=0.9,
        eval_split="development", source_name="source", target_name="target",
    )

    assert result["research_question"] == "RQ1"
    assert result["dataset"] == "halueval"
    assert result["source_model"]["name"] == "source"
    assert result["target_model"]["name"] == "target"
    assert 0.0 <= result["transfer_agreement_rate"] <= 1.0
    assert result["n_eval_examples"] == 10
```

Note: `test_run_pair_end_to_end_on_tiny_models_via_halueval_shaped_splits` passes plain string keys `"source"`/`"target"` (not real checkpoint names like `"llama3"`) — `run_pair` must not hardcode the five real checkpoint names anywhere; it only ever reads `source_name`/`target_name` as parameters and looks them up in `splits.calibration`/`models_registry`.

- [x] **Step 2: Run test to verify it fails**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_transfer_threshold_halueval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'transfer_threshold_halueval'`

- [x] **Step 3: Write `experiments/transfer_threshold_halueval.py`**

```python
"""RQ1 harness: does a fixed entropy-gating threshold, calibrated on one model,
transfer to a model from a different family or scale?

Runs once per (source_model, target_model) pair in configs/rq1.yaml's
transfer_pairs list, writing one result file per pair. Each checkpoint calibrates
on its own disjoint calibration subset (data/splits/halueval.json's per-checkpoint
shape — CLAUDE.md's "genuine held-out calibration splits per checkpoint"
requirement) rather than a shared calibration list.

Procedure per pair:
  1. Calibrate a gate natively on the source model's own calibration-split
     entropies — this is the threshold being tested for transfer.
  2. Calibrate a second gate natively on the target model's own calibration-split
     entropies — this is the target's "ground truth" threshold, used only as a
     comparison baseline, never as what actually gets applied.
  3. Apply the source's threshold directly to the target model (no refitting) via
     GatePolicy.set_threshold, and compare its routing decisions against the
     target's native gate on the development split.

Evaluated on development, never test (CLAUDE.md's data split discipline — test is
touched once, at the end, for the reported numbers).

Models are loaded and explicitly freed around each pair so peak GPU memory never
holds more than the current pair's two checkpoints, regardless of how many pairs
the config lists (some checkpoints, e.g. qwen3_8b, appear in more than one pair).
"""

import gc

import torch

from _common import calibration_entropies, load_halueval_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq1": load_yaml_config("rq1.yaml"),
    }


def run_pair(models_registry, examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name) -> dict:
    source_cfg = models_registry[source_name]
    target_cfg = models_registry[target_name]

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)
    try:
        source_view = splits.for_checkpoint(source_name)
        target_view = splits.for_checkpoint(target_name)
        eval_indices = getattr(splits, eval_split)

        # 1. Native source calibration — this threshold is what gets transferred.
        source_gate = GatePolicy()
        source_cal_entropies = calibration_entropies(
            source_model, source_tokenizer, examples, source_view.calibration, decoding_cfg, source_cfg
        )
        source_threshold = source_gate.calibrate(
            calibration_entropies=source_cal_entropies,
            calibration_indices=source_view.calibration,
            splits=source_view,
            quantile=quantile,
            source=source_name,
        )

        # 2. Native target calibration — comparison baseline only, never applied.
        target_native_gate = GatePolicy()
        target_cal_entropies = calibration_entropies(
            target_model, target_tokenizer, examples, target_view.calibration, decoding_cfg, target_cfg
        )
        target_native_threshold = target_native_gate.calibrate(
            calibration_entropies=target_cal_entropies,
            calibration_indices=target_view.calibration,
            splits=target_view,
            quantile=quantile,
            source=target_name,
        )

        # 3. Transferred gate: source's threshold applied directly to the target model.
        transferred_gate = GatePolicy()
        transferred_gate.set_threshold(source_threshold, source=f"transferred-from-{source_name}")

        target_eval_entropies = calibration_entropies(
            target_model, target_tokenizer, examples, eval_indices, decoding_cfg, target_cfg
        )

        transferred_decisions = [transferred_gate.decide(h) for h in target_eval_entropies]
        native_decisions = [target_native_gate.decide(h) for h in target_eval_entropies]

        agreement = sum(t == n for t, n in zip(transferred_decisions, native_decisions)) / len(eval_indices)

        return {
            "research_question": "RQ1",
            "dataset": "halueval",
            "eval_split": eval_split,
            "n_eval_examples": len(eval_indices),
            "quantile": quantile,
            "decoding": decoding_cfg,
            "source_model": {
                "name": source_name,
                "hf_repo": source_cfg["hf_repo"],
                "revision": source_cfg["revision"],
                "native_threshold": source_threshold,
            },
            "target_model": {
                "name": target_name,
                "hf_repo": target_cfg["hf_repo"],
                "revision": target_cfg["revision"],
                "native_threshold": target_native_threshold,
            },
            "transferred_threshold": source_threshold,
            "transferred_routing_rate": sum(transferred_decisions) / len(eval_indices),
            "native_routing_rate": sum(native_decisions) / len(eval_indices),
            "transfer_agreement_rate": agreement,
        }
    finally:
        del source_model, target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run() -> list[dict]:
    config = load_config()
    examples, splits = load_halueval_examples_and_splits()
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    eval_split = config["rq1"]["eval_split"]

    results = []
    for pair in config["rq1"]["transfer_pairs"]:
        source_name, target_name = pair["source_model"], pair["target_model"]
        result = run_pair(
            config["models"], examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name
        )
        write_results(f"rq1_transfer_halueval_{source_name}_to_{target_name}.json", result)
        results.append(result)
    return results


if __name__ == "__main__":
    run()
```

- [x] **Step 4: Delete the superseded TruthfulQA-scoped harness and test**

```bash
rm experiments/transfer_threshold_truthful_qa.py experiments/tests/test_transfer_threshold_truthful_qa.py
```

- [x] **Step 5: Run tests to verify they pass**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_transfer_threshold_halueval.py -v`
Expected: PASS (4 tests)

- [x] **Step 6: Run the full experiments suite to check nothing else references the deleted file**

Run: `cd experiments && .venv/bin/python -m pytest tests/ -q`
Expected: no collection errors, all non-GPU tests pass

- [x] **Step 7: Commit**

```bash
git add experiments/transfer_threshold_halueval.py experiments/tests/test_transfer_threshold_halueval.py
git rm experiments/transfer_threshold_truthful_qa.py experiments/tests/test_transfer_threshold_truthful_qa.py
git commit -m "Re-point RQ1 harness to HaluEval, five-checkpoint transfer pairs"
```

---

## Task 4: RQ2 → `configs/rq2.yaml` multi-pair shape

**Files:**
- Modify: `configs/rq2.yaml`

**Interfaces:**
- Produces: `config["rq2"]["transfer_pairs"]`, `config["rq2"]["eval_split"]` — same shape as Task 2, consumed by Task 5.

- [x] **Step 1: Replace the file contents**

```yaml
# RQ2 — does a self-adaptive threshold (recalibrated on the target model's own
# calibration data) preserve routing quality where a fixed (transferred) threshold
# does not?
#
# "Routing quality preserved" is operationalized as: how close the routing rate on
# held-out data lands to the quantile's own target rate (1 - quantile), which is
# what the threshold was calibrated to produce. A gate whose held-out routing rate
# tracks its intended rate is behaving as calibrated; one that doesn't has failed to
# transfer, whether or not that gap is due to a genuinely worse model fit.
#
# Same five pairs as configs/rq1.yaml (cross-family anchors + Qwen3 scale ladder,
# CLAUDE.md's two transfer axes) — kept as a separate file since it's consumed by a
# separate harness, not because the content differs.
transfer_pairs:
  - {source_model: llama3, target_model: mistral}
  - {source_model: llama3, target_model: qwen3_8b}
  - {source_model: mistral, target_model: qwen3_8b}
  - {source_model: qwen3_8b, target_model: qwen3_4b}
  - {source_model: qwen3_8b, target_model: qwen3_1_7b}

eval_split: development
```

- [x] **Step 2: Verify it parses**

Run: `python3 -c "import yaml; c = yaml.safe_load(open('configs/rq2.yaml')); print(c['transfer_pairs']); print(c['eval_split'])"`
Expected: prints the 5-item list and `development`

- [x] **Step 3: Commit**

```bash
git add configs/rq2.yaml
git commit -m "Reshape RQ2 config to a five-pair transfer list over HaluEval"
```

---

## Task 5: RQ2 harness → `experiments/adaptive_threshold_halueval.py`

**Files:**
- Create: `experiments/adaptive_threshold_halueval.py`
- Delete: `experiments/adaptive_threshold_truthful_qa.py`
- Create: `experiments/tests/test_adaptive_threshold_halueval.py` (rename from `test_adaptive_threshold_truthful_qa.py`, update imports/docstring only)
- Delete: `experiments/tests/test_adaptive_threshold_truthful_qa.py`

**Interfaces:**
- Consumes: same as Task 3, plus `routing_rate(gate, entropies) -> float` (unchanged helper, moves with the file).
- Produces: `routing_rate(gate, entropies) -> float`, `run_pair(models_registry, examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name) -> dict`, `run() -> list[dict]`.

- [x] **Step 1: Write the failing test**

Create `experiments/tests/test_adaptive_threshold_halueval.py`:

```python
"""Integration test for the RQ2 adaptive-threshold harness, on a small subset for
speed — adaptive_threshold_halueval.py itself runs the full committed
per-checkpoint splits. Uses tiny CPU-testable models standing in for real
checkpoints (CLAUDE.md: every component must be testable on CPU with a tiny
model).
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import calibration_entropies
from adaptive_threshold_halueval import routing_rate, run_pair
from sense_data.splits import generate_splits
from sense_data.truthful_qa import load_truthful_qa
from sense_orchestrator.gate import GatePolicy

SOURCE_MODEL = "sshleifer/tiny-gpt2"
TARGET_MODEL = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 5}
SOURCE_MODEL_CFG = {"hf_repo": SOURCE_MODEL}
TARGET_MODEL_CFG = {"hf_repo": TARGET_MODEL}


@pytest.fixture(scope="module")
def source():
    return AutoModelForCausalLM.from_pretrained(SOURCE_MODEL), AutoTokenizer.from_pretrained(SOURCE_MODEL)


@pytest.fixture(scope="module")
def target():
    return AutoModelForCausalLM.from_pretrained(TARGET_MODEL), AutoTokenizer.from_pretrained(TARGET_MODEL)


@pytest.fixture(scope="module")
def small_splits():
    examples = load_truthful_qa()
    return generate_splits(n=len(examples), calibration_frac=0.02, development_frac=0.02, test_frac=0.96, seed=0)


def test_adaptive_gate_calibrates_independently_of_fixed_gate(source, target, small_splits):
    source_model, source_tokenizer = source
    target_model, target_tokenizer = target
    examples = load_truthful_qa()
    cal_subset = small_splits.calibration[:5]
    dev_subset = small_splits.development[:5]

    source_entropies = calibration_entropies(
        source_model, source_tokenizer, examples, cal_subset, DECODING_CFG, SOURCE_MODEL_CFG
    )
    source_gate = GatePolicy()
    source_threshold = source_gate.calibrate(
        calibration_entropies=source_entropies,
        calibration_indices=cal_subset,
        splits=small_splits,
        quantile=0.9,
        source="source",
    )

    fixed_gate = GatePolicy()
    fixed_gate.set_threshold(source_threshold, source="transferred-from-source")

    target_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, cal_subset, DECODING_CFG, TARGET_MODEL_CFG
    )
    adaptive_gate = GatePolicy()
    adaptive_threshold = adaptive_gate.calibrate(
        calibration_entropies=target_entropies,
        calibration_indices=cal_subset,
        splits=small_splits,
        quantile=0.9,
        source="adaptive-target",
    )

    assert adaptive_gate.calibration_source == "adaptive-target"
    assert fixed_gate.calibration_source == "transferred-from-source"

    eval_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, dev_subset, DECODING_CFG, TARGET_MODEL_CFG
    )
    fixed_rate = routing_rate(fixed_gate, eval_entropies)
    adaptive_rate = routing_rate(adaptive_gate, eval_entropies)

    assert 0.0 <= fixed_rate <= 1.0
    assert 0.0 <= adaptive_rate <= 1.0
    assert adaptive_threshold != source_threshold


def test_routing_rate_helper_matches_manual_count():
    gate = GatePolicy()
    gate.set_threshold(0.5, source="test")
    entropies = [0.1, 0.6, 0.9, 0.2, 0.5]
    assert routing_rate(gate, entropies) == 3 / 5


def test_run_pair_end_to_end_on_tiny_models_via_halueval_shaped_splits():
    from sense_data.halueval import load_halueval
    from sense_data.splits import PerCheckpointSplitIndices

    examples = load_halueval()[:50]
    splits = PerCheckpointSplitIndices(
        development=list(range(40, 50)),
        test=list(range(30, 40)),
        calibration={"source": list(range(0, 10)), "target": list(range(10, 20))},
    )
    models_registry = {"source": SOURCE_MODEL_CFG, "target": TARGET_MODEL_CFG}

    result = run_pair(
        models_registry, examples, splits, DECODING_CFG, quantile=0.9,
        eval_split="development", source_name="source", target_name="target",
    )

    assert result["research_question"] == "RQ2"
    assert result["dataset"] == "halueval"
    assert 0.0 <= result["fixed_dev_routing_rate"] <= 1.0
    assert 0.0 <= result["adaptive_dev_routing_rate"] <= 1.0
    assert result["n_eval_examples"] == 10
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_adaptive_threshold_halueval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'adaptive_threshold_halueval'`

- [x] **Step 3: Write `experiments/adaptive_threshold_halueval.py`**

```python
"""RQ2 harness: does a self-adaptive threshold — recalibrated on the target model's
own calibration data — preserve routing quality on held-out data where a fixed
(transferred) threshold does not?

"Routing quality preserved" is operationalized as calibration fidelity: a threshold
calibrated at quantile q is designed to route roughly (1 - q) of calibration-split
examples. On held-out (development-split) data, the adaptive gate's routing rate
should track that same (1 - q) target closely, since it was fit on that model's own
entropy distribution. The fixed gate — the source model's threshold applied
unmodified — has no such guarantee, since it was fit on a different model's
distribution entirely.

Runs once per (source_model, target_model) pair in configs/rq2.yaml's
transfer_pairs list, writing one result file per pair. Each checkpoint calibrates
on its own disjoint calibration subset (data/splits/halueval.json's per-checkpoint
shape), same as RQ1.

Models are loaded and explicitly freed around each pair so peak GPU memory never
holds more than the current pair's two checkpoints.
"""

import gc

import torch

from _common import calibration_entropies, load_halueval_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq2": load_yaml_config("rq2.yaml"),
    }


def routing_rate(gate: GatePolicy, entropies: list[float]) -> float:
    return sum(gate.decide(h) for h in entropies) / len(entropies)


def run_pair(models_registry, examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name) -> dict:
    source_cfg = models_registry[source_name]
    target_cfg = models_registry[target_name]
    expected_routing_rate = 1.0 - quantile

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)
    try:
        source_view = splits.for_checkpoint(source_name)
        target_view = splits.for_checkpoint(target_name)
        eval_indices = getattr(splits, eval_split)

        # Source native calibration — what gets transferred to produce the fixed gate.
        source_gate = GatePolicy()
        source_cal_entropies = calibration_entropies(
            source_model, source_tokenizer, examples, source_view.calibration, decoding_cfg, source_cfg
        )
        source_threshold = source_gate.calibrate(
            calibration_entropies=source_cal_entropies,
            calibration_indices=source_view.calibration,
            splits=source_view,
            quantile=quantile,
            source=source_name,
        )

        # Fixed condition: source's threshold applied unmodified to the target model.
        fixed_gate = GatePolicy()
        fixed_gate.set_threshold(source_threshold, source=f"transferred-from-{source_name}")

        # Adaptive condition: recalibrated natively on the target model's own
        # calibration-split entropies.
        adaptive_gate = GatePolicy()
        target_cal_entropies = calibration_entropies(
            target_model, target_tokenizer, examples, target_view.calibration, decoding_cfg, target_cfg
        )
        adaptive_threshold = adaptive_gate.calibrate(
            calibration_entropies=target_cal_entropies,
            calibration_indices=target_view.calibration,
            splits=target_view,
            quantile=quantile,
            source=f"adaptive-{target_name}",
        )

        target_eval_entropies = calibration_entropies(
            target_model, target_tokenizer, examples, eval_indices, decoding_cfg, target_cfg
        )

        fixed_dev_rate = routing_rate(fixed_gate, target_eval_entropies)
        adaptive_dev_rate = routing_rate(adaptive_gate, target_eval_entropies)

        return {
            "research_question": "RQ2",
            "dataset": "halueval",
            "eval_split": eval_split,
            "n_eval_examples": len(eval_indices),
            "quantile": quantile,
            "expected_routing_rate": expected_routing_rate,
            "decoding": decoding_cfg,
            "source_model": {"name": source_name, "hf_repo": source_cfg["hf_repo"], "revision": source_cfg["revision"]},
            "target_model": {"name": target_name, "hf_repo": target_cfg["hf_repo"], "revision": target_cfg["revision"]},
            "fixed_threshold": source_threshold,
            "fixed_dev_routing_rate": fixed_dev_rate,
            "fixed_calibration_fidelity_gap": abs(fixed_dev_rate - expected_routing_rate),
            "adaptive_threshold": adaptive_threshold,
            "adaptive_dev_routing_rate": adaptive_dev_rate,
            "adaptive_calibration_fidelity_gap": abs(adaptive_dev_rate - expected_routing_rate),
        }
    finally:
        del source_model, target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run() -> list[dict]:
    config = load_config()
    examples, splits = load_halueval_examples_and_splits()
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    eval_split = config["rq2"]["eval_split"]

    results = []
    for pair in config["rq2"]["transfer_pairs"]:
        source_name, target_name = pair["source_model"], pair["target_model"]
        result = run_pair(
            config["models"], examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name
        )
        write_results(f"rq2_adaptive_halueval_{source_name}_to_{target_name}.json", result)
        results.append(result)
    return results


if __name__ == "__main__":
    run()
```

- [x] **Step 4: Delete the superseded TruthfulQA-scoped harness and test**

```bash
rm experiments/adaptive_threshold_truthful_qa.py experiments/tests/test_adaptive_threshold_truthful_qa.py
```

- [x] **Step 5: Run tests to verify they pass**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_adaptive_threshold_halueval.py -v`
Expected: PASS (3 tests)

- [x] **Step 6: Run the full experiments suite**

Run: `cd experiments && .venv/bin/python -m pytest tests/ -q`
Expected: no collection errors, all non-GPU tests pass

- [x] **Step 7: Commit**

```bash
git add experiments/adaptive_threshold_halueval.py experiments/tests/test_adaptive_threshold_halueval.py
git rm experiments/adaptive_threshold_truthful_qa.py experiments/tests/test_adaptive_threshold_truthful_qa.py
git commit -m "Re-point RQ2 harness to HaluEval, five-checkpoint transfer pairs"
```

---

## Task 6: RQ3 → `configs/rq3.yaml` multi-model shape

**Files:**
- Modify: `configs/rq3.yaml`

**Interfaces:**
- Produces: `config["rq3"]["models"]` (list of model-key strings), `config["rq3"]["eval_split"]`, `config["rq3"]["n_eval_examples"]`, `config["rq3"]["symbolic_probe"]` — consumed by Task 7.

- [x] **Step 1: Replace the file contents**

```yaml
# RQ3 — what accuracy-latency trade-off does routing introduce, and does it hold
# across all checkpoints? Loops over every entry in `models`; the harness writes
# one result file per model (rq3_accuracy_latency_halueval_<model>.json).
#
# n_eval_examples caps the development split. HaluEval's per-checkpoint split
# (data/splits/halueval.json) has 1,000 development-split rows; this runs the full
# development split on the GPU environment — no CPU-tractability subsampling
# applies here. Explicit and logged, not a silent cap. This is real compute: five
# checkpoints x 1,000 examples x 2 generations (gated + ungated) each — lower this
# deliberately (and note why in the run notes) if a first pass needs to be cheaper.
models: [llama3, mistral, qwen3_8b, qwen3_4b, qwen3_1_7b]
eval_split: development
n_eval_examples: 1000

# The symbolic backend verifies (subject, predicate, object) triples, but this
# project has no entity/relation extractor yet to turn a free-text HaluEval
# question into one (that's future work, tied to the still-undecided merge-back
# "replace" policy). This fixed, always-resolvable triple stands in for "some real
# claim" so the round-trip latency measured is real cost against the real backend
# (the Z3 symbolic backend by default — see services/symbolic) — it is NOT a claim
# about the HaluEval question being verified. Do not read symbolic_result content
# from this harness as a factuality signal. (Chosen deliberately: Albert
# Einstein/physicist/P106 is also a fact in the Z3 backend's fixed domain KB —
# services/symbolic/src/sense_symbolic/domain.py — so it resolves to verified=true
# under the default backend, not just a latency no-op.)
symbolic_probe:
  subject_label: Albert Einstein
  predicate_pid: P106
  object_label: physicist
```

- [x] **Step 2: Verify it parses**

Run: `python3 -c "import yaml; c = yaml.safe_load(open('configs/rq3.yaml')); print(c['models']); print(c['n_eval_examples'])"`
Expected: prints the 5-item list and `1000`

- [x] **Step 3: Commit**

```bash
git add configs/rq3.yaml
git commit -m "Reshape RQ3 config to loop over all five checkpoints on HaluEval"
```

---

## Task 7: RQ3 harness → `experiments/rq3_accuracy_latency_halueval.py`

**Files:**
- Create: `experiments/rq3_accuracy_latency_halueval.py`
- Delete: `experiments/rq3_accuracy_latency_truthful_qa.py`
- Create: `experiments/tests/test_rq3_accuracy_latency_halueval.py` (rename from `test_rq3_accuracy_latency_truthful_qa.py`, update imports/docstring + swap the factuality-verdict test to HaluEval's field names)
- Delete: `experiments/tests/test_rq3_accuracy_latency_truthful_qa.py`

**Interfaces:**
- Consumes: `load_halueval_examples_and_splits()`, `load_model()`, `load_model_registry()`, `load_yaml_config()`, `write_results()` from `_common.py`; `summarize_factuality`, `FactualityVerdict`, `lexical_containment_verdict`, `PLACEHOLDER_FACTUALITY_METRIC_LABEL` from `sense_eval.factuality`; `TokenEntropyMonitor` from `sense_neural.entropy`; `generate_with_latency` from `sense_neural.latency`; `GatePolicy`, `route_and_annotate` from `sense_orchestrator`.
- Produces: `SYMBOLIC_HOST`, `SYMBOLIC_PORT`, `SYMBOLIC_BASE_URL` module constants (unchanged names — Task 7's test imports these, same as the old file); `run_experiment_for_model(config, model_name) -> dict`; `run_all(config) -> list[dict]`; `main() -> list[dict]`.

- [x] **Step 1: Write the failing test**

Create `experiments/tests/test_rq3_accuracy_latency_halueval.py`:

```python
"""Integration test for the RQ3 harness on a tiny subset — the real script runs
against the full configured n_eval_examples across all five checkpoints. Starts a
real local symbolic server, same as the harness itself, rather than mocking the
round-trip.
"""

import asyncio

import pytest
import uvicorn
from sense_symbolic.app import app as symbolic_app
from transformers import AutoModelForCausalLM, AutoTokenizer

from rq3_accuracy_latency_halueval import SYMBOLIC_BASE_URL, SYMBOLIC_HOST, SYMBOLIC_PORT
from sense_data.halueval import load_halueval
from sense_data.splits import generate_splits
from sense_eval.factuality import lexical_containment_verdict
from sense_neural.entropy import TokenEntropyMonitor
from sense_neural.latency import generate_with_latency
from sense_orchestrator.gate import GatePolicy
from sense_orchestrator.router import route_and_annotate

TINY_MODEL = "sshleifer/tiny-gpt2"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 5}


@pytest.fixture
async def running_symbolic_server():
    config = uvicorn.Config(symbolic_app, host=SYMBOLIC_HOST, port=SYMBOLIC_PORT, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    yield
    server.should_exit = True
    await task


async def test_gated_and_ungated_generation_are_identical(running_symbolic_server):
    """The core RQ3 invariant: annotate-only merge-back must never change output."""
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    examples = load_halueval()
    question = examples[0].question

    ungated = generate_with_latency(model, tokenizer, question, DECODING_CFG)

    monitor = TokenEntropyMonitor(vocab_size=tokenizer.vocab_size)
    gated = generate_with_latency(model, tokenizer, question, DECODING_CFG, logits_processor=[monitor])

    assert gated["text"] == ungated["text"]

    entropy = sum(monitor.entropies) / len(monitor.entropies)
    gate = GatePolicy()
    gate.set_threshold(0.0, source="test")  # always routes, to exercise the symbolic call

    annotation = await route_and_annotate(
        gate,
        entropy,
        symbolic_base_url=SYMBOLIC_BASE_URL,
        subject_label="Albert Einstein",
        predicate_pid="P106",
        object_label="physicist",
    )
    assert annotation.routed is True
    assert annotation.symbolic_latency_ms is not None


def test_factuality_verdict_is_computed_on_ungated_text():
    examples = load_halueval()
    example = examples[0]
    verdict = lexical_containment_verdict(example.right_answer, example.right_answer, (), (example.hallucinated_answer,))
    assert verdict.label == "correct"


def test_eval_split_subsampling_respects_requested_count():
    examples = load_halueval()
    splits = generate_splits(n=len(examples), calibration_frac=0.4, development_frac=0.2, test_frac=0.4, seed=0)
    subset = splits.development[:5]
    assert len(subset) == 5
    assert set(subset).issubset(set(splits.development))
```

- [x] **Step 2: Run test to verify it fails**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_rq3_accuracy_latency_halueval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rq3_accuracy_latency_halueval'`

- [x] **Step 3: Write `experiments/rq3_accuracy_latency_halueval.py`**

```python
"""RQ3 harness: what accuracy-latency trade-off does routing introduce, and does it
hold across all five checkpoints?

Starts the real symbolic backend (services/symbolic, Z3 by default) as a live local
server once for the whole run, so the routed condition's symbolic round-trip is
genuine HTTP over loopback for every checkpoint — the same contract the
orchestrator uses in docker-compose, not an in-process shortcut.

For each checkpoint in configs/rq3.yaml's `models` list, and for each
development-split example, generates twice under identical greedy decoding
(CLAUDE.md's decoding-config constraint): once "ungated" (no monitor, no routing)
and once "gated" (entropy monitor attached, gate evaluated, symbolic round-trip on
routed examples). Since merge-back is annotate-only (CLAUDE.md's design decisions),
the two generations must produce byte-identical text — gating never touches
generation output. That equality is asserted, not just hoped for.

Each checkpoint calibrates its own gate on its own disjoint calibration subset
(data/splits/halueval.json's per-checkpoint shape), then is loaded, run, and freed
before the next checkpoint loads — so peak GPU memory only ever holds one
checkpoint at a time regardless of how many the config lists.

The symbolic round-trip uses a fixed, always-resolvable probe triple
(configs/rq3.yaml) rather than one derived from the question — this project has no
free-text-to-triple extractor yet (out of scope, tied to the undecided merge-back
"replace" policy). It measures real network/round-trip cost against the real
backend; it is not a factuality check of the question and must not be read as one.

Factuality is scored with eval/'s lexical-containment proxy — a placeholder, not a
paper-grade judge (see eval/README.md). Reported as task_accuracy/hallucination_rate/
abstention_rate together (never a subset — write_results enforces this), even
though abstention_rate is always 0 here: merge-back is annotate-only for now, so
nothing in this harness can abstain.
"""

import asyncio
import gc

import torch
import uvicorn
from sense_symbolic.app import app as symbolic_app

from _common import load_halueval_examples_and_splits, load_model, load_model_registry, load_yaml_config, mean_calibration_entropy, write_results
from sense_eval.factuality import (
    PLACEHOLDER_FACTUALITY_METRIC_LABEL,
    FactualityVerdict,
    lexical_containment_verdict,
    summarize_factuality,
)
from sense_neural.entropy import TokenEntropyMonitor
from sense_neural.latency import generate_with_latency
from sense_orchestrator.gate import GatePolicy
from sense_orchestrator.router import route_and_annotate

SYMBOLIC_HOST = "127.0.0.1"
SYMBOLIC_PORT = 8001
SYMBOLIC_BASE_URL = f"http://{SYMBOLIC_HOST}:{SYMBOLIC_PORT}"


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq3": load_yaml_config("rq3.yaml"),
    }


async def run_experiment_for_model(config: dict, model_name: str, examples, splits) -> dict:
    model_cfg = config["models"][model_name]
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    probe = config["rq3"]["symbolic_probe"]

    model, tokenizer = load_model(model_cfg)
    try:
        model_splits = splits.for_checkpoint(model_name)
        all_eval_indices = getattr(splits, config["rq3"]["eval_split"])
        n_requested = config["rq3"]["n_eval_examples"]
        eval_indices = all_eval_indices[:n_requested]
        if n_requested < len(all_eval_indices):
            print(
                f"subsampling {config['rq3']['eval_split']} split: using {len(eval_indices)} of "
                f"{len(all_eval_indices)} available examples for CPU tractability"
            )

        gate = GatePolicy()
        cal_entropies = [
            mean_calibration_entropy(model, tokenizer, examples[i].question, decoding_cfg, model_cfg)
            for i in model_splits.calibration
        ]
        gate.calibrate(
            calibration_entropies=cal_entropies,
            calibration_indices=model_splits.calibration,
            splits=model_splits,
            quantile=quantile,
            source=model_name,
        )

        per_example = []
        for index in eval_indices:
            example = examples[index]

            if index == eval_indices[0]:
                generate_with_latency(model, tokenizer, example.question, decoding_cfg)

            ungated = generate_with_latency(model, tokenizer, example.question, decoding_cfg)

            monitor = TokenEntropyMonitor(vocab_size=tokenizer.vocab_size)
            gated = generate_with_latency(model, tokenizer, example.question, decoding_cfg, logits_processor=[monitor])

            assert gated["text"] == ungated["text"], (
                "annotate-only merge-back must not change generation output — gated and "
                "ungated text diverged, which would mean routing is silently altering "
                "generation"
            )

            entropy = sum(monitor.entropies) / len(monitor.entropies)
            annotation = await route_and_annotate(
                gate,
                entropy,
                symbolic_base_url=SYMBOLIC_BASE_URL,
                subject_label=probe["subject_label"],
                predicate_pid=probe["predicate_pid"],
                object_label=probe["object_label"],
            )

            verdict = lexical_containment_verdict(
                ungated["text"], example.right_answer, (), (example.hallucinated_answer,)
            )

            per_example.append(
                {
                    "index": index,
                    "routed": annotation.routed,
                    "entropy": entropy,
                    "factuality": verdict.label,
                    "ungated_ttft_ms": ungated["ttft_ms"],
                    "ungated_total_ms": ungated["total_generation_ms"],
                    "gated_ttft_ms": gated["ttft_ms"],
                    "gated_total_ms": gated["total_generation_ms"],
                    "gate_eval_latency_ms": annotation.gate_eval_latency_ms,
                    "symbolic_latency_ms": annotation.symbolic_latency_ms,
                    "total_pipeline_latency_ms": gated["total_generation_ms"] + annotation.total_latency_ms,
                }
            )

        n = len(per_example)
        routed = [r for r in per_example if r["routed"]]

        n_abstained = 0
        verdicts = [FactualityVerdict(label=r["factuality"]) for r in per_example]
        factuality_report = summarize_factuality(verdicts, n_abstained=n_abstained)

        return {
            "research_question": "RQ3",
            "dataset": "halueval",
            "model_name": model_name,
            "hf_repo": model_cfg["hf_repo"],
            "revision": model_cfg["revision"],
            "eval_split": config["rq3"]["eval_split"],
            "n_eval_examples": n,
            "n_available_in_split": len(all_eval_indices),
            "quantile": quantile,
            "decoding": decoding_cfg,
            "routing_rate": len(routed) / n,
            "task_accuracy": factuality_report["task_accuracy"],
            "hallucination_rate": factuality_report["hallucination_rate"],
            "abstention_rate": factuality_report["abstention_rate"],
            "factuality_report": factuality_report,
            "factuality_metric": PLACEHOLDER_FACTUALITY_METRIC_LABEL,
            "mean_ungated_total_ms": sum(r["ungated_total_ms"] for r in per_example) / n,
            "mean_gated_total_ms": sum(r["gated_total_ms"] for r in per_example) / n,
            "mean_gate_eval_latency_ms": sum(r["gate_eval_latency_ms"] for r in per_example) / n,
            "mean_symbolic_latency_ms_when_routed": (
                sum(r["symbolic_latency_ms"] for r in routed) / len(routed) if routed else None
            ),
            "mean_total_pipeline_latency_ms": sum(r["total_pipeline_latency_ms"] for r in per_example) / n,
            "per_example": per_example,
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def run_all(config: dict) -> list[dict]:
    examples, splits = load_halueval_examples_and_splits()
    results = []
    for model_name in config["rq3"]["models"]:
        result = await run_experiment_for_model(config, model_name, examples, splits)
        write_results(
            f"rq3_accuracy_latency_halueval_{model_name}.json",
            result,
            print_exclude_keys=frozenset({"per_example"}),
        )
        results.append(result)
    return results


async def main() -> list[dict]:
    config = load_config()

    server_config = uvicorn.Config(symbolic_app, host=SYMBOLIC_HOST, port=SYMBOLIC_PORT, log_level="warning")
    server = uvicorn.Server(server_config)
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    try:
        results = await run_all(config)
    finally:
        server.should_exit = True
        await server_task

    return results


if __name__ == "__main__":
    asyncio.run(main())
```

- [x] **Step 4: Delete the superseded TruthfulQA-scoped harness and test**

```bash
rm experiments/rq3_accuracy_latency_truthful_qa.py experiments/tests/test_rq3_accuracy_latency_truthful_qa.py
```

- [x] **Step 5: Run tests to verify they pass**

Run: `cd experiments && .venv/bin/python -m pytest tests/test_rq3_accuracy_latency_halueval.py -v`
Expected: PASS (3 tests)

- [x] **Step 6: Run the full experiments suite**

Run: `cd experiments && .venv/bin/python -m pytest tests/ -q`
Expected: no collection errors, all non-GPU tests pass, same pass count as before this plan started (module renames are 1:1)

- [x] **Step 7: Commit**

```bash
git add experiments/rq3_accuracy_latency_halueval.py experiments/tests/test_rq3_accuracy_latency_halueval.py
git rm experiments/rq3_accuracy_latency_truthful_qa.py experiments/tests/test_rq3_accuracy_latency_truthful_qa.py
git commit -m "Re-point RQ3 harness to HaluEval, loop over all five checkpoints"
```

---

## Task 8: Full verification pass

**Files:** none (verification only)

- [x] **Step 1: Run the full experiments suite once more from a clean state**

Run: `cd experiments && uv sync -q && .venv/bin/python -m pytest tests/ -q`
Expected: all pass (or the same pre-existing skips as before this plan — e.g. the bitsandbytes-not-on-macOS skip), zero failures, zero collection errors

- [x] **Step 2: Confirm no stray references to the deleted files remain**

Run: `grep -rln "transfer_threshold_truthful_qa\|adaptive_threshold_truthful_qa\|rq3_accuracy_latency_truthful_qa" --include="*.py" --include="*.yaml" --include="*.md" . | grep -v .venv`
Expected: only CLAUDE.md's historical "Current status" narrative (if it mentions the old filenames) — no live code or config references. If CLAUDE.md does mention them, that's a docs-update note for the user, not something this plan's tests should touch.

- [x] **Step 3: Confirm the five-checkpoint quantization guard still passes with the new configs in place**

Run: `cd experiments && .venv/bin/python -c "from _common import load_model_registry; load_model_registry(); print('ok')"`
Expected: prints `ok` (raises loudly if this ever regresses, per the existing `assert_pinned_gpu_quantization` guard)

- [x] **Step 4: Report status to the user**

No commit for this task — it's verification only. Summarize: which of RQ1/RQ2/RQ3 are wired to HaluEval and five checkpoints, test results, and that the actual GPU run (real llama3/mistral/qwen3 weights) has NOT happened yet — this plan only wires the harnesses; running them against real 8B/7B/4B/1.7B checkpoints is separate GPU-environment work (Colab, per prior session history) and each run should be reported with real output or reported as failed/incomplete, never estimated.
