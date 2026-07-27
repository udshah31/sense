"""Integration test for the RQ1 transfer harness, on a small subset for speed —
transfer_threshold_truthful_qa.py itself runs the full committed splits.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from sense_data.splits import generate_splits
from sense_orchestrator.gate import GatePolicy
from transfer_threshold_truthful_qa import calibration_entropies
from sense_data.truthful_qa import load_truthful_qa

SOURCE_MODEL = "sshleifer/tiny-gpt2"
TARGET_MODEL = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 5}


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

    source_entropies = calibration_entropies(source_model, source_tokenizer, examples, cal_subset, DECODING_CFG)
    source_gate = GatePolicy()
    source_threshold = source_gate.calibrate(
        calibration_entropies=source_entropies,
        calibration_indices=cal_subset,
        splits=small_splits,
        quantile=0.9,
        source="source",
    )

    target_entropies = calibration_entropies(target_model, target_tokenizer, examples, cal_subset, DECODING_CFG)
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

    eval_entropies = calibration_entropies(target_model, target_tokenizer, examples, dev_subset, DECODING_CFG)

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

    entropies = calibration_entropies(source_model, source_tokenizer, examples, cal_subset, DECODING_CFG)
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
