"""Integration test for the RQ2 adaptive-threshold harness, on a small subset for
speed — adaptive_threshold_truthful_qa.py itself runs the full committed splits.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import calibration_entropies
from adaptive_threshold_truthful_qa import routing_rate
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

    # The two thresholds are fit independently and needn't match unless the two
    # models happen to share an entropy distribution (they don't — different
    # families, different vocab sizes).
    assert adaptive_gate.calibration_source == "adaptive-target"
    assert fixed_gate.calibration_source == "transferred-from-source"

    eval_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, dev_subset, DECODING_CFG, TARGET_MODEL_CFG
    )
    fixed_rate = routing_rate(fixed_gate, eval_entropies)
    adaptive_rate = routing_rate(adaptive_gate, eval_entropies)

    assert 0.0 <= fixed_rate <= 1.0
    assert 0.0 <= adaptive_rate <= 1.0
    # Sanity: an adaptive gate calibrated on the target's own entropies uses the
    # target's own threshold, not the source's.
    assert adaptive_threshold != source_threshold


def test_routing_rate_helper_matches_manual_count():
    gate = GatePolicy()
    gate.set_threshold(0.5, source="test")
    entropies = [0.1, 0.6, 0.9, 0.2, 0.5]
    assert routing_rate(gate, entropies) == 3 / 5
