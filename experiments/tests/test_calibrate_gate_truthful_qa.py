"""Integration test for the real calibration wiring, on a small subset for speed —
experiments/calibrate_gate_truthful_qa.py itself runs the full committed calibration
split.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import mean_calibration_entropy
from sense_data.splits import assert_no_test_leakage, generate_splits
from sense_data.truthful_qa import load_truthful_qa
from sense_orchestrator.gate import GatePolicy

TINY_MODEL = "sshleifer/tiny-gpt2"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 5}


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    return model, tokenizer


def test_gate_calibrates_on_a_real_truthful_qa_subset(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    examples = load_truthful_qa()
    splits = generate_splits(n=len(examples), calibration_frac=0.01, development_frac=0.01, test_frac=0.98, seed=0)
    subset = splits.calibration[:5]

    entropies = [
        mean_calibration_entropy(model, tokenizer, examples[i].question, DECODING_CFG) for i in subset
    ]

    gate = GatePolicy()
    threshold = gate.calibrate(
        calibration_entropies=entropies,
        calibration_indices=subset,
        splits=splits,
        quantile=0.9,
        source="test",
    )

    assert gate.is_calibrated
    assert 0.0 <= threshold <= 1.0
    assert gate.decide(1.0) is True  # max possible entropy always routes


def test_calibrating_on_real_test_split_indices_raises(tiny_model_and_tokenizer):
    model, tokenizer = tiny_model_and_tokenizer
    examples = load_truthful_qa()
    splits = generate_splits(n=len(examples), calibration_frac=0.01, development_frac=0.01, test_frac=0.98, seed=0)
    leaking_subset = splits.test[:5]

    entropies = [
        mean_calibration_entropy(model, tokenizer, examples[i].question, DECODING_CFG)
        for i in leaking_subset
    ]

    gate = GatePolicy()
    with pytest.raises(Exception):
        gate.calibrate(
            calibration_entropies=entropies,
            calibration_indices=leaking_subset,
            splits=splits,
            quantile=0.9,
        )
    assert not gate.is_calibrated
