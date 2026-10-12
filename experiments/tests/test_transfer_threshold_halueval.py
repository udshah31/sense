"""Integration test for the RQ1 transfer harness, on a small subset for speed —
transfer_threshold_halueval.py itself runs the full committed per-checkpoint
splits. Uses tiny CPU-testable models standing in for real checkpoints (CLAUDE.md:
every component must be testable on CPU with a tiny model) — the real five-
checkpoint run only happens on the GPU environment.
"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import calibration_entropies
from sense_eval.nli_judge import load_nli_model
from sense_data.splits import generate_splits
from sense_data.truthful_qa import load_truthful_qa
from sense_orchestrator.gate import GatePolicy
from transfer_threshold_halueval import run_pair

SOURCE_MODEL = "sshleifer/tiny-gpt2"
TARGET_MODEL = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
SOURCE_MODEL_REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
TARGET_MODEL_REVISION = "f417fcb49b46298ae7c01308ff33bfeadc104bd3"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 5}
SOURCE_MODEL_CFG = {"hf_repo": SOURCE_MODEL, "revision": SOURCE_MODEL_REVISION}
TARGET_MODEL_CFG = {"hf_repo": TARGET_MODEL, "revision": TARGET_MODEL_REVISION}

# The NLI judge now runs inside run_pair: routing quality is measured as detection
# performance against the ungated model's own correctness, which needs a verdict per
# example (proposal-v5 review issue C1). Same pinned judge the real runs use — small
# enough to load in a test, and loading the real one keeps the test on the real path.
NLI_CFG = {
    "hf_repo": "cliang1453/deberta-v3-xsmall-mnli",
    "revision": "d1ca70f9ece4d8afd33015893a69df9a6e45a672",
    "short_answer_entailment_threshold": 0.7,
}


@pytest.fixture(scope="module")
def nli():
    return load_nli_model(NLI_CFG["hf_repo"], NLI_CFG["revision"])



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


def test_run_pair_end_to_end_on_tiny_models_via_halueval_shaped_splits(nli):
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

    nli_model, nli_tokenizer = nli
    result = run_pair(
        models_registry, examples, splits, DECODING_CFG, quantile=0.9,
        eval_split="development", source_name="source", target_name="target",
        nli_model=nli_model, nli_tokenizer=nli_tokenizer, nli_cfg=NLI_CFG,
        bootstrap_cfg={"n_resamples": 50, "confidence": 0.95, "seed": 42},
    )

    assert result["research_question"] == "RQ1"
    assert result["dataset"] == "halueval"
    assert result["source_model"]["name"] == "source"
    assert result["target_model"]["name"] == "target"
    assert result["n_eval_examples"] == 10
    for scale in ("normalized", "raw"):
        arm = result["arms"][scale]
        assert arm["entropy_scale"] == scale
        # Concordance diagnostic is still reported, just no longer the headline.
        assert 0.0 <= arm["transfer_agreement_rate"] <= 1.0
        # The transferred threshold must be the source's own, unmodified.
        assert arm["transferred_threshold"] == arm["source_native_threshold"]
        for gate_key in ("transferred_gate", "native_gate"):
            metrics = arm[gate_key]
            assert metrics["n_outcomes"] == 10
            assert metrics["n_scored"] + metrics["n_unknown_excluded"] == 10
            assert 0.0 <= metrics["routing_rate"] <= 1.0
            assert metrics["entropy_auroc"] is None or 0.0 <= metrics["entropy_auroc"] <= 1.0
        assert "delta_f1" in arm["routing_quality_delta"]

    # The raw arm must transfer a genuinely different number from the normalized one;
    # if these matched, the second arm would not be testing anything new.
    assert result["arms"]["raw"]["transferred_threshold"] != result["arms"]["normalized"]["transferred_threshold"]

    # The required factuality triple, prefixed, so write_results' always-together
    # check treats it as its own group.
    for key in ("ungated_task_accuracy", "ungated_hallucination_rate", "ungated_abstention_rate"):
        assert key in result
    assert result["ungated_abstention_rate"] == 0.0  # RQ1 acts on no routing decision


def test_run_pair_n_eval_examples_takes_a_fixed_prefix_of_the_split(nli):
    from sense_data.halueval import load_halueval
    from sense_data.splits import PerCheckpointSplitIndices

    examples = load_halueval()[:50]
    splits = PerCheckpointSplitIndices(
        development=list(range(40, 50)),
        test=list(range(30, 40)),
        calibration={"source": list(range(0, 10)), "target": list(range(10, 20))},
    )
    nli_model, nli_tokenizer = nli
    result = run_pair(
        {"source": SOURCE_MODEL_CFG, "target": TARGET_MODEL_CFG}, examples, splits, DECODING_CFG, quantile=0.9,
        eval_split="test", source_name="source", target_name="target",
        nli_model=nli_model, nli_tokenizer=nli_tokenizer, nli_cfg=NLI_CFG,
        bootstrap_cfg={"n_resamples": 20, "confidence": 0.95, "seed": 42},
        n_eval_examples=4,
    )

    assert result["n_eval_examples"] == 4
    assert result["eval_split"] == "test"
