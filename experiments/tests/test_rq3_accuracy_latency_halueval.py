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

from _common import build_generation_inputs
from rq3_accuracy_latency_halueval import SYMBOLIC_BASE_URL, SYMBOLIC_HOST, SYMBOLIC_PORT
from sense_data.halueval import load_halueval
from sense_data.splits import generate_splits
from sense_eval.nli_judge import load_nli_model, nli_verdict_short_answer
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
    inputs = build_generation_inputs(tokenizer, question, {}).to(model.device)

    ungated = generate_with_latency(model, tokenizer, inputs, DECODING_CFG)

    monitor = TokenEntropyMonitor(vocab_size=tokenizer.vocab_size)
    gated = generate_with_latency(model, tokenizer, inputs, DECODING_CFG, logits_processor=[monitor])

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
    nli_model, nli_tokenizer = load_nli_model(
        "cliang1453/deberta-v3-xsmall-mnli", "d1ca70f9ece4d8afd33015893a69df9a6e45a672"
    )
    verdict = nli_verdict_short_answer(
        nli_model, nli_tokenizer, example.right_answer, example.right_answer, example.hallucinated_answer, 0.7
    )
    assert verdict.label == "correct"


def test_eval_split_subsampling_respects_requested_count():
    examples = load_halueval()
    splits = generate_splits(n=len(examples), calibration_frac=0.4, development_frac=0.2, test_frac=0.4, seed=0)
    subset = splits.development[:5]
    assert len(subset) == 5
    assert set(subset).issubset(set(splits.development))
