"""Retrieval-gated harness on the tiny CPU model: the per-example routing logic and
the aggregation, with a stub judge and retriever (the judge and the FAISS index have
their own tests; the real checkpoints only run on the GPU host)."""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval_gated_halueval import answer_example, context_token_budget, summarize
from sense_data.halueval import load_halueval
from sense_orchestrator.gate import GatePolicy

TINY_MODEL = "sshleifer/tiny-gpt2"
DECODING_CFG = {"do_sample": False, "max_new_tokens": 5}


@pytest.fixture(scope="module")
def tiny():
    return AutoModelForCausalLM.from_pretrained(TINY_MODEL), AutoTokenizer.from_pretrained(TINY_MODEL)


def _gate(threshold):
    gate = GatePolicy()
    gate.set_threshold(threshold, source="test")
    return gate


def _answer(tiny, threshold, retrieve_fn):
    model, tokenizer = tiny
    example = load_halueval()[0]
    budget = context_token_budget(model, tokenizer, DECODING_CFG["max_new_tokens"])
    record = answer_example(
        model, tokenizer, {}, DECODING_CFG, _gate(threshold), example, retrieve_fn,
        lambda text, ex: "unknown", budget,
    )
    return example, record


def test_unrouted_example_keeps_first_answer_and_never_retrieves(tiny):
    def fail_if_called(query):
        raise AssertionError("retrieval ran although the gate did not fire")

    _, record = _answer(tiny, 1e9, fail_if_called)

    assert record["routed"] is False
    assert record["final_text"] == record["ungated_text"]
    assert record["retrieval_latency_ms"] is None
    assert record["regeneration_total_ms"] is None
    assert record["total_pipeline_latency_ms"] >= record["first_pass_total_ms"]


def test_routed_example_retrieves_for_its_question_and_regenerates(tiny):
    seen = []

    def retrieve_fn(query):
        seen.append(query)
        return ["Paris is the capital of France."]

    example, record = _answer(tiny, -1.0, retrieve_fn)

    assert record["routed"] is True
    assert seen == [example.question]
    assert record["n_passages_used"] == 1
    assert record["regeneration_total_ms"] > 0
    assert record["total_pipeline_latency_ms"] >= record["first_pass_total_ms"] + record["regeneration_total_ms"]


def _record(routed, before, after):
    return {
        "routed": routed, "entropy": 0.9 if routed else 0.1, "raw_entropy": 1.0,
        "ungated_factuality": before, "factuality": after,
        "first_pass_total_ms": 10.0, "gate_eval_latency_ms": 0.1,
        "retrieval_latency_ms": 5.0 if routed else None,
        "regeneration_total_ms": 8.0 if routed else None,
        "total_pipeline_latency_ms": 23.1 if routed else 10.1,
        "prompt_truncated": False if routed else None,
    }


def test_summarize_reports_final_and_ungated_triples_and_label_transitions():
    records = [
        _record(True, "incorrect", "correct"),
        _record(True, "correct", "incorrect"),
        _record(False, "correct", "correct"),
        _record(False, "incorrect", "incorrect"),
    ]
    out = summarize(records, "m", {"n_resamples": 20, "confidence": 0.9, "seed": 0})

    assert out["routing_rate"] == 0.5
    assert out["task_accuracy"] == 0.5 and out["ungated_task_accuracy"] == 0.5
    assert out["abstention_rate"] == 0 and out["ungated_abstention_rate"] == 0
    assert out["routed_label_transitions"] == {"incorrect->correct": 1, "correct->incorrect": 1}
    assert out["mean_retrieval_latency_ms_when_routed"] == 5.0
    assert out["mean_regeneration_total_ms_when_routed"] == 8.0
    assert out["mean_total_pipeline_latency_ms"] == pytest.approx((23.1 * 2 + 10.1 * 2) / 4)
