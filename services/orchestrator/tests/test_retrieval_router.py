import pytest

from sense_orchestrator.gate import GatePolicy, UncalibratedGateError
from sense_orchestrator.router import route_to_retrieval


@pytest.fixture
def calibrated_gate():
    gate = GatePolicy()
    gate.set_threshold(0.5, source="test")
    return gate


def test_low_entropy_does_not_retrieve(calibrated_gate):
    def fail_if_called(query):
        raise AssertionError("retrieval should not run when the gate doesn't fire")

    result = route_to_retrieval(calibrated_gate, 0.1, "q", fail_if_called)

    assert result.routed is False
    assert result.passages is None
    assert result.retrieval_latency_ms is None


def test_high_entropy_retrieves_for_the_query(calibrated_gate):
    seen = []

    def retrieve_fn(query):
        seen.append(query)
        return ["passage a", "passage b"]

    result = route_to_retrieval(calibrated_gate, 0.9, "who?", retrieve_fn)

    assert result.routed is True
    assert result.passages == ["passage a", "passage b"]
    assert seen == ["who?"]
    assert result.retrieval_latency_ms >= 0
    assert result.total_latency_ms >= result.gate_eval_latency_ms


def test_decision_matches_the_symbolic_router_gate(calibrated_gate):
    """Same gate, same entropy -> same routed decision; only the second stage differs."""
    for entropy in (0.1, 0.5, 0.9):
        result = route_to_retrieval(calibrated_gate, entropy, "q", lambda q: [])
        assert result.routed == calibrated_gate.decide(entropy)


def test_uncalibrated_gate_raises_rather_than_defaulting():
    with pytest.raises(UncalibratedGateError):
        route_to_retrieval(GatePolicy(), 0.9, "q", lambda q: [])
