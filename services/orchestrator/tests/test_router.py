"""Router tests use the real sense_symbolic FastAPI app in-process via ASGI
transport — a real symbolic response, no live network hop (except the one test
that deliberately exercises the sparql fallback), and no mocking of the router's
own logic. sense_symbolic is a test-only dependency (see pyproject.toml); at
runtime the orchestrator only ever talks to it over HTTP.

`symbolic_client` wires the default app — z3 backend (2026-08-13 scope
reconciliation) — so most of this file exercises the same local, deterministic,
no-network backend router.py talks to by default. The one test still named
"real_symbolic_backend" is the pre-existing sparql/Wikidata integration test,
now pointed at create_app("sparql") explicitly rather than relying on it being
the default.
"""

import httpx
import pytest
from sense_symbolic.app import app as symbolic_app
from sense_symbolic.app import create_app as create_symbolic_app

from sense_orchestrator.gate import GatePolicy, UncalibratedGateError
from sense_orchestrator.router import route_and_annotate

EINSTEIN = "Albert Einstein"
PHYSICIST = "physicist"
OCCUPATION_PID = "P106"


@pytest.fixture
def calibrated_gate():
    gate = GatePolicy()
    gate.set_threshold(0.5, source="test")
    return gate


@pytest.fixture
def symbolic_client():
    """The default (z3) backend, in-process."""
    transport = httpx.ASGITransport(app=symbolic_app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


@pytest.fixture
def sparql_symbolic_client():
    transport = httpx.ASGITransport(app=create_symbolic_app("sparql"))
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


async def test_low_entropy_does_not_route_or_call_symbolic(calibrated_gate, monkeypatch):
    async def fail_if_called(*args, **kwargs):
        raise AssertionError("symbolic backend should not be called when the gate doesn't fire")

    monkeypatch.setattr("sense_orchestrator.router.verify_triple", fail_if_called)

    annotation = await route_and_annotate(
        calibrated_gate, entropy=0.1, symbolic_base_url="unused",
        subject_label=EINSTEIN, predicate_pid=OCCUPATION_PID, object_label=PHYSICIST,
    )

    assert annotation.routed is False
    assert annotation.symbolic_result is None
    assert annotation.symbolic_latency_ms is None
    assert annotation.gate_eval_latency_ms >= 0


async def test_high_entropy_routes_and_calls_real_symbolic_backend(calibrated_gate, symbolic_client, monkeypatch):
    """Default (z3) backend — local and deterministic, so this asserts the actual
    verification outcome, not just that a round-trip happened."""

    async def use_test_client(base_url, subject_label, predicate_pid, object_label, client=None):
        from sense_orchestrator.symbolic_client import verify_triple as real_verify_triple

        return await real_verify_triple("", subject_label, predicate_pid, object_label, client=symbolic_client)

    monkeypatch.setattr("sense_orchestrator.router.verify_triple", use_test_client)

    annotation = await route_and_annotate(
        calibrated_gate, entropy=0.9, symbolic_base_url="unused",
        subject_label=EINSTEIN, predicate_pid=OCCUPATION_PID, object_label=PHYSICIST,
    )

    assert annotation.routed is True
    assert annotation.symbolic_result is not None
    assert annotation.symbolic_result["subject_qid"] == "albert einstein"
    assert annotation.symbolic_result["verified"] is True
    assert annotation.symbolic_latency_ms is not None
    assert annotation.symbolic_latency_ms >= 0
    assert annotation.total_latency_ms >= annotation.gate_eval_latency_ms


async def test_high_entropy_routes_and_calls_sparql_fallback_backend(
    calibrated_gate, sparql_symbolic_client, monkeypatch
):
    """The sparql backend is still fully wired behind the same router — it's a
    fallback now, not deleted (CLAUDE.md's design decisions)."""

    async def use_test_client(base_url, subject_label, predicate_pid, object_label, client=None):
        from sense_orchestrator.symbolic_client import verify_triple as real_verify_triple

        return await real_verify_triple("", subject_label, predicate_pid, object_label, client=sparql_symbolic_client)

    monkeypatch.setattr("sense_orchestrator.router.verify_triple", use_test_client)

    annotation = await route_and_annotate(
        calibrated_gate, entropy=0.9, symbolic_base_url="unused",
        subject_label=EINSTEIN, predicate_pid=OCCUPATION_PID, object_label=PHYSICIST,
    )

    assert annotation.routed is True
    assert annotation.symbolic_result is not None
    assert annotation.symbolic_result["subject_qid"] == "Q937"
    assert annotation.symbolic_latency_ms is not None
    assert annotation.symbolic_latency_ms >= 0
    assert annotation.total_latency_ms >= annotation.gate_eval_latency_ms


async def test_uncalibrated_gate_raises_rather_than_defaulting():
    gate = GatePolicy()
    with pytest.raises(UncalibratedGateError):
        await route_and_annotate(
            gate, entropy=0.9, symbolic_base_url="unused",
            subject_label=EINSTEIN, predicate_pid=OCCUPATION_PID, object_label=PHYSICIST,
        )


async def test_symbolic_backend_failure_is_annotated_not_raised(calibrated_gate, monkeypatch):
    from sense_orchestrator.symbolic_client import SymbolicBackendError

    async def always_fails(*args, **kwargs):
        raise SymbolicBackendError("simulated network failure")

    monkeypatch.setattr("sense_orchestrator.router.verify_triple", always_fails)

    annotation = await route_and_annotate(
        calibrated_gate, entropy=0.9, symbolic_base_url="unused",
        subject_label=EINSTEIN, predicate_pid=OCCUPATION_PID, object_label=PHYSICIST,
    )

    assert annotation.routed is True
    assert annotation.symbolic_result is None
    assert annotation.symbolic_error == "simulated network failure"
