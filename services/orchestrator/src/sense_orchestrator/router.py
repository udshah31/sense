"""Routes a generation step to the symbolic stream when the gate fires, and
annotates the result — merge-back policy is annotate-only for now (CLAUDE.md's
design decisions: run annotate-only first, add replace only once the symbolic
backend is stable). This module never modifies model output, only records what the
gate and symbolic backend said about it, plus per-stage latency.
"""

import time
from dataclasses import dataclass

from sense_orchestrator.gate import GatePolicy
from sense_orchestrator.symbolic_client import SymbolicBackendError, verify_triple


@dataclass(frozen=True)
class RoutingAnnotation:
    routed: bool
    entropy: float
    gate_eval_latency_ms: float
    symbolic_result: dict | None
    symbolic_latency_ms: float | None
    symbolic_error: str | None
    total_latency_ms: float


async def route_and_annotate(
    gate: GatePolicy,
    entropy: float,
    symbolic_base_url: str,
    subject_label: str,
    predicate_pid: str,
    object_label: str,
) -> RoutingAnnotation:
    """Evaluate the gate on `entropy`; if it fires, call the symbolic backend and
    attach its response as an annotation. Never raises on a symbolic-backend
    failure — that's recorded in `symbolic_error` so a flaky external dependency
    can't take down generation; it does propagate GatePolicy's own errors (e.g.
    UncalibratedGateError), since running uncalibrated is a hard failure by design.
    """
    total_start = time.perf_counter()

    gate_start = time.perf_counter()
    routed = gate.decide(entropy)
    gate_eval_latency_ms = (time.perf_counter() - gate_start) * 1000

    symbolic_result = None
    symbolic_latency_ms = None
    symbolic_error = None

    if routed:
        symbolic_start = time.perf_counter()
        try:
            symbolic_result = await verify_triple(symbolic_base_url, subject_label, predicate_pid, object_label)
        except SymbolicBackendError as e:
            symbolic_error = str(e)
        symbolic_latency_ms = (time.perf_counter() - symbolic_start) * 1000

    total_latency_ms = (time.perf_counter() - total_start) * 1000

    return RoutingAnnotation(
        routed=routed,
        entropy=entropy,
        gate_eval_latency_ms=gate_eval_latency_ms,
        symbolic_result=symbolic_result,
        symbolic_latency_ms=symbolic_latency_ms,
        symbolic_error=symbolic_error,
        total_latency_ms=total_latency_ms,
    )
