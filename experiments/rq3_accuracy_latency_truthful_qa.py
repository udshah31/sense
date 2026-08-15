"""RQ3 harness: what accuracy-latency trade-off does routing introduce?

Starts the real symbolic backend (services/symbolic) as a live local server, so the
routed condition's symbolic round-trip is genuine HTTP over loopback — the same
contract the orchestrator uses in docker-compose, not an in-process shortcut.

For each development-split example, generates twice under identical greedy decoding
(CLAUDE.md's decoding-config constraint): once "ungated" (no monitor, no routing)
and once "gated" (entropy monitor attached, gate evaluated, symbolic round-trip on
routed examples). Since merge-back is annotate-only (CLAUDE.md's design decisions),
the two generations must produce byte-identical text — gating never touches
generation output. That equality is asserted, not just hoped for: it's the same
no-op guarantee services/neural's monitor tests check, extended through the whole
routed pipeline.

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
from pathlib import Path

import uvicorn
from sense_symbolic.app import app as symbolic_app

from _common import load_examples_and_splits, load_model, load_model_registry, load_yaml_config, mean_calibration_entropy, write_results
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


async def run_experiment(config: dict) -> dict:
    model_name = config["rq3"]["model"]
    model_cfg = config["models"][model_name]
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    probe = config["rq3"]["symbolic_probe"]

    model, tokenizer = load_model(model_cfg)
    examples, splits = load_examples_and_splits()

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
        for i in splits.calibration
    ]
    gate.calibrate(
        calibration_entropies=cal_entropies,
        calibration_indices=splits.calibration,
        splits=splits,
        quantile=quantile,
        source=model_name,
    )

    per_example = []
    for index in eval_indices:
        example = examples[index]

        # Warm-up discarded before measuring, per CLAUDE.md — first-call kernel
        # compilation would otherwise dominate the very first example's timing.
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
            ungated["text"], example.best_answer, example.correct_answers, example.incorrect_answers
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

    # Merge-back is annotate-only for now (CLAUDE.md's design decisions) — abstain
    # is a configurable-but-not-yet-built merge-back policy, so no example here is
    # ever abstained. n_abstained is passed explicitly rather than defaulted so
    # that becoming untrue (once abstain is implemented) can't silently slip past
    # this harness without updating it.
    n_abstained = 0
    verdicts = [FactualityVerdict(label=r["factuality"]) for r in per_example]
    factuality_report = summarize_factuality(verdicts, n_abstained=n_abstained)

    result = {
        "research_question": "RQ3",
        "dataset": "truthful_qa",
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
    return result


async def main() -> dict:
    config = load_config()

    server_config = uvicorn.Config(symbolic_app, host=SYMBOLIC_HOST, port=SYMBOLIC_PORT, log_level="warning")
    server = uvicorn.Server(server_config)
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    try:
        result = await run_experiment(config)
    finally:
        server.should_exit = True
        await server_task

    write_results(
        f"rq3_accuracy_latency_truthful_qa_{config['rq3']['model']}.json",
        result,
        print_exclude_keys=frozenset({"per_example"}),
    )
    return result


if __name__ == "__main__":
    asyncio.run(main())
