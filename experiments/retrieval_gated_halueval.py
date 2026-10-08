"""Retrieval-gated condition: the same entropy gate as RQ3, routing to retrieval.

Per example, under identical greedy decoding (CLAUDE.md constraint #3): generate once
with the entropy monitor attached; if the gate fires, retrieve passages for the
question and regenerate with them in the prompt, and that regeneration is the final
answer; if it doesn't fire, the first answer stands. This is what FLARE/DRAGIN do
(retrieve on uncertainty, then regenerate), so it is the fair retrieval counterpart
to SENSE's solver stage — with one deliberate asymmetry the results must say out
loud: SENSE's merge-back is annotate-only (output never changes), whereas retrieval
changes the output. Accuracy here is therefore the retrieval condition's *effect*,
and the SENSE condition's accuracy is the ungated accuracy by construction.

The gate, its calibration split and quantile, the first-pass prompt, the judge and
its threshold are all the same code and config as RQ3, so a difference between the
two conditions is attributable to the second stage.

Reports task_accuracy/hallucination_rate/abstention_rate together for the final
answers (write_results enforces this) and the same triple for the first-pass answers
under `ungated_`. abstention_rate is 0: nothing here abstains.
"""

import asyncio
import gc
from collections import Counter

import torch

from _common import (
    REPO_ROOT,
    bootstrap_config,
    build_generation_inputs,
    derived_seed,
    load_halueval_examples_and_splits,
    load_model,
    load_model_registry,
    load_yaml_config,
    mean_calibration_entropy,
    seed_everything,
    write_results,
)
from rag_baseline_halueval import build_prompt_within_budget
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, load_nli_model, nli_verdict_short_answer
from sense_eval.routing_quality import RoutingOutcome, detection_metrics_with_ci
from sense_neural.entropy import TokenEntropyMonitor
from sense_neural.latency import generate_with_latency
from sense_orchestrator.gate import GatePolicy
from sense_orchestrator.router import route_to_retrieval
from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "retrieval_gated": load_yaml_config("retrieval_gated.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }


def context_token_budget(model, tokenizer, max_new_tokens: int) -> int:
    """Prompt-token budget. tokenizer.model_max_length can be a placeholder (1e30) for
    models without an explicit value, so fall back to the model config's real limit —
    never a hard-coded family (CLAUDE.md)."""
    limit = tokenizer.model_max_length
    if limit is None or limit > 1_000_000:
        limit = getattr(model.config, "max_position_embeddings", None) or model.config.n_positions
    return limit - max_new_tokens


def answer_example(model, tokenizer, model_cfg, decoding_cfg, gate, example, retrieve_fn, judge, token_budget) -> dict:
    inputs = build_generation_inputs(tokenizer, example.question, model_cfg).to(model.device)
    monitor = TokenEntropyMonitor(vocab_size=tokenizer.vocab_size)
    first = generate_with_latency(model, tokenizer, inputs, decoding_cfg, logits_processor=[monitor])

    entropy = sum(monitor.entropies) / len(monitor.entropies)
    raw_entropy = sum(monitor.raw_entropies) / len(monitor.raw_entropies)
    routing = route_to_retrieval(gate, entropy, example.question, retrieve_fn)

    final_text = first["text"]
    regeneration_ms = None
    n_passages_used = None
    prompt_truncated = None
    if routing.routed:
        prompt, n_passages_used, prompt_truncated = build_prompt_within_budget(
            routing.passages, example.question, tokenizer, token_budget
        )
        second_inputs = build_generation_inputs(tokenizer, prompt, model_cfg).to(model.device)
        second = generate_with_latency(model, tokenizer, second_inputs, decoding_cfg)
        final_text = second["text"]
        regeneration_ms = second["total_generation_ms"]

    return {
        "routed": routing.routed,
        "entropy": entropy,
        "raw_entropy": raw_entropy,
        "ungated_text": first["text"],
        "final_text": final_text,
        "ungated_factuality": judge(first["text"], example),
        "factuality": judge(final_text, example),
        "first_pass_total_ms": first["total_generation_ms"],
        "gate_eval_latency_ms": routing.gate_eval_latency_ms,
        "retrieval_latency_ms": routing.retrieval_latency_ms,
        "regeneration_total_ms": regeneration_ms,
        "total_pipeline_latency_ms": first["total_generation_ms"]
        + routing.total_latency_ms
        + (regeneration_ms or 0.0),
        "n_passages_used": n_passages_used,
        "prompt_truncated": prompt_truncated,
    }


def summarize(per_example: list[dict], model_name: str, bootstrap_cfg: dict) -> dict:
    """Pure aggregation over `answer_example` records."""
    n = len(per_example)
    routed = [r for r in per_example if r["routed"]]
    final_report = summarize_factuality(
        [FactualityVerdict(label=r["factuality"]) for r in per_example], n_abstained=0
    )
    ungated_report = summarize_factuality(
        [FactualityVerdict(label=r["ungated_factuality"]) for r in per_example], n_abstained=0
    )
    return {
        "routing_rate": len(routed) / n,
        "routing_detection": detection_metrics_with_ci(
            [
                RoutingOutcome(entropy=r["entropy"], routed=r["routed"], factuality=r["ungated_factuality"])
                for r in per_example
            ],
            n_resamples=bootstrap_cfg["n_resamples"],
            confidence=bootstrap_cfg["confidence"],
            seed=derived_seed(bootstrap_cfg["seed"], model_name, "retrieval_gated", "detection"),
        ),
        "task_accuracy": final_report["task_accuracy"],
        "hallucination_rate": final_report["hallucination_rate"],
        "abstention_rate": final_report["abstention_rate"],
        "factuality_report": final_report,
        "ungated_task_accuracy": ungated_report["task_accuracy"],
        "ungated_hallucination_rate": ungated_report["hallucination_rate"],
        "ungated_abstention_rate": ungated_report["abstention_rate"],
        "ungated_factuality_report": ungated_report,
        # What retrieval did to routed answers, as "<first-pass label>-><final label>".
        # Unrouted answers are unchanged by construction, so only routed ones appear.
        "routed_label_transitions": dict(
            Counter(f"{r['ungated_factuality']}->{r['factuality']}" for r in routed)
        ),
        "mean_first_pass_total_ms": sum(r["first_pass_total_ms"] for r in per_example) / n,
        "mean_gate_eval_latency_ms": sum(r["gate_eval_latency_ms"] for r in per_example) / n,
        "mean_retrieval_latency_ms_when_routed": (
            sum(r["retrieval_latency_ms"] for r in routed) / len(routed) if routed else None
        ),
        "mean_regeneration_total_ms_when_routed": (
            sum(r["regeneration_total_ms"] for r in routed) / len(routed) if routed else None
        ),
        "mean_total_pipeline_latency_ms": sum(r["total_pipeline_latency_ms"] for r in per_example) / n,
        "n_routed_prompts_truncated": sum(1 for r in routed if r["prompt_truncated"]),
    }


def run_experiment_for_model(config, model_name, examples, splits, index, nli_model, nli_tokenizer, bootstrap_cfg) -> dict:
    model_cfg = config["models"][model_name]
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    rg_cfg = config["retrieval_gated"]
    threshold = config["nli_judge"]["short_answer_entailment_threshold"]

    def judge(text, example):
        return nli_verdict_short_answer(
            nli_model, nli_tokenizer, text, example.right_answer, example.hallucinated_answer, threshold,
            question=example.question,
        ).label

    model, tokenizer = load_model(model_cfg)
    try:
        model_splits = splits.for_checkpoint(model_name)
        all_eval_indices = getattr(splits, rg_cfg["eval_split"])
        eval_indices = all_eval_indices[: rg_cfg["n_eval_examples"]]

        gate = GatePolicy()
        gate.calibrate(
            calibration_entropies=[
                mean_calibration_entropy(model, tokenizer, examples[i].question, decoding_cfg, model_cfg)
                for i in model_splits.calibration
            ],
            calibration_indices=model_splits.calibration,
            splits=model_splits,
            quantile=quantile,
            source=model_name,
        )

        token_budget = context_token_budget(model, tokenizer, decoding_cfg["max_new_tokens"])
        retrieve_fn = lambda query: retrieve(index, query, k=rg_cfg["top_k"])  # noqa: E731

        # Warm-up, discarded (CLAUDE.md latency rule): first-call kernel compilation.
        warm = examples[eval_indices[0]]
        generate_with_latency(
            model, tokenizer, build_generation_inputs(tokenizer, warm.question, model_cfg).to(model.device), decoding_cfg
        )

        per_example = []
        for i in eval_indices:
            record = answer_example(
                model, tokenizer, model_cfg, decoding_cfg, gate, examples[i], retrieve_fn, judge, token_budget
            )
            per_example.append({"index": i, **record})

        return {
            "research_question": "retrieval-gated condition",
            "dataset": "halueval",
            "model_name": model_name,
            "hf_repo": model_cfg["hf_repo"],
            "revision": model_cfg["revision"],
            "eval_split": rg_cfg["eval_split"],
            "n_eval_examples": len(per_example),
            "n_available_in_split": len(all_eval_indices),
            "quantile": quantile,
            "top_k": rg_cfg["top_k"],
            "corpus_path": rg_cfg["corpus_path"],
            "decoding": decoding_cfg,
            "bootstrap": bootstrap_cfg,
            **summarize(per_example, model_name, bootstrap_cfg),
            "factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
                hf_repo=config["nli_judge"]["hf_repo"], revision=config["nli_judge"]["revision"]
            ),
            "per_example": per_example,
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> list[dict]:
    seed_everything()
    config = load_config()
    examples, splits = load_halueval_examples_and_splits()
    nli_model, nli_tokenizer = load_nli_model(config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"])
    index = PassageIndex.from_file(REPO_ROOT / config["retrieval_gated"]["corpus_path"])
    bootstrap_cfg = bootstrap_config()

    results = []
    for model_name in config["retrieval_gated"]["models"]:
        result = run_experiment_for_model(
            config, model_name, examples, splits, index, nli_model, nli_tokenizer, bootstrap_cfg
        )
        write_results(
            f"retrieval_gated_halueval_{model_name}.json",
            result,
            print_exclude_keys=frozenset({"per_example"}),
        )
        results.append(result)
    return results


if __name__ == "__main__":
    main()
