"""RQ2 harness: does a self-adaptive threshold — recalibrated on the target model's
own calibration data — preserve routing quality on held-out data where a fixed
(transferred) threshold does not?

Runs once per (source_model, target_model) pair in configs/rq2.yaml's
transfer_pairs list, writing one result file per pair. Each checkpoint calibrates
on its own disjoint calibration subset (data/splits/halueval.json's per-checkpoint
shape), same as RQ1.

**What "routing quality" means here, revised 2026-10-04** (proposal-v5 review issue
C1, docs/research/sense-proposal-v5-review.md).

It previously meant *calibration fidelity*: a threshold calibrated at quantile q is
designed to route roughly (1 - q) of examples, so the gap between a gate's observed
held-out routing rate and that (1 - q) target measured whether the threshold still
behaved as designed on unseen data. The adaptive gate has a reason to track it (it was
fit on this model's own distribution); the fixed gate does not (it was fit on a
different model's).

That is a genuine property of a threshold, and it is still reported below. But it is
not routing *quality*: a gate that fires on exactly (1 - q) of examples chosen at
random scores perfectly on it. It says nothing about whether the examples routed were
the ones worth routing.

Routing quality is therefore now measured primarily as detection performance against
the ungated model's own correctness — does the gate fire on the examples the model
actually got wrong — alongside the threshold-free AUROC of the entropy signal. See
eval/src/sense_eval/routing_quality.py. Calibration fidelity is retained as a
secondary diagnostic, which is what it always was.

Both entropy scales get a full condition, for the same reason as RQ1: the normalized
scale (raw / ln(vocab_size)) has the cross-tokenizer difference divided out of it
before the gate ever sees it, so a fixed threshold transferred on that scale is a
weaker test than one transferred on the raw scale. See the RQ1 harness docstring.

Models are loaded and explicitly freed around each pair so peak GPU memory never
holds more than the current pair's two checkpoints.
"""

import gc

import torch

from _common import (
    entropies_on_scale,
    example_signals,
    load_halueval_examples_and_splits,
    load_model,
    load_model_registry,
    load_yaml_config,
    seed_everything,
    write_results,
)
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, load_nli_model, nli_verdict_short_answer
from sense_eval.routing_quality import RoutingOutcome, routing_detection_metrics, routing_quality_delta
from sense_orchestrator.gate import GatePolicy

ENTROPY_SCALES = ("normalized", "raw")


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq2": load_yaml_config("rq2.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }


def routing_rate(gate: GatePolicy, entropies: list[float]) -> float:
    return sum(gate.decide(h) for h in entropies) / len(entropies)


def adaptive_arm(
    scale: str,
    source_signals,
    target_cal_signals,
    target_eval_signals,
    source_view,
    target_view,
    quantile: float,
    source_name: str,
    target_name: str,
    verdict_labels: list[str],
) -> dict:
    """Fixed-vs-adaptive comparison on a single entropy scale.

    `verdict_labels` are the NLI judge's labels for the target model's eval-split
    generations, index-aligned with `target_eval_signals` and independent of `scale`.
    """
    expected_routing_rate = 1.0 - quantile

    source_cal_entropies = entropies_on_scale(source_signals, scale)
    target_cal_entropies = entropies_on_scale(target_cal_signals, scale)
    target_eval_entropies = entropies_on_scale(target_eval_signals, scale)

    # Source native calibration — what gets transferred to produce the fixed gate.
    source_gate = GatePolicy()
    source_threshold = source_gate.calibrate(
        calibration_entropies=source_cal_entropies,
        calibration_indices=source_view.calibration,
        splits=source_view,
        quantile=quantile,
        source=f"{source_name}-{scale}",
    )

    # Fixed condition: source's threshold applied unmodified to the target model.
    fixed_gate = GatePolicy()
    fixed_gate.set_threshold(source_threshold, source=f"transferred-from-{source_name}-{scale}")

    # Adaptive condition: recalibrated natively on the target's own calibration split.
    adaptive_gate = GatePolicy()
    adaptive_threshold = adaptive_gate.calibrate(
        calibration_entropies=target_cal_entropies,
        calibration_indices=target_view.calibration,
        splits=target_view,
        quantile=quantile,
        source=f"adaptive-{target_name}-{scale}",
    )

    fixed_decisions = [fixed_gate.decide(h) for h in target_eval_entropies]
    adaptive_decisions = [adaptive_gate.decide(h) for h in target_eval_entropies]

    def outcomes(decisions: list[bool]) -> list[RoutingOutcome]:
        return [
            RoutingOutcome(entropy=entropy, routed=routed, factuality=label)
            for entropy, routed, label in zip(target_eval_entropies, decisions, verdict_labels)
        ]

    fixed_metrics = routing_detection_metrics(outcomes(fixed_decisions))
    adaptive_metrics = routing_detection_metrics(outcomes(adaptive_decisions))

    fixed_dev_rate = routing_rate(fixed_gate, target_eval_entropies)
    adaptive_dev_rate = routing_rate(adaptive_gate, target_eval_entropies)

    return {
        "entropy_scale": scale,
        "expected_routing_rate": expected_routing_rate,
        "fixed_threshold": source_threshold,
        "adaptive_threshold": adaptive_threshold,
        # Primary evidence: detection performance of each gate on the target model.
        "fixed_gate": fixed_metrics,
        "adaptive_gate": adaptive_metrics,
        # adaptive minus fixed. Positive means recalibrating bought something.
        "routing_quality_delta": routing_quality_delta(fixed_metrics, adaptive_metrics),
        # Secondary diagnostic: does each threshold still hit its own design rate?
        "fixed_dev_routing_rate": fixed_dev_rate,
        "fixed_calibration_fidelity_gap": abs(fixed_dev_rate - expected_routing_rate),
        "adaptive_dev_routing_rate": adaptive_dev_rate,
        "adaptive_calibration_fidelity_gap": abs(adaptive_dev_rate - expected_routing_rate),
    }


def run_pair(
    models_registry,
    examples,
    splits,
    decoding_cfg,
    quantile,
    eval_split,
    source_name,
    target_name,
    nli_model,
    nli_tokenizer,
    nli_cfg,
) -> dict:
    source_cfg = models_registry[source_name]
    target_cfg = models_registry[target_name]

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)
    try:
        source_view = splits.for_checkpoint(source_name)
        target_view = splits.for_checkpoint(target_name)
        eval_indices = getattr(splits, eval_split)

        source_signals = example_signals(
            source_model, source_tokenizer, examples, source_view.calibration, decoding_cfg, source_cfg
        )
        target_cal_signals = example_signals(
            target_model, target_tokenizer, examples, target_view.calibration, decoding_cfg, target_cfg
        )
        target_eval_signals = example_signals(
            target_model, target_tokenizer, examples, eval_indices, decoding_cfg, target_cfg
        )

        verdict_labels = [
            nli_verdict_short_answer(
                nli_model,
                nli_tokenizer,
                signal.generated_text,
                examples[index].right_answer,
                examples[index].hallucinated_answer,
                nli_cfg["short_answer_entailment_threshold"],
            ).label
            for index, signal in zip(eval_indices, target_eval_signals)
        ]

        arms = {
            scale: adaptive_arm(
                scale,
                source_signals,
                target_cal_signals,
                target_eval_signals,
                source_view,
                target_view,
                quantile,
                source_name,
                target_name,
                verdict_labels,
            )
            for scale in ENTROPY_SCALES
        }

        # RQ2, like RQ1, compares routing decisions without acting on them, so
        # nothing abstains and n_abstained is explicitly 0.
        ungated = summarize_factuality([FactualityVerdict(label=lbl) for lbl in verdict_labels], n_abstained=0)

        result = {
            "research_question": "RQ2",
            "dataset": "halueval",
            "eval_split": eval_split,
            "n_eval_examples": len(eval_indices),
            "quantile": quantile,
            "decoding": decoding_cfg,
            "factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
                hf_repo=nli_cfg["hf_repo"], revision=nli_cfg["revision"]
            ),
            "source_model": {"name": source_name, "hf_repo": source_cfg["hf_repo"], "revision": source_cfg["revision"]},
            "target_model": {"name": target_name, "hf_repo": target_cfg["hf_repo"], "revision": target_cfg["revision"]},
            "arms": arms,
            "per_example": [
                {
                    "index": index,
                    "normalized_entropy": signal.normalized_entropy,
                    "raw_entropy": signal.raw_entropy,
                    "factuality": label,
                }
                for index, signal, label in zip(eval_indices, target_eval_signals, verdict_labels)
            ],
        }
        result.update({f"ungated_{key}": value for key, value in ungated.items()})
        return result
    finally:
        del source_model, target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run() -> list[dict]:
    seed_everything()
    config = load_config()
    examples, splits = load_halueval_examples_and_splits()
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    eval_split = config["rq2"]["eval_split"]
    nli_cfg = config["nli_judge"]
    nli_model, nli_tokenizer = load_nli_model(nli_cfg["hf_repo"], nli_cfg["revision"])

    results = []
    for pair in config["rq2"]["transfer_pairs"]:
        source_name, target_name = pair["source_model"], pair["target_model"]
        result = run_pair(
            config["models"],
            examples,
            splits,
            decoding_cfg,
            quantile,
            eval_split,
            source_name,
            target_name,
            nli_model,
            nli_tokenizer,
            nli_cfg,
        )
        write_results(
            f"rq2_adaptive_halueval_{source_name}_to_{target_name}.json",
            result,
            print_exclude_keys=frozenset({"per_example"}),
        )
        results.append(result)
    return results


if __name__ == "__main__":
    run()
