"""RQ2 harness: does a self-adaptive threshold — recalibrated on the target model's
own calibration data — preserve routing quality on held-out data where a fixed
(transferred) threshold does not?

"Routing quality preserved" is operationalized as calibration fidelity: a threshold
calibrated at quantile q is designed to route roughly (1 - q) of calibration-split
examples. On held-out (development-split) data, the adaptive gate's routing rate
should track that same (1 - q) target closely, since it was fit on that model's own
entropy distribution. The fixed gate — the source model's threshold applied
unmodified — has no such guarantee, since it was fit on a different model's
distribution entirely.

Runs once per (source_model, target_model) pair in configs/rq2.yaml's
transfer_pairs list, writing one result file per pair. Each checkpoint calibrates
on its own disjoint calibration subset (data/splits/halueval.json's per-checkpoint
shape), same as RQ1.

Models are loaded and explicitly freed around each pair so peak GPU memory never
holds more than the current pair's two checkpoints.
"""

import gc

import torch

from _common import calibration_entropies, load_halueval_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq2": load_yaml_config("rq2.yaml"),
    }


def routing_rate(gate: GatePolicy, entropies: list[float]) -> float:
    return sum(gate.decide(h) for h in entropies) / len(entropies)


def run_pair(models_registry, examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name) -> dict:
    source_cfg = models_registry[source_name]
    target_cfg = models_registry[target_name]
    expected_routing_rate = 1.0 - quantile

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)
    try:
        source_view = splits.for_checkpoint(source_name)
        target_view = splits.for_checkpoint(target_name)
        eval_indices = getattr(splits, eval_split)

        # Source native calibration — what gets transferred to produce the fixed gate.
        source_gate = GatePolicy()
        source_cal_entropies = calibration_entropies(
            source_model, source_tokenizer, examples, source_view.calibration, decoding_cfg, source_cfg
        )
        source_threshold = source_gate.calibrate(
            calibration_entropies=source_cal_entropies,
            calibration_indices=source_view.calibration,
            splits=source_view,
            quantile=quantile,
            source=source_name,
        )

        # Fixed condition: source's threshold applied unmodified to the target model.
        fixed_gate = GatePolicy()
        fixed_gate.set_threshold(source_threshold, source=f"transferred-from-{source_name}")

        # Adaptive condition: recalibrated natively on the target model's own
        # calibration-split entropies.
        adaptive_gate = GatePolicy()
        target_cal_entropies = calibration_entropies(
            target_model, target_tokenizer, examples, target_view.calibration, decoding_cfg, target_cfg
        )
        adaptive_threshold = adaptive_gate.calibrate(
            calibration_entropies=target_cal_entropies,
            calibration_indices=target_view.calibration,
            splits=target_view,
            quantile=quantile,
            source=f"adaptive-{target_name}",
        )

        target_eval_entropies = calibration_entropies(
            target_model, target_tokenizer, examples, eval_indices, decoding_cfg, target_cfg
        )

        fixed_dev_rate = routing_rate(fixed_gate, target_eval_entropies)
        adaptive_dev_rate = routing_rate(adaptive_gate, target_eval_entropies)

        return {
            "research_question": "RQ2",
            "dataset": "halueval",
            "eval_split": eval_split,
            "n_eval_examples": len(eval_indices),
            "quantile": quantile,
            "expected_routing_rate": expected_routing_rate,
            "decoding": decoding_cfg,
            "source_model": {"name": source_name, "hf_repo": source_cfg["hf_repo"], "revision": source_cfg["revision"]},
            "target_model": {"name": target_name, "hf_repo": target_cfg["hf_repo"], "revision": target_cfg["revision"]},
            "fixed_threshold": source_threshold,
            "fixed_dev_routing_rate": fixed_dev_rate,
            "fixed_calibration_fidelity_gap": abs(fixed_dev_rate - expected_routing_rate),
            "adaptive_threshold": adaptive_threshold,
            "adaptive_dev_routing_rate": adaptive_dev_rate,
            "adaptive_calibration_fidelity_gap": abs(adaptive_dev_rate - expected_routing_rate),
        }
    finally:
        del source_model, target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run() -> list[dict]:
    config = load_config()
    examples, splits = load_halueval_examples_and_splits()
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    eval_split = config["rq2"]["eval_split"]

    results = []
    for pair in config["rq2"]["transfer_pairs"]:
        source_name, target_name = pair["source_model"], pair["target_model"]
        result = run_pair(
            config["models"], examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name
        )
        write_results(f"rq2_adaptive_halueval_{source_name}_to_{target_name}.json", result)
        results.append(result)
    return results


if __name__ == "__main__":
    run()
