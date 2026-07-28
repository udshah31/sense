"""RQ2 harness: does a self-adaptive threshold — recalibrated on the target model's
own calibration data — preserve routing quality on held-out data where a fixed
(transferred) threshold does not?

"Routing quality preserved" is operationalized as calibration fidelity: a threshold
calibrated at quantile q is designed to route roughly (1 - q) of calibration-split
examples. On held-out (development-split) data, the adaptive gate's routing rate
should track that same (1 - q) target closely, since it was fit on that model's own
entropy distribution. The fixed gate — the source model's threshold applied
unmodified — has no such guarantee, since it was fit on a different model's
distribution entirely (this is exactly the RQ1 harness's transferred_gate).

This script reuses RQ1's transfer computation (same source/target calibration,
same transferred gate) and adds the recalibrated adaptive gate as the third
condition, so all three — fixed, adaptive, and their common source baseline — are
compared under identical data and decoding config.
"""

from _common import calibration_entropies, load_examples_and_splits, load_model, load_yaml_config, write_results
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    return {
        "models": load_yaml_config("model.yaml")["models"],
        "gate": load_yaml_config("gate.yaml"),
        "rq2": load_yaml_config("rq2.yaml"),
    }


def routing_rate(gate: GatePolicy, entropies: list[float]) -> float:
    return sum(gate.decide(h) for h in entropies) / len(entropies)


def run() -> dict:
    config = load_config()
    source_name = config["rq2"]["source_model"]
    target_name = config["rq2"]["target_model"]
    source_cfg = config["models"][source_name]
    target_cfg = config["models"][target_name]
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    expected_routing_rate = 1.0 - quantile

    examples, splits = load_examples_and_splits()
    eval_indices = getattr(splits, config["rq2"]["eval_split"])

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)

    # Source native calibration — what gets transferred to produce the fixed gate.
    source_gate = GatePolicy()
    source_cal_entropies = calibration_entropies(
        source_model, source_tokenizer, examples, splits.calibration, decoding_cfg
    )
    source_threshold = source_gate.calibrate(
        calibration_entropies=source_cal_entropies,
        calibration_indices=splits.calibration,
        splits=splits,
        quantile=quantile,
        source=source_name,
    )

    # Fixed condition: source's threshold applied unmodified to the target model.
    fixed_gate = GatePolicy()
    fixed_gate.set_threshold(source_threshold, source=f"transferred-from-{source_name}")

    # Adaptive condition: recalibrated natively on the target model's own
    # calibration-split entropies — the self-adaptive threshold RQ2 asks about.
    adaptive_gate = GatePolicy()
    target_cal_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, splits.calibration, decoding_cfg
    )
    adaptive_threshold = adaptive_gate.calibrate(
        calibration_entropies=target_cal_entropies,
        calibration_indices=splits.calibration,
        splits=splits,
        quantile=quantile,
        source=f"adaptive-{target_name}",
    )

    target_eval_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, eval_indices, decoding_cfg
    )

    fixed_dev_rate = routing_rate(fixed_gate, target_eval_entropies)
    adaptive_dev_rate = routing_rate(adaptive_gate, target_eval_entropies)

    result = {
        "research_question": "RQ2",
        "dataset": "truthful_qa",
        "eval_split": config["rq2"]["eval_split"],
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

    write_results(f"rq2_adaptive_truthful_qa_{source_name}_to_{target_name}.json", result)
    return result


if __name__ == "__main__":
    run()
