"""RQ1 harness: does a fixed entropy-gating threshold, calibrated on one model,
transfer to a model from a different family?

Procedure:
  1. Calibrate a gate natively on the source model's calibration-split entropies —
     this is the threshold being tested for transfer.
  2. Calibrate a second gate natively on the target model's own calibration-split
     entropies — this is the target's "ground truth" threshold, used only as a
     comparison baseline, never as what actually gets applied.
  3. Apply the source's threshold directly to the target model (no refitting) via
     GatePolicy.set_threshold, and compare its routing decisions against the
     target's native gate on the development split.

Evaluated on development, never test (CLAUDE.md's data split discipline — test is
touched once, at the end, for the reported numbers).
"""

from _common import calibration_entropies, load_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq1": load_yaml_config("rq1.yaml"),
    }


def run() -> dict:
    config = load_config()
    source_name = config["rq1"]["source_model"]
    target_name = config["rq1"]["target_model"]
    source_cfg = config["models"][source_name]
    target_cfg = config["models"][target_name]
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]

    examples, splits = load_examples_and_splits()
    eval_indices = getattr(splits, config["rq1"]["eval_split"])

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)

    # 1. Native source calibration — this threshold is what gets transferred.
    source_gate = GatePolicy()
    source_cal_entropies = calibration_entropies(
        source_model, source_tokenizer, examples, splits.calibration, decoding_cfg, source_cfg
    )
    source_threshold = source_gate.calibrate(
        calibration_entropies=source_cal_entropies,
        calibration_indices=splits.calibration,
        splits=splits,
        quantile=quantile,
        source=source_name,
    )

    # 2. Native target calibration — comparison baseline only, never applied.
    target_native_gate = GatePolicy()
    target_cal_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, splits.calibration, decoding_cfg, target_cfg
    )
    target_native_threshold = target_native_gate.calibrate(
        calibration_entropies=target_cal_entropies,
        calibration_indices=splits.calibration,
        splits=splits,
        quantile=quantile,
        source=target_name,
    )

    # 3. Transferred gate: source's threshold applied directly to the target model.
    transferred_gate = GatePolicy()
    transferred_gate.set_threshold(source_threshold, source=f"transferred-from-{source_name}")

    target_eval_entropies = calibration_entropies(
        target_model, target_tokenizer, examples, eval_indices, decoding_cfg, target_cfg
    )

    transferred_decisions = [transferred_gate.decide(h) for h in target_eval_entropies]
    native_decisions = [target_native_gate.decide(h) for h in target_eval_entropies]

    agreement = sum(t == n for t, n in zip(transferred_decisions, native_decisions)) / len(eval_indices)

    result = {
        "research_question": "RQ1",
        "dataset": "truthful_qa",
        "eval_split": config["rq1"]["eval_split"],
        "n_eval_examples": len(eval_indices),
        "quantile": quantile,
        "decoding": decoding_cfg,
        "source_model": {
            "name": source_name,
            "hf_repo": source_cfg["hf_repo"],
            "revision": source_cfg["revision"],
            "native_threshold": source_threshold,
        },
        "target_model": {
            "name": target_name,
            "hf_repo": target_cfg["hf_repo"],
            "revision": target_cfg["revision"],
            "native_threshold": target_native_threshold,
        },
        "transferred_threshold": source_threshold,
        "transferred_routing_rate": sum(transferred_decisions) / len(eval_indices),
        "native_routing_rate": sum(native_decisions) / len(eval_indices),
        "transfer_agreement_rate": agreement,
    }

    write_results(f"rq1_transfer_truthful_qa_{source_name}_to_{target_name}.json", result)
    return result


if __name__ == "__main__":
    run()
