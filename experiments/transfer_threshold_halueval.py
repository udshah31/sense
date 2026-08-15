"""RQ1 harness: does a fixed entropy-gating threshold, calibrated on one model,
transfer to a model from a different family or scale?

Runs once per (source_model, target_model) pair in configs/rq1.yaml's
transfer_pairs list, writing one result file per pair. Each checkpoint calibrates
on its own disjoint calibration subset (data/splits/halueval.json's per-checkpoint
shape — CLAUDE.md's "genuine held-out calibration splits per checkpoint"
requirement) rather than a shared calibration list.

Procedure per pair:
  1. Calibrate a gate natively on the source model's own calibration-split
     entropies — this is the threshold being tested for transfer.
  2. Calibrate a second gate natively on the target model's own calibration-split
     entropies — this is the target's "ground truth" threshold, used only as a
     comparison baseline, never as what actually gets applied.
  3. Apply the source's threshold directly to the target model (no refitting) via
     GatePolicy.set_threshold, and compare its routing decisions against the
     target's native gate on the development split.

Evaluated on development, never test (CLAUDE.md's data split discipline — test is
touched once, at the end, for the reported numbers).

Models are loaded and explicitly freed around each pair so peak GPU memory never
holds more than the current pair's two checkpoints, regardless of how many pairs
the config lists (some checkpoints, e.g. qwen3_8b, appear in more than one pair).
"""

import gc

import torch

from _common import calibration_entropies, load_halueval_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq1": load_yaml_config("rq1.yaml"),
    }


def run_pair(models_registry, examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name) -> dict:
    source_cfg = models_registry[source_name]
    target_cfg = models_registry[target_name]

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)
    try:
        source_view = splits.for_checkpoint(source_name)
        target_view = splits.for_checkpoint(target_name)
        eval_indices = getattr(splits, eval_split)

        # 1. Native source calibration — this threshold is what gets transferred.
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

        # 2. Native target calibration — comparison baseline only, never applied.
        target_native_gate = GatePolicy()
        target_cal_entropies = calibration_entropies(
            target_model, target_tokenizer, examples, target_view.calibration, decoding_cfg, target_cfg
        )
        target_native_threshold = target_native_gate.calibrate(
            calibration_entropies=target_cal_entropies,
            calibration_indices=target_view.calibration,
            splits=target_view,
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

        return {
            "research_question": "RQ1",
            "dataset": "halueval",
            "eval_split": eval_split,
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
    eval_split = config["rq1"]["eval_split"]

    results = []
    for pair in config["rq1"]["transfer_pairs"]:
        source_name, target_name = pair["source_model"], pair["target_model"]
        result = run_pair(
            config["models"], examples, splits, decoding_cfg, quantile, eval_split, source_name, target_name
        )
        write_results(f"rq1_transfer_halueval_{source_name}_to_{target_name}.json", result)
        results.append(result)
    return results


if __name__ == "__main__":
    run()
