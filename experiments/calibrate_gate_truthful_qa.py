"""Wire the entropy-gating policy against real TruthfulQA calibration entropies.

For each calibration-split example, greedily generate a short answer from the
config-selected model with TokenEntropyMonitor attached, and reduce its per-token
entropy trace to a single scalar (mean token entropy over the generated answer) —
that scalar is this example's calibration signal. The resulting per-example
entropies feed GatePolicy.calibrate, which is itself leakage-guarded against the
committed TruthfulQA split.

Config-driven per CLAUDE.md: model identity from configs/model.yaml, gate quantile
and decoding config from configs/gate.yaml. Swapping `active` in model.yaml to
llama3/mistral on the GPU environment runs the identical code path.
"""

import json

from _common import RESULTS_DIR, SPLIT_PATH, calibration_entropies, load_model, load_yaml_config
from sense_data.splits import load_splits
from sense_data.truthful_qa import load_truthful_qa
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    model_cfg = load_yaml_config("model.yaml")
    gate_cfg = load_yaml_config("gate.yaml")
    active = model_cfg["active"]
    return {
        "model": model_cfg["models"][active],
        "model_name": active,
        "gate": gate_cfg,
    }


def run() -> dict:
    config = load_config()
    model_cfg = config["model"]

    model, tokenizer = load_model(model_cfg)

    examples = load_truthful_qa()
    splits = load_splits(SPLIT_PATH)

    entropies = calibration_entropies(model, tokenizer, examples, splits.calibration, config["gate"]["decoding"])

    gate = GatePolicy()
    threshold = gate.calibrate(
        calibration_entropies=entropies,
        calibration_indices=splits.calibration,
        splits=splits,
        quantile=config["gate"]["quantile"],
        source=f"{config['model_name']}:{model_cfg['hf_repo']}@{model_cfg['revision']}",
    )

    result = {
        "dataset": "truthful_qa",
        "model_name": config["model_name"],
        "hf_repo": model_cfg["hf_repo"],
        "revision": model_cfg["revision"],
        "quantization": model_cfg["quantization"],
        "quantile": config["gate"]["quantile"],
        "decoding": config["gate"]["decoding"],
        "n_calibration_examples": len(splits.calibration),
        "threshold": threshold,
        "calibration_source": gate.calibration_source,
        "mean_calibration_entropy": sum(entropies) / len(entropies),
        "min_calibration_entropy": min(entropies),
        "max_calibration_entropy": max(entropies),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"gate_calibration_truthful_qa_{config['model_name']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
