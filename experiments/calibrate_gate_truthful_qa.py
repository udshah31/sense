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
from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sense_data.splits import load_splits
from sense_data.truthful_qa import load_truthful_qa
from sense_neural.entropy import TokenEntropyMonitor
from sense_orchestrator.gate import GatePolicy

REPO_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "truthful_qa.json"
RESULTS_DIR = REPO_ROOT / "results"


def load_config() -> dict:
    model_cfg = yaml.safe_load((CONFIGS_DIR / "model.yaml").read_text())
    gate_cfg = yaml.safe_load((CONFIGS_DIR / "gate.yaml").read_text())
    active = model_cfg["active"]
    return {
        "model": model_cfg["models"][active],
        "model_name": active,
        "gate": gate_cfg,
    }


def mean_calibration_entropy(model, tokenizer, question: str, decoding_cfg: dict) -> float:
    monitor = TokenEntropyMonitor(vocab_size=tokenizer.vocab_size)
    inputs = tokenizer(question, return_tensors="pt")
    model.generate(
        **inputs,
        max_new_tokens=decoding_cfg["max_new_tokens"],
        do_sample=decoding_cfg["do_sample"],
        logits_processor=[monitor],
    )
    return sum(monitor.entropies) / len(monitor.entropies)


def run() -> dict:
    config = load_config()
    model_cfg = config["model"]

    if not model_cfg.get("revision"):
        raise ValueError(
            f"model '{config['model_name']}' has no pinned revision — "
            "CLAUDE.md requires pinning before any run"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_repo"], revision=model_cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_repo"], revision=model_cfg["revision"])

    examples = load_truthful_qa()
    splits = load_splits(SPLIT_PATH)

    calibration_entropies = []
    for index in splits.calibration:
        entropy = mean_calibration_entropy(model, tokenizer, examples[index].question, config["gate"]["decoding"])
        calibration_entropies.append(entropy)

    gate = GatePolicy()
    threshold = gate.calibrate(
        calibration_entropies=calibration_entropies,
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
        "mean_calibration_entropy": sum(calibration_entropies) / len(calibration_entropies),
        "min_calibration_entropy": min(calibration_entropies),
        "max_calibration_entropy": max(calibration_entropies),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"gate_calibration_truthful_qa_{config['model_name']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
