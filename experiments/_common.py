"""Shared plumbing for experiment scripts: config loading, model loading, and the
per-example calibration-entropy computation used by every gate-calibration harness.
"""

from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sense_neural.entropy import TokenEntropyMonitor

REPO_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "truthful_qa.json"
RESULTS_DIR = REPO_ROOT / "results"


def load_yaml_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text())


def load_model(model_cfg: dict):
    if not model_cfg.get("revision"):
        raise ValueError(f"model '{model_cfg['hf_repo']}' has no pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_repo"], revision=model_cfg["revision"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_repo"], revision=model_cfg["revision"])
    return model, tokenizer


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


def calibration_entropies(model, tokenizer, examples, indices, decoding_cfg) -> list[float]:
    return [mean_calibration_entropy(model, tokenizer, examples[i].question, decoding_cfg) for i in indices]
