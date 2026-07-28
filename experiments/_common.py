"""Shared plumbing for experiment scripts: config loading, model loading, the
per-example calibration-entropy computation used by every gate-calibration harness,
and the example/split loading + results-writing boilerplate every harness repeats.
"""

import json
from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sense_data.splits import SplitIndices, load_splits
from sense_data.truthful_qa import TruthfulQAExample, load_truthful_qa
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


def load_examples_and_splits() -> tuple[list[TruthfulQAExample], SplitIndices]:
    return load_truthful_qa(), load_splits(SPLIT_PATH)


def write_results(filename: str, result: dict, *, print_exclude_keys: frozenset[str] = frozenset()) -> None:
    """Write `result` as the full JSON at RESULTS_DIR/filename, then print a summary
    to stdout with `print_exclude_keys` (e.g. a large per_example array) omitted."""
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / filename).write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k not in print_exclude_keys}
    print(json.dumps(summary, indent=2))
