"""Shared plumbing for experiment scripts: config loading, model loading, the
per-example calibration-entropy computation used by every gate-calibration harness,
and the example/split loading + results-writing boilerplate every harness repeats.
"""

import json
import os
from pathlib import Path

import torch
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


# configs/model.yaml's quantization schemes (CLAUDE.md #2: one scheme, held constant
# across every model and condition it's used in). "none"/missing means no dtype
# override — the transformers default (fp32), matching the tiny CPU-test models'
# existing behavior. Unknown values raise rather than silently falling back, per
# CLAUDE.md's never-add-a-silent-default rule.
_QUANTIZATION_TORCH_DTYPES: dict[str, torch.dtype | None] = {
    "none": None,
    "bf16": torch.bfloat16,
}


def load_model(model_cfg: dict):
    if not model_cfg.get("revision"):
        raise ValueError(f"model '{model_cfg['hf_repo']}' has no pinned revision")

    quantization = model_cfg.get("quantization", "none")
    if quantization not in _QUANTIZATION_TORCH_DTYPES:
        raise ValueError(
            f"model '{model_cfg['hf_repo']}' has unknown quantization scheme {quantization!r} — "
            f"expected one of {sorted(_QUANTIZATION_TORCH_DTYPES)}"
        )
    torch_dtype = _QUANTIZATION_TORCH_DTYPES[quantization]

    # Gated repos (e.g. Llama-3) require an authenticated token; public repos (the
    # CPU-test models, Mistral) ignore it. Read once per call, not at import time,
    # so tests can monkeypatch the environment without reloading the module.
    hf_token = os.environ.get("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_repo"], revision=model_cfg["revision"], token=hf_token)
    model_kwargs = {"revision": model_cfg["revision"], "token": hf_token}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_repo"], **model_kwargs)
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
