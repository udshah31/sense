"""Shared plumbing for experiment scripts: config loading, model loading, the
per-example calibration-entropy computation used by every gate-calibration harness,
and the example/split loading + results-writing boilerplate every harness repeats.
"""

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sense_data.factscore import FActScoreExample, load_factscore
from sense_data.factscore_labeled import FActScoreLabeledExample, load_factscore_labeled
from sense_data.halueval import HaluEvalExample, load_halueval
from sense_data.splits import SplitIndices, load_splits, PerCheckpointSplitIndices, load_per_checkpoint_splits
from sense_data.truthful_qa import TruthfulQAExample, load_truthful_qa
from sense_eval.factuality import assert_factuality_metrics_reported_together
from sense_neural.entropy import TokenEntropyMonitor
from sense_neural.latency import generate_with_latency

REPO_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "truthful_qa.json"
HALUEVAL_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "halueval.json"
FACTSCORE_SPLIT_PATH = REPO_ROOT / "data" / "splits" / "factscore.json"
FACTSCORE_LABELED_DIR = REPO_ROOT / "data" / "fixtures" / "factscore_labeled"
RESULTS_DIR = REPO_ROOT / "results"


def load_yaml_config(name: str) -> dict:
    return yaml.safe_load((CONFIGS_DIR / name).read_text())


def run_seed() -> int:
    """The run-level seed from configs/run.yaml.

    Raises rather than defaulting: a silently-defaulted seed is the same class of
    problem as a silently-defaulted gate threshold (CLAUDE.md: never add a default
    threshold value as a convenience fallback), because it makes an irreproducible
    run look reproducible.
    """
    seed = load_yaml_config("run.yaml").get("seed")
    if seed is None:
        raise ValueError("configs/run.yaml must define `seed` — see CLAUDE.md reproducibility requirements")
    return int(seed)


def seed_everything() -> int:
    """Seed python and torch (CPU + CUDA) from configs/run.yaml, returning the seed.

    Call once at the top of a harness's run(). Greedy decoding is already
    deterministic, so this matters for any sampled condition and for making the
    recorded seed in the result file a true statement about the run rather than a
    decorative field.
    """
    seed = run_seed()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# configs/model.yaml's quantization schemes (CLAUDE.md #2: one scheme, held constant
# across every model and condition it's used in — 4-bit for every GPU checkpoint per
# the 2026-08-10 scope reconciliation). "none"/missing means no dtype override — the
# transformers default (fp32), matching the tiny CPU-test models' existing behavior.
# Unknown values raise rather than silently falling back, per CLAUDE.md's
# never-add-a-silent-default rule.
_QUANTIZATION_SCHEMES = frozenset({"none", "bf16", "4bit"})

# The five checkpoints named in CLAUDE.md's model table. cpu_test/cpu_test_transfer_target
# are CPU-only stand-ins and are exempt — model.yaml already notes their scheme is
# irrelevant at that size.
GPU_CHECKPOINT_KEYS = frozenset({"llama3", "mistral", "qwen3_8b", "qwen3_4b", "qwen3_1_7b"})
PINNED_GPU_QUANTIZATION_SCHEME = "4bit"


class QuantizationSchemeError(ValueError):
    """Raised when a GPU checkpoint's quantization scheme doesn't match the one
    pinned for all five (CLAUDE.md non-negotiable constraint #2). Quantization
    perturbs the logit distribution the entropy gate reads directly, so mixing
    schemes across checkpoints makes cross-model comparisons meaningless — this
    must fail loudly, never default or silently pass, same as GatePolicy's
    uncalibrated-decide guard.
    """


def assert_pinned_gpu_quantization(models: dict) -> None:
    missing = GPU_CHECKPOINT_KEYS - models.keys()
    if missing:
        raise QuantizationSchemeError(f"model.yaml is missing required GPU checkpoints: {sorted(missing)}")

    offending = {
        key: models[key].get("quantization")
        for key in GPU_CHECKPOINT_KEYS
        if models[key].get("quantization") != PINNED_GPU_QUANTIZATION_SCHEME
    }
    if offending:
        raise QuantizationSchemeError(
            f"all five GPU checkpoints must be pinned to quantization: {PINNED_GPU_QUANTIZATION_SCHEME!r} "
            f"(CLAUDE.md non-negotiable constraint #2) — found mismatches: {offending}"
        )


def load_model_registry() -> dict:
    """model.yaml's `models` map, validated against the constant-quantization
    constraint. Every experiment harness should read model configs through this
    instead of load_yaml_config("model.yaml")["models"] directly, so the check runs
    on every real invocation and not only in tests.
    """
    models = load_yaml_config("model.yaml")["models"]
    assert_pinned_gpu_quantization(models)
    return models


def load_model(model_cfg: dict):
    if not model_cfg.get("revision"):
        raise ValueError(f"model '{model_cfg['hf_repo']}' has no pinned revision")

    quantization = model_cfg.get("quantization", "none")
    if quantization not in _QUANTIZATION_SCHEMES:
        raise ValueError(
            f"model '{model_cfg['hf_repo']}' has unknown quantization scheme {quantization!r} — "
            f"expected one of {sorted(_QUANTIZATION_SCHEMES)}"
        )

    # Gated repos (e.g. Llama-3) require an authenticated token; public repos (the
    # CPU-test models, Mistral) ignore it. Read once per call, not at import time,
    # so tests can monkeypatch the environment without reloading the module.
    hf_token = os.environ.get("HF_TOKEN") or None

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["hf_repo"], revision=model_cfg["revision"], token=hf_token)
    model_kwargs = {"revision": model_cfg["revision"], "token": hf_token}
    if quantization == "bf16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif quantization == "4bit":
        # NF4 (bitsandbytes) — compute dtype stays bf16 so the entropy gate's logits
        # keep the same precision profile the earlier bf16-only plan assumed. This
        # is CLAUDE.md's #2 constraint: perturbs the logit distribution somewhat,
        # not zero, so the no-op/bounds correctness checks must still be sanity-
        # checked under it before trusting RQ1-RQ3 numbers (see CLAUDE.md).
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # bitsandbytes 4-bit layers are placed directly onto the GPU at load time —
        # device_map is required (not optional the way it is for bf16/fp32, which
        # default to CPU and work fine there). Without it here, the quantized model
        # ends up split unpredictably and inputs built on CPU mismatch its device.
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_cfg["hf_repo"], **model_kwargs)
    return model, tokenizer


def build_generation_inputs(tokenizer, question: str, model_cfg: dict):
    """Tokenize `question` for generation. Any model whose tokenizer carries a
    chat_template is Instruct-tuned against that template (llama3, mistral, the
    qwen3 ladder) — generating via plain-completion tokenization instead would
    feed it a format it wasn't fine-tuned on. Qwen3 additionally sets
    model.yaml's `thinking_mode: false` on every qwen3_* entry, since forcing
    non-thinking mode requires the enable_thinking=False kwarg on
    apply_chat_template — the only interface Qwen3 exposes for it. Models with
    no chat_template at all (the tiny CPU-test stand-ins) keep the plain-
    completion tokenization the pipeline always used for them.
    """
    if model_cfg.get("thinking_mode") is False:
        if tokenizer.chat_template is None:
            raise ValueError(
                f"model '{model_cfg['hf_repo']}' sets thinking_mode: false but its tokenizer has no "
                "chat_template — cannot enforce non-thinking mode without one"
            )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    return tokenizer(question, return_tensors="pt")


def mean_calibration_entropy(model, tokenizer, question: str, decoding_cfg: dict, model_cfg: dict) -> float:
    monitor = TokenEntropyMonitor(vocab_size=tokenizer.vocab_size)
    inputs = build_generation_inputs(tokenizer, question, model_cfg).to(model.device)
    model.generate(
        **inputs,
        max_new_tokens=decoding_cfg["max_new_tokens"],
        do_sample=decoding_cfg["do_sample"],
        logits_processor=[monitor],
    )
    return sum(monitor.entropies) / len(monitor.entropies)


def calibration_entropies(model, tokenizer, examples, indices, decoding_cfg, model_cfg) -> list[float]:
    return [mean_calibration_entropy(model, tokenizer, examples[i].question, decoding_cfg, model_cfg) for i in indices]


def load_examples_and_splits() -> tuple[list[TruthfulQAExample], SplitIndices]:
    return load_truthful_qa(), load_splits(SPLIT_PATH)


def load_halueval_examples_and_splits() -> tuple[list[HaluEvalExample], PerCheckpointSplitIndices]:
    return load_halueval(), load_per_checkpoint_splits(HALUEVAL_SPLIT_PATH)


def load_factscore_examples_and_splits() -> tuple[list[FActScoreExample], SplitIndices]:
    return load_factscore(), load_splits(FACTSCORE_SPLIT_PATH)


def load_factscore_labeled_examples() -> list[FActScoreLabeledExample]:
    """The FActScore paper's own 183-entity human-annotated set (InstructGPT /
    ChatGPT / PerplexityAI generations, S/NS/IR labels) — disjoint from the
    500-entity `dskar/FActScore` prompt set above and its committed split.
    Used only by experiments/calibrate_factscore_thresholds.py."""
    return load_factscore_labeled(FACTSCORE_LABELED_DIR)


def write_results(filename: str, result: dict, *, print_exclude_keys: frozenset[str] = frozenset()) -> None:
    """Write `result` as the full JSON at RESULTS_DIR/filename, then print a summary
    to stdout with `print_exclude_keys` (e.g. a large per_example array) omitted.

    Every result funnels through here, so this is the one place that enforces
    task_accuracy/hallucination_rate/abstention_rate always being reported
    together (CLAUDE.md/the proposal) — raises rather than writing a result that
    reports a subset. It also carries a dead-code safety net: a warning that
    fires if a result's factuality_metric ever names itself "placeholder". The
    lexical-containment placeholder scorer this originally guarded against is
    fully retired (every harness now scores with the real NLI judge), so this
    branch should never fire today — it's left in place as a tripwire in case a
    future scorer is ever added under a name containing "placeholder".
    """
    # Stamp the run seed unless the caller set one explicitly, so every result file
    # records the seed that produced it (CLAUDE.md: "Fixed seeds, recorded per run").
    result = {**result, "seed": result.get("seed", run_seed())}
    assert_factuality_metrics_reported_together(result)
    factuality_metric = result.get("factuality_metric", "")
    if "placeholder" in factuality_metric.lower():
        print(
            f"WARNING: {filename} was scored with a placeholder factuality metric "
            f"({factuality_metric!r}) — not a real judge, not a reportable result.",
            file=sys.stderr,
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / filename).write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k not in print_exclude_keys}
    print(json.dumps(summary, indent=2))


@dataclass(frozen=True)
class ExampleSignal:
    """Everything one evaluation example yields in a single generation pass.

    `mean_calibration_entropy` above returns only the normalized mean and discards the
    generated text. Both of the things it throws away are needed to measure routing
    quality as detection performance (eval/src/sense_eval/routing_quality.py):

      - `raw_entropy` — the un-normalized mean, in nats. RQ1's premise is that raw
        entropy is not comparable across tokenizers, so testing that premise requires
        the raw scale; a run that records only the normalized value cannot test it
        afterwards.
      - `generated_text` — needed to score the example with the NLI judge, which is
        what supplies the correct/incorrect label the gate's detection performance is
        measured against.

    Captured in one pass, so this costs no extra generation over the entropy-only path
    it replaces — only the NLI judge call at the call site.
    """

    normalized_entropy: float
    raw_entropy: float
    generated_text: str


class EmptyGenerationError(ValueError):
    """Raised when a generation produced no decoding steps, so no entropy was recorded.

    Loud rather than returning 0.0: a zero-entropy example would sit at the bottom of
    every distribution and silently drag a calibrated quantile downward.
    """


def example_signal(model, tokenizer, question: str, decoding_cfg: dict, model_cfg: dict) -> ExampleSignal:
    """Generate once for `question`, returning mean entropy on both scales plus the text.

    Uses generate_with_latency (the same call RQ3 makes) rather than model.generate
    directly, so the generated text follows exactly one decode convention across the
    repo — skip_prompt, skip_special_tokens — instead of two that could drift.
    """
    monitor = TokenEntropyMonitor(vocab_size=tokenizer.vocab_size)
    inputs = build_generation_inputs(tokenizer, question, model_cfg).to(model.device)
    generated = generate_with_latency(model, tokenizer, inputs, decoding_cfg, logits_processor=[monitor])

    if not monitor.entropies:
        raise EmptyGenerationError(
            f"generation for question {question[:60]!r} recorded no decoding steps — "
            "cannot compute a mean entropy"
        )

    return ExampleSignal(
        normalized_entropy=sum(monitor.entropies) / len(monitor.entropies),
        raw_entropy=sum(monitor.raw_entropies) / len(monitor.raw_entropies),
        generated_text=generated["text"],
    )


def example_signals(model, tokenizer, examples, indices, decoding_cfg, model_cfg) -> list[ExampleSignal]:
    """`example_signal` over a list of dataset indices, index-aligned with `indices`."""
    return [
        example_signal(model, tokenizer, examples[i].question, decoding_cfg, model_cfg) for i in indices
    ]


def entropies_on_scale(signals: list[ExampleSignal], scale: str) -> list[float]:
    """Pull one entropy scale out of a signal list. `scale` is "normalized" or "raw".

    Unknown scales raise rather than falling back, so a typo can't silently select the
    wrong scale and quietly change which arm of RQ1 is being measured.
    """
    if scale == "normalized":
        return [s.normalized_entropy for s in signals]
    if scale == "raw":
        return [s.raw_entropy for s in signals]
    raise ValueError(f"unknown entropy scale {scale!r} — expected 'normalized' or 'raw'")
