import json

import pytest
import torch
import yaml
from transformers import AutoTokenizer

from _common import (
    run_seed,
    CONFIGS_DIR,
    RESULTS_DIR,
    GPU_CHECKPOINT_KEYS,
    QuantizationSchemeError,
    assert_pinned_gpu_quantization,
    build_generation_inputs,
    load_examples_and_splits,
    load_model,
    load_model_registry,
    write_results,
)
from sense_data.factscore import FActScoreExample
from sense_data.halueval import HaluEvalExample
from sense_data.splits import SplitIndices, PerCheckpointSplitIndices
from sense_data.truthful_qa import TruthfulQAExample

TINY_GPT2 = {"hf_repo": "sshleifer/tiny-gpt2", "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"}


def test_load_examples_and_splits_returns_real_data():
    examples, splits = load_examples_and_splits()

    assert len(examples) > 0
    assert isinstance(examples[0], TruthfulQAExample)
    assert isinstance(splits, SplitIndices)
    assert len(splits.calibration) > 0
    assert len(splits.development) > 0
    assert len(splits.test) > 0


def test_load_halueval_examples_and_splits_returns_real_data():
    from _common import load_halueval_examples_and_splits

    examples, splits = load_halueval_examples_and_splits()

    assert len(examples) == 10_000
    assert isinstance(examples[0], HaluEvalExample)
    assert isinstance(splits, PerCheckpointSplitIndices)
    assert len(splits.development) > 0
    assert len(splits.test) > 0
    assert set(splits.calibration.keys()) == {"llama3", "mistral", "qwen3_8b", "qwen3_4b", "qwen3_1_7b"}


def test_load_factscore_examples_and_splits_returns_real_data():
    from _common import load_factscore_examples_and_splits

    examples, splits = load_factscore_examples_and_splits()

    assert len(examples) == 500
    assert isinstance(examples[0], FActScoreExample)
    assert isinstance(splits, SplitIndices)
    assert len(splits.calibration) > 0
    assert len(splits.development) > 0
    assert len(splits.test) > 0


def test_write_results_writes_full_json_to_results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("_common.RESULTS_DIR", tmp_path)
    result = {"a": 1, "per_example": [1, 2, 3]}

    write_results("test_write_results.json", result)

    written = json.loads((tmp_path / "test_write_results.json").read_text())
    assert written == {**result, "seed": run_seed()}


def test_write_results_prints_summary_excluding_given_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("_common.RESULTS_DIR", tmp_path)
    result = {"a": 1, "per_example": [1, 2, 3]}

    write_results("test_write_results_summary.json", result, print_exclude_keys=frozenset({"per_example"}))

    printed = json.loads(capsys.readouterr().out)
    assert printed == {"a": 1, "seed": run_seed()}
    written = json.loads((tmp_path / "test_write_results_summary.json").read_text())
    assert written == {**result, "seed": run_seed()}


def test_load_model_defaults_to_no_dtype_override_when_quantization_missing():
    model, _ = load_model(TINY_GPT2)

    assert model.dtype == torch.float32


def test_load_model_none_quantization_matches_missing_key():
    model, _ = load_model({**TINY_GPT2, "quantization": "none"})

    assert model.dtype == torch.float32


def test_load_model_bf16_quantization_sets_torch_dtype():
    model, _ = load_model({**TINY_GPT2, "quantization": "bf16"})

    assert model.dtype == torch.bfloat16


def test_load_model_rejects_unknown_quantization_scheme():
    with pytest.raises(ValueError, match="unknown quantization scheme"):
        load_model({**TINY_GPT2, "quantization": "int8"})


def test_load_model_4bit_quantization_passes_bitsandbytes_nf4_config(monkeypatch):
    # bitsandbytes has no macOS wheels (see services/neural/pyproject.toml), so this
    # skips on the CPU dev box and only runs on the GPU environment (Linux). Even
    # there it verifies the *config object* passed to from_pretrained rather than
    # doing an end-to-end load — BitsAndBytesConfig.__init__ itself checks bitsandbytes
    # is importable, which is enough to exercise without a full CUDA model load.
    pytest.importorskip("bitsandbytes")
    import _common
    from transformers import BitsAndBytesConfig

    captured = {}
    real_from_pretrained = _common.AutoModelForCausalLM.from_pretrained

    def fake_from_pretrained(repo, **kwargs):
        captured["quantization_config"] = kwargs.get("quantization_config")
        return real_from_pretrained(repo, revision=kwargs["revision"], token=kwargs["token"])

    monkeypatch.setattr(_common.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)

    load_model({**TINY_GPT2, "quantization": "4bit"})

    cfg = captured["quantization_config"]
    assert isinstance(cfg, BitsAndBytesConfig)
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_compute_dtype == torch.bfloat16


def test_load_model_raises_on_missing_revision():
    with pytest.raises(ValueError, match="no pinned revision"):
        load_model({"hf_repo": "sshleifer/tiny-gpt2"})


def test_load_model_passes_hf_token_env_var_to_from_pretrained(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-test")
    captured = {}
    import _common

    real_model_from_pretrained = _common.AutoModelForCausalLM.from_pretrained
    real_tokenizer_from_pretrained = _common.AutoTokenizer.from_pretrained

    def fake_model_from_pretrained(repo, **kwargs):
        captured["model_token"] = kwargs.get("token")
        return real_model_from_pretrained(repo, **{k: v for k, v in kwargs.items() if k != "token"})

    def fake_tokenizer_from_pretrained(repo, **kwargs):
        captured["tokenizer_token"] = kwargs.get("token")
        return real_tokenizer_from_pretrained(repo, **{k: v for k, v in kwargs.items() if k != "token"})

    monkeypatch.setattr(_common.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(_common.AutoTokenizer, "from_pretrained", fake_tokenizer_from_pretrained)

    load_model(TINY_GPT2)

    assert captured["model_token"] == "fake-token-for-test"
    assert captured["tokenizer_token"] == "fake-token-for-test"


def test_load_model_passes_none_token_when_hf_token_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    captured = {}
    import _common

    real_model_from_pretrained = _common.AutoModelForCausalLM.from_pretrained

    def fake_model_from_pretrained(repo, **kwargs):
        captured["model_token"] = kwargs.get("token")
        return real_model_from_pretrained(repo, **{k: v for k, v in kwargs.items() if k != "token"})

    monkeypatch.setattr(_common.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    load_model(TINY_GPT2)

    assert captured["model_token"] is None


def test_build_generation_inputs_uses_plain_tokenization_without_thinking_mode():
    tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2["hf_repo"])

    inputs = build_generation_inputs(tokenizer, "What is the capital of France?", TINY_GPT2)

    expected = tokenizer("What is the capital of France?", return_tensors="pt")
    assert inputs["input_ids"].tolist() == expected["input_ids"].tolist()


def test_build_generation_inputs_raises_without_chat_template_when_thinking_mode_false():
    tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2["hf_repo"])
    assert tokenizer.chat_template is None  # tiny-gpt2 has no chat template

    with pytest.raises(ValueError, match="no chat_template"):
        build_generation_inputs(tokenizer, "question", {**TINY_GPT2, "thinking_mode": False})


def test_build_generation_inputs_routes_through_chat_template_when_thinking_mode_false(monkeypatch):
    tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2["hf_repo"])
    tokenizer.chat_template = "fake-template"
    captured = {}

    def fake_apply_chat_template(self, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    monkeypatch.setattr(type(tokenizer), "apply_chat_template", fake_apply_chat_template)

    build_generation_inputs(tokenizer, "question", {**TINY_GPT2, "thinking_mode": False})

    assert captured["messages"] == [{"role": "user", "content": "question"}]
    assert captured["kwargs"]["enable_thinking"] is False


def test_build_generation_inputs_routes_through_chat_template_when_present_without_thinking_mode(monkeypatch):
    """llama3/mistral are Instruct-tuned checkpoints with a chat_template but no
    thinking_mode key — they must still go through the template, not plain
    tokenization, or generation feeds them a format they weren't tuned on."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_GPT2["hf_repo"])
    tokenizer.chat_template = "fake-template"
    captured = {}

    def fake_apply_chat_template(self, messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]])}

    monkeypatch.setattr(type(tokenizer), "apply_chat_template", fake_apply_chat_template)

    build_generation_inputs(tokenizer, "question", TINY_GPT2)

    assert captured["messages"] == [{"role": "user", "content": "question"}]
    assert "enable_thinking" not in captured["kwargs"]
    assert captured["kwargs"]["add_generation_prompt"] is True


def test_qwen3_ladder_shares_a_consistent_tokenizer():
    """CLAUDE.md's stated reason for choosing the Qwen3 ladder is that the three
    sizes share a tokenizer (raw entropy comparable across scale, unlike the
    cross-family axis). Verify that at load time rather than assuming it — flag it
    loudly if it's ever not true, since RQ1-RQ3's cross-scale comparisons would be
    invalid without it.
    """
    model_cfg = yaml.safe_load((CONFIGS_DIR / "model.yaml").read_text())["models"]
    qwen3_entries = {name: cfg for name, cfg in model_cfg.items() if name.startswith("qwen3_")}
    assert len(qwen3_entries) == 3, "expected all three Qwen3 ladder rungs in model.yaml"

    vocabs = {
        name: AutoTokenizer.from_pretrained(cfg["hf_repo"], revision=cfg["revision"]).get_vocab()
        for name, cfg in qwen3_entries.items()
    }

    (first_name, first_vocab), *rest = vocabs.items()
    for name, vocab in rest:
        assert vocab == first_vocab, (
            f"Qwen3 tokenizer mismatch: {name} does not share {first_name}'s vocabulary — "
            "the scale-ladder premise (comparable raw entropy across scale) does not hold"
        )


# CLAUDE.md non-negotiable constraint #2: quantization is held constant across every
# GPU checkpoint. This must fail loudly the moment any checkpoint drifts from the
# pinned scheme — same pattern as TokenEntropyMonitor's no-op-generation guard and
# GatePolicy's uncalibrated-decide guard.


def test_assert_pinned_gpu_quantization_passes_for_committed_model_yaml():
    models = yaml.safe_load((CONFIGS_DIR / "model.yaml").read_text())["models"]

    assert_pinned_gpu_quantization(models)  # must not raise


def test_load_model_registry_returns_validated_models_map():
    models = load_model_registry()

    assert GPU_CHECKPOINT_KEYS <= models.keys()


def test_assert_pinned_gpu_quantization_raises_when_a_checkpoint_drifts():
    models = yaml.safe_load((CONFIGS_DIR / "model.yaml").read_text())["models"]
    models["mistral"] = {**models["mistral"], "quantization": "bf16"}

    with pytest.raises(QuantizationSchemeError, match="mistral"):
        assert_pinned_gpu_quantization(models)


def test_assert_pinned_gpu_quantization_raises_when_a_checkpoint_is_missing():
    models = yaml.safe_load((CONFIGS_DIR / "model.yaml").read_text())["models"]
    del models["qwen3_1_7b"]

    with pytest.raises(QuantizationSchemeError, match="qwen3_1_7b"):
        assert_pinned_gpu_quantization(models)
