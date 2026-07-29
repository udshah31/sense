import json

import pytest
import torch

from _common import RESULTS_DIR, load_examples_and_splits, load_model, write_results
from sense_data.splits import SplitIndices
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


def test_write_results_writes_full_json_to_results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("_common.RESULTS_DIR", tmp_path)
    result = {"a": 1, "per_example": [1, 2, 3]}

    write_results("test_write_results.json", result)

    written = json.loads((tmp_path / "test_write_results.json").read_text())
    assert written == result


def test_write_results_prints_summary_excluding_given_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("_common.RESULTS_DIR", tmp_path)
    result = {"a": 1, "per_example": [1, 2, 3]}

    write_results("test_write_results_summary.json", result, print_exclude_keys=frozenset({"per_example"}))

    printed = json.loads(capsys.readouterr().out)
    assert printed == {"a": 1}
    written = json.loads((tmp_path / "test_write_results_summary.json").read_text())
    assert written == result


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
