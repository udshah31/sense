"""Correctness checks for TokenEntropyMonitor, per CLAUDE.md's required checks:
the monitor must be a no-op on generation, and entropy values must be well-formed.
"""

import math

import torch
from transformers import LogitsProcessorList

from sense_neural.entropy import TokenEntropyMonitor

PROMPT = "The quick brown fox"


def _generate(model, tokenizer, logits_processor=None, seed=0):
    torch.manual_seed(seed)
    inputs = tokenizer(PROMPT, return_tensors="pt")
    kwargs = {}
    if logits_processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([logits_processor])
    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,  # greedy decoding, per CLAUDE.md's decoding-config constraint
        **kwargs,
    )
    return output


def test_monitor_is_noop_on_generation(tiny_model, tiny_tokenizer):
    baseline = _generate(tiny_model, tiny_tokenizer, logits_processor=None)
    monitor = TokenEntropyMonitor(vocab_size=tiny_tokenizer.vocab_size)
    monitored = _generate(tiny_model, tiny_tokenizer, logits_processor=monitor)

    assert torch.equal(baseline, monitored), (
        "TokenEntropyMonitor changed the generated tokens — it must be read-only."
    )
    assert len(monitor.entropies) > 0


def test_entropy_values_are_normalized(tiny_model, tiny_tokenizer):
    monitor = TokenEntropyMonitor(vocab_size=tiny_tokenizer.vocab_size)
    _generate(tiny_model, tiny_tokenizer, logits_processor=monitor)

    assert len(monitor.entropies) == 10  # one entry per generated token
    for h in monitor.entropies:
        assert 0.0 <= h <= 1.0 + 1e-6, f"normalized entropy {h} out of [0, 1]"
        assert not math.isnan(h)


def test_vocab_size_read_from_tokenizer_not_hardcoded(tiny_tokenizer):
    monitor = TokenEntropyMonitor(vocab_size=tiny_tokenizer.vocab_size)
    assert monitor.vocab_size == tiny_tokenizer.vocab_size
    assert monitor.vocab_size != 50000  # sanity: not some arbitrary hardcoded value


def test_reset_clears_recorded_entropies(tiny_model, tiny_tokenizer):
    monitor = TokenEntropyMonitor(vocab_size=tiny_tokenizer.vocab_size)
    _generate(tiny_model, tiny_tokenizer, logits_processor=monitor)
    assert len(monitor.entropies) > 0

    monitor.reset()
    assert monitor.entropies == []
