"""Truncated-vs-exact entropy correctness check, required by CLAUDE.md: the
orchestrator's API-derived (truncated top-k) estimate must carry explicit lower and
upper bounds, and the exact in-process value must fall within them.

sense_neural is a test-only dependency here (see pyproject.toml) — it supplies the
ground-truth exact entropy computed in-process; the orchestrator never talks to it
this way at runtime, only over HTTP.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sense_neural.entropy import exact_token_entropy
from sense_orchestrator.truncated_entropy import truncated_entropy_bounds

TINY_MODEL = "sshleifer/tiny-gpt2"


def _next_token_probs(prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return logits, probs, tokenizer.vocab_size


def test_exact_entropy_falls_within_truncated_bounds():
    logits, probs, vocab_size = _next_token_probs("The quick brown fox")
    exact = exact_token_entropy(logits.unsqueeze(0))

    topk = torch.topk(probs, k=10)
    topk_probs = topk.values.tolist()

    lower, upper = truncated_entropy_bounds(topk_probs, vocab_size)

    assert lower <= exact <= upper


def test_bounds_tighten_as_k_grows():
    logits, probs, vocab_size = _next_token_probs("The quick brown fox")

    narrow_lower, narrow_upper = truncated_entropy_bounds(
        torch.topk(probs, k=5).values.tolist(), vocab_size
    )
    wide_lower, wide_upper = truncated_entropy_bounds(
        torch.topk(probs, k=50).values.tolist(), vocab_size
    )

    assert (wide_upper - wide_lower) <= (narrow_upper - narrow_lower)


def test_full_vocab_top_k_collapses_bounds_to_a_point():
    _, probs, vocab_size = _next_token_probs("The quick brown fox")
    full = torch.topk(probs, k=vocab_size).values.tolist()

    lower, upper = truncated_entropy_bounds(full, vocab_size)
    assert abs(upper - lower) < 1e-9
