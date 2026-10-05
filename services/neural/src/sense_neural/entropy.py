"""Token-level Shannon entropy monitor.

Token entropy (not semantic entropy — see project CLAUDE.md terminology section) is
computed over the next-token distribution at every decoding step via a LogitsProcessor.
The processor is read-only: it records entropy and returns scores unmodified, so
attaching it must not change what gets generated.

Both scales are retained. The normalized scale (raw / ln(vocab_size)) is what the gate
calibrates on by default; the raw scale (nats) is kept because the project's RQ1 premise
is specifically that *un-normalized* entropy is not comparable across tokenizers, and a
run that only records the normalized value cannot be used to test that premise
afterwards. Recording both costs nothing — the raw value is computed either way — and
lets RQ1 run a raw-scale transfer arm alongside the normalized one.
"""

import math

import torch
from transformers import LogitsProcessor


def exact_token_entropy(scores: torch.FloatTensor) -> float:
    """Exact Shannon entropy (nats) of the next-token distribution given full-vocab
    logits. Used as ground truth against the orchestrator's truncated top-k estimate.
    """
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).item()


class TokenEntropyMonitor(LogitsProcessor):
    """Records Shannon entropy of the next-token distribution at each step, on both
    the normalized and raw scales.

    Read-only by construction: __call__ returns `scores` unmodified.

    - `entropies` — normalized by ln(vocab_size), so values are comparable across
      tokenizers with different vocabulary sizes (CLAUDE.md constraint #4) and land
      in [0, 1]. This is the default gating scale and the attribute name every
      existing call site uses.
    - `raw_entropies` — the same values in nats, before normalization. Not comparable
      across tokenizers by construction; that is the point. RQ1's raw-scale arm
      transfers a threshold on this scale, which is the naive thing a practitioner
      would do and the condition the RQ1 premise is actually about.

    The two lists are always the same length and index-aligned.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._log_vocab_size = math.log(vocab_size)
        self.entropies: list[float] = []
        self.raw_entropies: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raw_entropy = exact_token_entropy(scores)
        self.raw_entropies.append(raw_entropy)
        self.entropies.append(raw_entropy / self._log_vocab_size)
        return scores

    def reset(self) -> None:
        self.entropies = []
        self.raw_entropies = []
