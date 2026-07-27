"""Token-level Shannon entropy monitor.

Token entropy (not semantic entropy — see project CLAUDE.md terminology section) is
computed over the next-token distribution at every decoding step via a LogitsProcessor.
The processor is read-only: it records entropy and returns scores unmodified, so
attaching it must not change what gets generated.
"""

import math

import torch
from transformers import LogitsProcessor


class TokenEntropyMonitor(LogitsProcessor):
    """Records normalized Shannon entropy of the next-token distribution at each step.

    Read-only by construction: __call__ returns `scores` unmodified. Entropy is
    normalized by ln(vocab_size) so values are comparable across tokenizers with
    different vocabulary sizes (see CLAUDE.md constraint #4) and land in [0, 1].
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._log_vocab_size = math.log(vocab_size)
        self.entropies: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        log_probs = torch.log_softmax(scores.float(), dim=-1)
        probs = log_probs.exp()
        # Shannon entropy H = -sum(p * log p), computed per batch row; batch size 1
        # is assumed for the single-sequence generation this monitor targets.
        raw_entropy = -(probs * log_probs).sum(dim=-1)
        normalized = (raw_entropy / self._log_vocab_size).item()
        self.entropies.append(normalized)
        return scores

    def reset(self) -> None:
        self.entropies = []
