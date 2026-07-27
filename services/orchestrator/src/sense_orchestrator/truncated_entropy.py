"""Bounds on true token entropy derivable from a truncated top-k logprob response
(the OpenAI-compatible / vLLM serving path, which exposes only top-k logprobs and no
hidden states — see CLAUDE.md's architecture note on why this path can't run the
gated pipeline directly, but can still produce a bounded entropy estimate).

Both bounds follow from two facts about a sorted top-k slice of a probability
distribution over a vocabulary of size V:

  1. Every entropy term -p*log(p) is non-negative, so dropping the unseen V-k terms
     can only ever *reduce* the sum. The entropy computed from the k known
     probabilities alone is therefore always <= the true entropy: a valid lower bound.

  2. Because the top-k slice is sorted descending, every unseen probability is
     <= p_k (the smallest known probability), and their mean is also <= p_k, since
     they sum to the leftover mass R over (V-k) slots. -x*log(x) is concave, so by
     Jensen's inequality the sum of the unseen terms is maximized when that leftover
     mass R is spread as evenly as possible across all V-k unseen slots. That gives a
     closed-form upper bound on the unseen contribution, hence on the true entropy.
"""

import math


def truncated_entropy_bounds(topk_probs: list[float], vocab_size: int) -> tuple[float, float]:
    """Return (lower, upper) bounds on the true Shannon entropy (nats), given only
    the top-k probabilities of the next-token distribution and the full vocab size.
    """
    if not topk_probs:
        raise ValueError("topk_probs must be non-empty")
    k = len(topk_probs)
    if k > vocab_size:
        raise ValueError("topk_probs cannot be longer than vocab_size")

    known_entropy = -sum(p * math.log(p) for p in topk_probs if p > 0)
    lower = known_entropy

    remaining_mass = 1.0 - sum(topk_probs)
    remaining_slots = vocab_size - k
    if remaining_mass <= 0 or remaining_slots == 0:
        upper = known_entropy
    else:
        mean_remaining = remaining_mass / remaining_slots
        tail_upper = -remaining_mass * math.log(mean_remaining)
        upper = known_entropy + tail_upper

    return lower, upper
