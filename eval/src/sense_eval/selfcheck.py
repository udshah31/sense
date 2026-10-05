"""SelfCheckGPT-NLI: the post-hoc verification baseline.

The proposal (§5) promises a comparison against post-hoc verification alongside the
ungated base model and RAG, citing SelfCheckGPT [12] and Chain-of-Verification [13].
This implements the NLI variant of SelfCheckGPT, which is the cheaper of the two here
because the project already has a pinned NLI model.

**The method.** Sample several stochastic continuations for the same prompt, then score
the main (greedy) answer's consistency against them: a claim the model actually knows
should survive resampling, while a hallucinated one should drift. Per sentence of the
main answer, the score is the mean probability that a sampled passage *contradicts*
it; the passage score is the mean over sentences. Higher means less consistent, so more
likely hallucinated.

**Why it is the baseline that matters for SENSE.** The proposal's distinguishing claim
is that the symbolic component is "neither always on, as in retrieval, nor applied
after the fact, as in post-hoc verification, but invoked selectively during
generation." That is a claim *relative to* after-the-fact checking, and without this
baseline there is nothing quantitative behind it. Two comparisons come out of it:

  - **Detector quality at matched budget.** Calibrate this score's flag threshold at
    the same quantile the entropy gate uses, so both flag the same fraction of
    examples, then compare precision, recall and AUROC. That asks directly whether
    entropy or sampling-consistency is the better hallucination signal per unit of
    intervention.
  - **Cost.** This needs N extra full generations per example. The entropy gate needs
    a logarithm over a distribution the forward pass already produced. That gap is the
    proposal's efficiency argument, and the harness measures it rather than asserting
    it.

**Deviations from the paper, to state as limitations rather than bury.** Manakul et al.
use N = 20 samples and a DeBERTa-v3-large MNLI model; this project's pinned judge is
deberta-v3-xsmall-mnli and the sample count is config-driven and defaults lower, for
compute budget. Both choices weaken the baseline, and a weakened baseline flatters
SENSE — so the sample count and judge size belong in the results discussion, not just
in a config file.
"""

from dataclasses import dataclass
from typing import Callable, Sequence

# A callable scoring many (premise, hypothesis) pairs at once and returning the
# contradiction probability for each, order preserved. Injected rather than taking a
# model directly so the aggregation logic is testable without loading anything.
ContradictionScorer = Callable[[Sequence[tuple[str, str]]], list[float]]


@dataclass(frozen=True)
class SelfCheckScore:
    """Inconsistency of one main answer against its samples.

    `score` is None when it could not be computed — an empty answer, or no samples —
    rather than 0.0, which would read as "perfectly consistent" and mark a
    non-generation as maximally trustworthy.
    """

    score: float | None
    per_sentence_scores: tuple[float, ...]
    n_sentences: int
    n_samples: int

    def as_dict(self) -> dict:
        return {
            "selfcheck_inconsistency": self.score,
            "per_sentence_inconsistency": list(self.per_sentence_scores),
            "n_sentences": self.n_sentences,
            "n_samples": self.n_samples,
        }


def selfcheck_inconsistency(
    sentences: Sequence[str],
    samples: Sequence[str],
    score_contradictions: ContradictionScorer,
) -> SelfCheckScore:
    """Mean contradiction probability of `sentences` against `samples`.

    Each sampled passage is the premise and each sentence of the main answer the
    hypothesis — the direction that asks "does this sample contradict what the main
    answer asserted", which is what SelfCheckGPT-NLI measures.

    All sentence x sample pairs go to `score_contradictions` in one call so a batched
    implementation can use its batching; this is the hot path (N samples per example).
    """
    if not sentences or not samples:
        return SelfCheckScore(None, (), len(sentences), len(samples))

    pairs = [(sample, sentence) for sentence in sentences for sample in samples]
    contradictions = score_contradictions(pairs)
    if len(contradictions) != len(pairs):
        raise ValueError(
            f"scorer returned {len(contradictions)} scores for {len(pairs)} pairs — "
            "the contract is one score per pair, order preserved"
        )

    n_samples = len(samples)
    per_sentence = tuple(
        sum(contradictions[i * n_samples : (i + 1) * n_samples]) / n_samples
        for i in range(len(sentences))
    )
    return SelfCheckScore(
        score=sum(per_sentence) / len(per_sentence),
        per_sentence_scores=per_sentence,
        n_sentences=len(sentences),
        n_samples=n_samples,
    )


def make_nli_contradiction_scorer(model, tokenizer, batch_size: int = 32) -> ContradictionScorer:
    """A `ContradictionScorer` backed by the project's pinned NLI model.

    Resolves the contradiction label from the model's own `id2label` once, so a model
    that orders its labels differently cannot silently invert the score.
    """
    from sense_eval.nli_judge import contradiction_label, entailment_scores_batch

    label = contradiction_label(model)

    def score(pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scored = entailment_scores_batch(model, tokenizer, list(pairs), batch_size=batch_size)
        return [row[label] for row in scored]

    return score
