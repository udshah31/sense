"""NLI-based factuality judge: replaces the substring-containment placeholder
(eval/src/sense_eval/factuality.py's former lexical_containment_verdict) with
real semantic entailment scoring, using a small pinned NLI model.

Two verdict shapes for the two data shapes this project scores:
- nli_verdict_short_answer: HaluEval-style short-answer QA, where a generated
  answer is checked against a right_answer and a hallucinated_answer.
- factscore_style_verdict: FActScore-style open-ended biography generation,
  decomposed into atomic (sentence-level) claims and scored against a
  reference text, matching the real FActScore paper's own methodology
  (decompose into atomic facts, score the supported fraction).

Named limitation (see docs/superpowers/specs/2026-08-16-nli-judge-design.md):
the thresholds this module's callers use are a first cut, chosen for
reasonable behavior on manual spot checks, not calibrated against a
human-labeled validation set — none exists yet for this project. This judge
is real (semantic entailment, not substring matching) but its threshold
calibration is unvalidated. State this wherever its numbers are reported.
"""

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sense_eval.factuality import FactualityVerdict

NLI_METRIC_LABEL_TEMPLATE = "nli_entailment ({hf_repo} @ {revision})"


def load_nli_model(hf_repo: str, revision: str, device: str | None = None):
    """Loads the NLI model + tokenizer. `revision` is required (not
    defaulted) — CLAUDE.md #5, pin the commit SHA, not just the repo name.

    Pins `tokenizer.model_max_length` to the model's real positional limit
    (read from `model.config.max_position_embeddings`, never hard-coded —
    same principle CLAUDE.md states for `tokenizer.vocab_size`). Left at its
    default, this repo's tokenizer reports an effectively-infinite sentinel
    length, so every `truncation=True` call site silently truncates nothing
    ("no maximum length is provided... default to no truncation") — harmless
    for short QA answers, but a real problem once a premise is a full
    Wikipedia article (thousands of tokens): unbounded sequences blew up
    batched-inference memory badly enough to swap-thrash a calibration run
    that should have taken minutes into hours (see
    experiments/calibrate_factscore_thresholds.py's docstring).

    `device` defaults to CUDA if available, else CPU — this project's own
    dev/CI environment has no GPU (CLAUDE.md: "every component must be
    testable on CPU"), so that default is always "cpu" here and behavior is
    unchanged; it only takes effect somewhere CUDA actually exists (e.g. a
    Colab GPU runtime running this same calibration for real speed)."""
    if not revision:
        raise ValueError(f"NLI model '{hf_repo}' has no pinned revision")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo, revision=revision)
    model.eval()
    model.to(device)
    tokenizer.model_max_length = model.config.max_position_embeddings
    return model, tokenizer


def entailment_scores(model, tokenizer, premise: str, hypothesis: str) -> dict[str, float]:
    """Softmax probabilities over the model's own labels for "does `premise`
    entail `hypothesis`", keyed by label name (not index — id2label read at
    call time, since label ordering isn't standardized across NLI models)."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {model.config.id2label[i]: probs[i].item() for i in range(len(probs))}


class ContradictionLabelError(ValueError):
    """Raised when an NLI model's label set has no identifiable contradiction class.

    Loud rather than guessing an index: label ordering is not standardized across NLI
    models (which is why `entailment_scores` keys by name), so picking index 0 or 2 by
    convention would silently invert the score on a model that orders them differently.
    """


def contradiction_label(model) -> str:
    """The model's own name for its contradiction class, resolved at call time.

    Matches case-insensitively on the label containing "contradiction", which covers
    the usual spellings (CONTRADICTION, contradiction, contradiction_label). A model
    using opaque names (LABEL_0/1/2) raises with its label set named, so the fix is to
    map them explicitly rather than to assume an order.
    """
    labels = list(model.config.id2label.values())
    matches = [label for label in labels if "contradiction" in str(label).lower()]
    if len(matches) != 1:
        raise ContradictionLabelError(
            f"expected exactly one contradiction label, found {matches} in {labels} — "
            "map this model's labels explicitly rather than assuming an order"
        )
    return matches[0]


def entailment_scores_batch(
    model, tokenizer, pairs: list[tuple[str, str]], batch_size: int = 32
) -> list[dict[str, float]]:
    """Same contract as `entailment_scores`, one result per (premise, hypothesis)
    pair in `pairs`, order preserved — but batches pairs through the model
    (padded, `batch_size` at a time) instead of one forward pass per pair.

    Exists for bulk offline scoring (calibration sweeps scoring thousands of
    pairs) where per-call Python/tokenization overhead dominates wall time on
    CPU; online call sites (`nli_verdict_short_answer`,
    `factscore_style_verdict`, scoring one example at a time) keep using
    `entailment_scores` — batching a single pair would add padding-shape
    complexity for no benefit there.
    """
    results = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        premises = [premise for premise, _ in batch]
        hypotheses = [hypothesis for _, hypothesis in batch]
        inputs = tokenizer(premises, hypotheses, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        results.extend({model.config.id2label[i]: row[i].item() for i in range(row.shape[0])} for row in probs)
    return results


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_into_atomic_claims(text: str) -> list[str]:
    """A simple sentence-level splitter, not an LLM extractor (see the design
    doc's "Two problems, not one" section) — splits on sentence-ending
    punctuation followed by whitespace, drops empty/whitespace-only
    fragments. This is a deliberate simplification of FActScore's real
    atomic-fact decomposition, not a claim to reproduce it exactly."""
    stripped = text.strip()
    if not stripped:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(stripped) if s.strip()]


_WORD = re.compile(r"\w+")

# How many candidate reference sentences the lexical pre-filter hands to the
# NLI model per claim. Deliberately small and not config-driven — this is an
# implementation detail of the retrieval step, not a scientific threshold
# (see factscore_style_verdict's docstring for why retrieval exists at all).
TOP_K_REFERENCE_SENTENCES = 3


def select_candidate_reference_sentences(claim: str, reference_sentences: list[str], top_k: int) -> list[str]:
    """Cheap, model-free pre-filter: ranks `reference_sentences` by word
    overlap with `claim` and returns the `top_k` best matches (all of them,
    unranked, if there are fewer than `top_k`).

    This exists because scoring a claim's entailment against an entire
    multi-paragraph reference document directly gives badly degraded
    results with this small NLI model, even when the document contains a
    sentence that plainly supports the claim — see factscore_style_verdict's
    docstring for a concrete before/after example. Finding the specific
    relevant sentence first (retrieval), then running NLI on just that
    short premise, is what real FActScore's own methodology does and what
    fixes this. Word overlap is a standard, cheap way to do that filtering
    without a second model — good enough to find the right sentence when
    it shares vocabulary with the claim, which atomic biographical facts
    reliably do (both mention the same name/date/place/institution).
    """
    if len(reference_sentences) <= top_k:
        return list(reference_sentences)
    claim_words = set(_WORD.findall(claim.lower()))
    ranked = sorted(
        reference_sentences,
        key=lambda sentence: len(claim_words & set(_WORD.findall(sentence.lower()))),
        reverse=True,
    )
    return ranked[:top_k]


def best_claim_entailment_batch(
    model, tokenizer, items: list[tuple[str, list[str]]], top_k: int = TOP_K_REFERENCE_SENTENCES
) -> list[tuple[float, str | None]]:
    """For each `(claim, reference_sentences)` pair in `items`, retrieves the
    `top_k` best lexical-overlap candidate sentences and returns the highest
    NLI entailment score among them, plus which sentence achieved it (`None`
    if `reference_sentences` was empty).

    All candidate pairs across every item in `items` are scored in one
    `entailment_scores_batch` call — batching across claims (not just within
    one claim's candidates) is what keeps this tractable at calibration
    scale (tens of thousands of claims); a single `factscore_style_verdict`
    call passes a short `items` list (one biography's claims) and gets the
    same treatment for free.
    """
    per_item_candidates = [
        select_candidate_reference_sentences(claim, reference_sentences, top_k) for claim, reference_sentences in items
    ]
    pairs = [
        (sentence, claim)
        for (claim, _), candidates in zip(items, per_item_candidates)
        for sentence in candidates
    ]
    scores = entailment_scores_batch(model, tokenizer, pairs)

    results = []
    offset = 0
    for candidates in per_item_candidates:
        if not candidates:
            results.append((0.0, None))
            continue
        item_scores = scores[offset : offset + len(candidates)]
        offset += len(candidates)
        best_index = max(range(len(item_scores)), key=lambda i: item_scores[i]["entailment"])
        results.append((item_scores[best_index]["entailment"], candidates[best_index]))
    return results


def nli_verdict_short_answer(
    model,
    tokenizer,
    generated_text: str,
    right_answer: str,
    hallucinated_answer: str,
    entailment_threshold: float,
) -> FactualityVerdict:
    """"correct" if the generated text entails the right answer (and doesn't
    also entail the hallucinated one at/above threshold), "incorrect" the
    symmetric case, "unknown" otherwise (including both or neither clearing
    threshold) — same tri-state contract the retired lexical_containment_verdict
    had, so call sites don't change shape, only semantics."""
    right_entailment = entailment_scores(model, tokenizer, generated_text, right_answer)["entailment"]
    wrong_entailment = entailment_scores(model, tokenizer, generated_text, hallucinated_answer)["entailment"]

    right_clears = right_entailment >= entailment_threshold
    wrong_clears = wrong_entailment >= entailment_threshold

    if right_clears and not wrong_clears:
        return FactualityVerdict(label="correct")
    if wrong_clears and not right_clears:
        return FactualityVerdict(label="incorrect")
    return FactualityVerdict(label="unknown")


@dataclass(frozen=True)
class FActScoreVerdictDetail:
    claims: tuple[str, ...]
    claim_entailment_scores: tuple[float, ...]
    supported_fraction: float | None
    best_reference_sentences: tuple[str | None, ...] = ()


def factscore_style_verdict(
    model,
    tokenizer,
    generated_text: str,
    reference_text: str,
    claim_supported_threshold: float,
    fraction_correct_threshold: float,
    fraction_incorrect_threshold: float,
) -> tuple[FactualityVerdict, FActScoreVerdictDetail]:
    """Splits `generated_text` into atomic claims and `reference_text` into
    sentences, and for each claim scores entailment against its single best-
    matching reference sentence (see `best_claim_entailment_batch`) rather
    than against the whole reference text at once — computes the supported
    fraction, and discretizes it into a tri-state verdict: "correct" if
    supported_fraction >= fraction_correct_threshold, "incorrect" if
    supported_fraction <= fraction_incorrect_threshold, "unknown" otherwise
    (including when there are no claims to score at all). Returns the
    verdict plus a detail record for per-example logging — the aggregate
    number alone would hide exactly the kind of nuance CLAUDE.md's
    abstention-rate warning is about.

    Scoring against the whole reference document as one premise (this
    function's original approach) badly degrades this small NLI model's
    accuracy even when the document plainly contains a supporting sentence
    — concretely, the claim "Taral Hicks is an American" scored 0.06
    entailment against Wikipedia's Taral Hicks article as a whole premise
    (leaning "contradiction" at 0.49) but 0.96 against just that article's
    first three sentences, which contain the literal supporting text
    ("...is an American actress..."). This model was trained on MNLI's
    single-sentence premises; a multi-paragraph premise dilutes its
    decision regardless of whether the relevant sentence survives
    truncation. See `experiments/calibrate_factscore_thresholds.py`'s
    docstring and `results/factscore_threshold_calibration.json` for the
    calibration run that surfaced this and the retrieval-based fix's
    resulting numbers.
    """
    claims = split_into_atomic_claims(generated_text)
    if not claims:
        return FactualityVerdict(label="unknown"), FActScoreVerdictDetail(
            claims=(), claim_entailment_scores=(), supported_fraction=None, best_reference_sentences=()
        )

    reference_sentences = split_into_atomic_claims(reference_text)
    results = best_claim_entailment_batch(model, tokenizer, [(claim, reference_sentences) for claim in claims])
    claim_scores = tuple(score for score, _ in results)
    best_sentences = tuple(sentence for _, sentence in results)

    n_supported = sum(1 for score in claim_scores if score >= claim_supported_threshold)
    supported_fraction = n_supported / len(claims)

    if supported_fraction >= fraction_correct_threshold:
        label = "correct"
    elif supported_fraction <= fraction_incorrect_threshold:
        label = "incorrect"
    else:
        label = "unknown"

    detail = FActScoreVerdictDetail(
        claims=tuple(claims),
        claim_entailment_scores=claim_scores,
        supported_fraction=supported_fraction,
        best_reference_sentences=best_sentences,
    )
    return FactualityVerdict(label=label), detail
