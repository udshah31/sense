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

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_METRIC_LABEL_TEMPLATE = "nli_entailment ({hf_repo} @ {revision})"


def load_nli_model(hf_repo: str, revision: str):
    """Loads the NLI model + tokenizer. `revision` is required (not
    defaulted) — CLAUDE.md #5, pin the commit SHA, not just the repo name."""
    if not revision:
        raise ValueError(f"NLI model '{hf_repo}' has no pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(hf_repo, revision=revision)
    model.eval()
    return model, tokenizer


def entailment_scores(model, tokenizer, premise: str, hypothesis: str) -> dict[str, float]:
    """Softmax probabilities over the model's own labels for "does `premise`
    entail `hypothesis`", keyed by label name (not index — id2label read at
    call time, since label ordering isn't standardized across NLI models)."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return {model.config.id2label[i]: probs[i].item() for i in range(len(probs))}
