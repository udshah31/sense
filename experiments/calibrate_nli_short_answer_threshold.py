"""Calibrates nli_verdict_short_answer's entailment threshold
(configs/nli_judge.yaml's short_answer_entailment_threshold) against real ground
truth, narrowing the design spec's "Named limitation, not calibrated" gap for
HaluEval-shaped harnesses (RQ1-RQ3, the RAG baseline) — see
docs/superpowers/specs/2026-08-16-nli-judge-design.md.

Ground truth without any human labeling: HaluEval's own construction pairs a real
right_answer with a deliberately-fabricated hallucinated_answer for the same
question. For each calibration-split row, this builds two known-label proxy
cases — right_answer as the "generated" text (expected verdict "correct") and
hallucinated_answer as the "generated" text (expected verdict "incorrect") — and
sweeps the threshold to find the value that best resolves both.

Honest limitation, stated here and wherever this calibration's result is cited:
self-entailment (comparing a string against itself) is always near-maximal
regardless of threshold, so this proxy tests whether the threshold correctly
prefers the matching canonical answer over the other one — not how the judge
performs on a realistic, paraphrased or hedged real model generation. It is real
progress over an arbitrary manual-spot-check value, not a substitute for
calibrating against actual model outputs a human has judged.

Uses the union of all five checkpoints' HaluEval calibration splits
(data/splits/halueval.json) — 5,000 pairwise-disjoint examples, none of which
ever appear in the shared development/test splits RQ1-3 report numbers from
(CLAUDE.md's data-split-discipline constraint). Runs entirely on CPU: the NLI
judge model is small (~70M params) and this calibration needs no LLM checkpoint
or GPU at all.

Entailment scores are computed once per example (four scores: right_answer vs.
right_answer/hallucinated_answer, and hallucinated_answer vs. the same pair) and
cached — the threshold sweep afterward only re-runs the tri-state discretization
(pure Python, no further model inference), not the model itself, since only the
discretization depends on the candidate threshold. `_tri_state_verdict` below
duplicates nli_verdict_short_answer's own tri-state contract on precomputed
scores rather than calling it directly (which always recomputes); a dedicated
test asserts the two never drift apart.

Does not write configs/nli_judge.yaml itself — reports the recommended value in
this run's result JSON for a human to review and apply deliberately, per
CLAUDE.md's "never add a default threshold value as a convenience fallback."
"""

import random

from _common import load_halueval_examples_and_splits, load_yaml_config, write_results
from sense_eval.nli_judge import entailment_scores, load_nli_model

CANDIDATE_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95
FIT_FRACTION = 0.8
SPLIT_SEED = 42


def load_config() -> dict:
    return {"nli_judge": load_yaml_config("nli_judge.yaml")}


def build_calibration_pool(splits) -> list[int]:
    """Union of all five checkpoints' disjoint HaluEval calibration splits —
    5,000 examples never touched by RQ1-3's reported development/test numbers."""
    return sorted({index for indices in splits.calibration.values() for index in indices})


def split_fit_holdout(pool: list[int]) -> tuple[list[int], list[int]]:
    """Deterministic 80/20 split of the calibration pool itself — fit the sweep
    on 80%, sanity-check the picked value on the held-out 20%. Both halves stay
    inside the calibration tier; this never touches development or test."""
    shuffled = list(pool)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_fit = round(len(shuffled) * FIT_FRACTION)
    return sorted(shuffled[:n_fit]), sorted(shuffled[n_fit:])


def compute_calibration_scores(model, tokenizer, examples, indices) -> list[dict]:
    """Precomputes the four entailment probabilities each candidate threshold
    needs per example, once per example — right_vs_right, right_vs_wrong,
    wrong_vs_right, wrong_vs_wrong (entailment isn't symmetric, so all four are
    distinct) — so sweeping many thresholds afterward needs no further model
    inference."""
    scores = []
    for index in indices:
        example = examples[index]
        scores.append(
            {
                "index": index,
                "right_vs_right": entailment_scores(model, tokenizer, example.right_answer, example.right_answer)[
                    "entailment"
                ],
                "right_vs_wrong": entailment_scores(
                    model, tokenizer, example.right_answer, example.hallucinated_answer
                )["entailment"],
                "wrong_vs_right": entailment_scores(
                    model, tokenizer, example.hallucinated_answer, example.right_answer
                )["entailment"],
                "wrong_vs_wrong": entailment_scores(
                    model, tokenizer, example.hallucinated_answer, example.hallucinated_answer
                )["entailment"],
            }
        )
    return scores


def _tri_state_verdict(right_entailment: float, wrong_entailment: float, threshold: float) -> str:
    """Same tri-state contract as sense_eval.nli_judge.nli_verdict_short_answer,
    operating on precomputed entailment probabilities instead of recomputing
    them: "correct" if right clears and wrong doesn't, "incorrect" the symmetric
    case, "unknown" otherwise (including both or neither clearing)."""
    right_clears = right_entailment >= threshold
    wrong_clears = wrong_entailment >= threshold
    if right_clears and not wrong_clears:
        return "correct"
    if wrong_clears and not right_clears:
        return "incorrect"
    return "unknown"


def accuracy_at_threshold(cached_scores: list[dict], threshold: float) -> dict:
    """Scores one candidate threshold against both known-label proxy cases per
    example: does the tri-state contract correctly call right_answer "correct"
    and hallucinated_answer "incorrect" at this threshold?"""
    n_correct_verdict = 0
    n_unknown = 0
    n_total = 0
    for row in cached_scores:
        right_case_verdict = _tri_state_verdict(row["right_vs_right"], row["right_vs_wrong"], threshold)
        n_total += 1
        n_correct_verdict += right_case_verdict == "correct"
        n_unknown += right_case_verdict == "unknown"

        wrong_case_verdict = _tri_state_verdict(row["wrong_vs_right"], row["wrong_vs_wrong"], threshold)
        n_total += 1
        n_correct_verdict += wrong_case_verdict == "incorrect"
        n_unknown += wrong_case_verdict == "unknown"

    return {
        "threshold": threshold,
        "n_cases": n_total,
        "verdict_accuracy": n_correct_verdict / n_total,
        "unknown_rate": n_unknown / n_total,
    }


def sweep_thresholds(cached_scores: list[dict], candidate_thresholds) -> list[dict]:
    return [accuracy_at_threshold(cached_scores, threshold) for threshold in candidate_thresholds]


def pick_best_threshold(sweep_results: list[dict]) -> dict:
    """Highest verdict_accuracy; ties broken by lower unknown_rate, then by the
    lower threshold value — deterministic, no silent randomness in the pick."""
    return min(sweep_results, key=lambda r: (-r["verdict_accuracy"], r["unknown_rate"], r["threshold"]))


def run() -> dict:
    config = load_config()
    model, tokenizer = load_nli_model(config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"])

    examples, splits = load_halueval_examples_and_splits()
    pool = build_calibration_pool(splits)
    fit_indices, holdout_indices = split_fit_holdout(pool)

    fit_scores = compute_calibration_scores(model, tokenizer, examples, fit_indices)
    fit_sweep = sweep_thresholds(fit_scores, CANDIDATE_THRESHOLDS)
    best = pick_best_threshold(fit_sweep)

    holdout_scores = compute_calibration_scores(model, tokenizer, examples, holdout_indices)
    holdout_metrics = accuracy_at_threshold(holdout_scores, best["threshold"])

    return {
        "calibration_target": "nli_verdict_short_answer's short_answer_entailment_threshold (configs/nli_judge.yaml)",
        "hf_repo": config["nli_judge"]["hf_repo"],
        "revision": config["nli_judge"]["revision"],
        "n_calibration_pool_examples": len(pool),
        "n_fit_examples": len(fit_indices),
        "n_holdout_examples": len(holdout_indices),
        "candidate_thresholds": CANDIDATE_THRESHOLDS,
        "fit_sweep": fit_sweep,
        "recommended_threshold": best["threshold"],
        "recommended_threshold_fit_metrics": best,
        "recommended_threshold_holdout_metrics": holdout_metrics,
        "method": (
            "proxy ground truth from HaluEval's own right_answer/hallucinated_answer "
            "construction (self-entailment against the canonical answer text, not a "
            "real model generation) — see module docstring for the honest limitation"
        ),
    }


def main() -> dict:
    result = run()
    write_results("nli_short_answer_calibration.json", result, print_exclude_keys=frozenset({"fit_sweep"}))
    return result


if __name__ == "__main__":
    main()
