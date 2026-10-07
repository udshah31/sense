"""Prototype: a margin-based short-answer verdict, evaluated against the absolute-threshold rule.

**Not wired into the judge, and not a result.** `nli_verdict_short_answer` (question-prefixed,
PR #8) calls a generation "correct" when the right answer's entailment clears one absolute
threshold and the hallucinated answer's does not. With the question on both sides of the NLI
pair, the shared text inflates BOTH answers' entailment, so that rule needs a very high
threshold (0.97) on canonical answers — which real hedged generations (median ~0.93) do not
reach. A margin rule asks instead whether the right answer is entailed MORE than the
hallucinated one, which a shared prefix mostly cancels:

    m = right_entailment - hallucinated_entailment
    correct   if m >=  delta
    incorrect if m <= -delta
    unknown   otherwise
    (variant "floor": additionally require max(right, hallucinated) >= floor, so two near-zero
     scores that happen to differ slightly cannot produce a verdict)

This script scores the calibration pool's canonical answers (the same stand-in as
calibrate_nli_short_answer_threshold.py, same 80/20 fit/held-out split, calibration tier only —
never development or test) and sweeps delta. The scores are cached so sweeps are instant.
Its limitation is the one already named: canonical text is not a real generation.

Second mode, `probes`: the stand-in cannot choose delta or test the floor (any delta near 0 wins
on canonical text, and the floor never binds). So it also scores SYNTHETIC probes that imitate
the failure modes seen on real Mistral output, on calibration-pool examples only:
  hedged_right  "The answer is likely {right}"        should be correct
  hedged_wrong  "The answer is likely {hallucinated}"  should be incorrect
  unrelated     another example's right answer         should be unknown
Synthetic text is not model output; this is a prototype check, not evidence.

Usage (from experiments/):
  uv run python prototype_margin_verdict.py [cache_path]
  uv run python prototype_margin_verdict.py probes [n_examples] [cache_path]
"""

import json
import sys
from pathlib import Path

from _common import load_halueval_examples_and_splits, load_yaml_config
from calibrate_nli_short_answer_threshold import (
    build_calibration_pool,
    compute_calibration_scores,
    split_fit_holdout,
)
from sense_eval.nli_judge import load_nli_model

DELTAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FLOOR = 0.5


def margin_verdict(right: float, wrong: float, delta: float, floor: float | None = None) -> str:
    if floor is not None and max(right, wrong) < floor:
        return "unknown"
    margin = right - wrong
    if margin >= delta:
        return "correct"
    if margin <= -delta:
        return "incorrect"
    return "unknown"


def threshold_verdict(right: float, wrong: float, threshold: float) -> str:
    rc, wc = right >= threshold, wrong >= threshold
    return "correct" if rc and not wc else "incorrect" if wc and not rc else "unknown"


def accuracy(scores: list[dict], verdict) -> dict:
    """Both stand-in cases per example: text=right should be "correct", text=wrong "incorrect"."""
    hit = unknown = wrong_label = total = 0
    for row in scores:
        for expected, (r, w) in (
            ("correct", (row["right_vs_right"], row["right_vs_wrong"])),
            ("incorrect", (row["wrong_vs_right"], row["wrong_vs_wrong"])),
        ):
            v = verdict(r, w)
            total += 1
            hit += v == expected
            unknown += v == "unknown"
            wrong_label += v not in (expected, "unknown")
    return {"accuracy": hit / total, "unknown": unknown / total, "opposite_label": wrong_label / total}


def run_probes(n_examples: int, cache_path: str) -> None:
    import random

    from sense_eval.nli_judge import entailment_scores, short_answer_nli_pair

    cache = Path(cache_path)
    examples, splits = load_halueval_examples_and_splits()
    pool = build_calibration_pool(splits)
    fit_idx, _ = split_fit_holdout(pool)
    rng = random.Random(0)
    chosen = rng.sample(fit_idx, n_examples)
    others = [examples[i].right_answer for i in rng.sample(fit_idx, n_examples)]

    if cache.exists():
        probes = json.loads(cache.read_text())
    else:
        cfg = load_yaml_config("nli_judge.yaml")
        model, tokenizer = load_nli_model(cfg["hf_repo"], cfg["revision"])

        def rw(example, text):
            return tuple(
                entailment_scores(model, tokenizer, *short_answer_nli_pair(example.question, text, a))["entailment"]
                for a in (example.right_answer, example.hallucinated_answer)
            )

        probes = {"hedged_right": [], "hedged_wrong": [], "unrelated": []}
        for i, other in zip(chosen, others):
            e = examples[i]
            probes["hedged_right"].append(rw(e, f"The answer is likely {e.right_answer}"))
            probes["hedged_wrong"].append(rw(e, f"The answer is likely {e.hallucinated_answer}"))
            probes["unrelated"].append(rw(e, other))
        cache.write_text(json.dumps(probes))

    rules = {"threshold 0.7": lambda r, w: threshold_verdict(r, w, 0.7), "threshold 0.97": lambda r, w: threshold_verdict(r, w, 0.97)}
    for d in (0.01, 0.05, 0.1, 0.2, 0.3):
        rules[f"margin d={d}"] = lambda r, w, d=d: margin_verdict(r, w, d)
        rules[f"margin d={d}+floor"] = lambda r, w, d=d: margin_verdict(r, w, d, FLOOR)
    expected = {"hedged_right": "correct", "hedged_wrong": "incorrect", "unrelated": "unknown"}
    print(f"n={len(probes['unrelated'])} per probe | share labeled as expected (hedged_right->correct, hedged_wrong->incorrect, unrelated->unknown)")
    print(f"{'rule':22s} {'h_right':>8s} {'h_wrong':>8s} {'unrel':>8s}  false-verdict-on-unrelated")
    for name, rule in rules.items():
        cells = {}
        for probe, exp in expected.items():
            vs = [rule(r, w) for r, w in probes[probe]]
            cells[probe] = sum(v == exp for v in vs) / len(vs)
        unrel_false = 1 - cells["unrelated"]
        print(f"{name:22s} {cells['hedged_right']:8.3f} {cells['hedged_wrong']:8.3f} {cells['unrelated']:8.3f}  {unrel_false:.3f}")


def main(cache_path: str) -> None:
    cache = Path(cache_path)
    examples, splits = load_halueval_examples_and_splits()
    pool = build_calibration_pool(splits)
    fit_idx, hold_idx = split_fit_holdout(pool)

    if cache.exists():
        scores = json.loads(cache.read_text())
    else:
        cfg = load_yaml_config("nli_judge.yaml")
        model, tokenizer = load_nli_model(cfg["hf_repo"], cfg["revision"])
        scores = {
            "fit": compute_calibration_scores(model, tokenizer, examples, fit_idx),
            "holdout": compute_calibration_scores(model, tokenizer, examples, hold_idx),
        }
        cache.write_text(json.dumps(scores))

    def line(name, verdict):
        f, h = accuracy(scores["fit"], verdict), accuracy(scores["holdout"], verdict)
        print(
            f"{name:22s} fit acc {f['accuracy']:.4f} unk {f['unknown']:.4f} opp {f['opposite_label']:.4f} | "
            f"holdout acc {h['accuracy']:.4f} unk {h['unknown']:.4f} opp {h['opposite_label']:.4f}"
        )
        return f["accuracy"]

    print("--- absolute-threshold rule (current judge) ---")
    for t in (0.7, 0.97):
        line(f"threshold {t}", lambda r, w, t=t: threshold_verdict(r, w, t))
    for label, floor in (("margin", None), (f"margin+floor{FLOOR}", FLOOR)):
        print(f"--- {label} rule: sweep delta (fit split) ---")
        results = {d: accuracy(scores["fit"], lambda r, w, d=d: margin_verdict(r, w, d, floor)) for d in DELTAS}
        for d, r in results.items():
            print(f"  delta {d:<5} fit acc {r['accuracy']:.4f} unk {r['unknown']:.4f} opp {r['opposite_label']:.4f}")
        best = max(results, key=lambda d: (results[d]["accuracy"], -results[d]["unknown"], -d))
        edge = best in (DELTAS[0], DELTAS[-1])
        print(f"  best delta {best}{'  (RANGE EDGE - not an interior optimum)' if edge else ''}")
        line(f"{label} delta {best}", lambda r, w, d=best, fl=floor: margin_verdict(r, w, d, fl))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probes":
        run_probes(
            int(sys.argv[2]) if len(sys.argv) > 2 else 1000,
            sys.argv[3] if len(sys.argv) > 3 else "/tmp/margin_probe_scores.json",
        )
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/margin_pool_scores.json")
