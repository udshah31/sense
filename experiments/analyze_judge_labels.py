"""Score candidate short-answer verdict rules against the hand-labeled real generations.

Reads data/fixtures/judge_labels/judge_labels_*.json (see its `rubric`/`annotator`: labels
are Claude's, not a human's). Pure Python, no model. Rules are chosen on the `fit` rows
and reported once on `holdout`; the sample is small and imbalanced (few "correct"), so
the headline metric is BALANCED accuracy and a boundary optimum is flagged, not trusted.

Rules (verdict "correct" iff ...):
  threshold t     R >= t and W < t              (the judge's current rule)
  right-only t    R >= t
  margin d        R - W >= d
  margin d+floor  R - W >= d and R >= floor
where R/W are the question-prefixed entailment of the right/hallucinated answer.

Usage (from experiments/):  uv run python analyze_judge_labels.py [fixture_path]
"""

import json
import sys
from pathlib import Path

from _common import REPO_ROOT

FIXTURE = REPO_ROOT / "data" / "fixtures" / "judge_labels" / "judge_labels_2026-10-07.json"
GRID = [round(0.05 * i, 2) for i in range(1, 20)] + [0.96, 0.97, 0.98, 0.99]


def confusion(rows: list[dict], rule) -> dict:
    tp = fp = tn = fn = 0
    for r in rows:
        predicted = rule(r["right_entailment"], r["hallucinated_entailment"])
        actual = r["label"] == "correct"
        tp += predicted and actual
        fp += predicted and not actual
        fn += (not predicted) and actual
        tn += (not predicted) and not actual
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "balanced_accuracy": (recall + specificity) / 2,
    }


def candidate_rules() -> dict:
    rules = {}
    for t in GRID:
        rules[("threshold", t)] = lambda r, w, t=t: r >= t and w < t
        rules[("right-only", t)] = lambda r, w, t=t: r >= t
        rules[("margin", t)] = lambda r, w, t=t: r - w >= t
        rules[("margin+floor0.5", t)] = lambda r, w, t=t: r - w >= t and r >= 0.5
    return rules


def best_per_family(rows: list[dict]) -> dict:
    rules = candidate_rules()
    best = {}
    for (family, param), rule in rules.items():
        score = confusion(rows, rule)["balanced_accuracy"]
        if family not in best or (score, -param) > (best[family][1], -best[family][0]):
            best[family] = (param, score)
    return best


def main(path: Path) -> None:
    data = json.loads(path.read_text())
    rows = data["rows"]
    fit = [r for r in rows if r["tier"] == "fit"]
    holdout = [r for r in rows if r["tier"] == "holdout"]
    n_correct = sum(r["label"] == "correct" for r in rows)
    print(f"{len(rows)} rows ({len(fit)} fit / {len(holdout)} holdout), {n_correct} labeled correct; annotator: {data['annotator']}")
    for model in sorted({r["model"] for r in rows}):
        m = [r for r in rows if r["model"] == model]
        print(f"  {model}: {sum(r['label'] == 'correct' for r in m)}/{len(m)} correct")

    rules = candidate_rules()
    print("\nfit-set balanced accuracy, best parameter per rule family; then that rule on holdout:")
    for family, (param, score) in best_per_family(fit).items():
        rule = rules[(family, param)]
        edge = param in (GRID[0], GRID[-1])
        h = confusion(holdout, rule)
        f = confusion(fit, rule)
        print(
            f"  {family:16s} param={param:<5} fit bal-acc {score:.3f} (P {f['precision']:.2f} R {f['recall']:.2f}) | "
            f"holdout bal-acc {h['balanced_accuracy']:.3f} (P {h['precision']:.2f} R {h['recall']:.2f}, tp {h['tp']} fp {h['fp']} fn {h['fn']})"
            + ("  [GRID EDGE]" if edge else "")
        )
    cur = rules[("threshold", 0.7)]
    c = confusion(rows, cur)
    print(f"\ncurrent judge (threshold 0.7) on all rows: bal-acc {c['balanced_accuracy']:.3f} P {c['precision']:.2f} R {c['recall']:.2f} tp {c['tp']} fp {c['fp']} fn {c['fn']} tn {c['tn']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE)
