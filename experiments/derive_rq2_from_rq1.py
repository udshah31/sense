"""RQ2 derived from RQ1's saved results; no GPU, no new generation.

RQ2 asks whether a self-adaptive threshold (recalibrated on the target model's own
calibration split) preserves routing quality where a fixed (transferred) one does not.
RQ1 already builds exactly those two gates on the same target-model generations —
"transferred" is the fixed gate, "native" is the adaptive one — and reports their paired
difference with a bootstrap interval. RQ2's harness (adaptive_threshold_halueval.py)
recomputes the same two gates from the same inputs, so re-running it would repeat the
same generations to reproduce RQ1's numbers. This script re-expresses RQ1's summaries
as RQ2: fixed vs adaptive, adaptive-minus-fixed, plus the secondary calibration-fidelity
gap (|observed routing rate - designed (1 - quantile)|). The paper should say RQ1 and RQ2
share one experiment viewed two ways. Not a separate measurement.

Usage (from experiments/):  python derive_rq2_from_rq1.py [test]
Reads full development results when available, otherwise the archived development summaries.
With `test`: requires full rq1_transfer_test_halueval_*.json results.
Every input must identify the requested eval_split. No model generation is performed.
"""

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"


def f1_verdict(f1_delta: list[float]) -> str:
    """f1_delta is [point, ci_low, ci_high] of adaptive minus fixed."""
    if f1_delta[1] > 0:
        return "adaptive better"
    if f1_delta[2] < 0:
        return "fixed better"
    return "no clear difference"


def summarize_full(rq1: dict) -> dict:
    """Convert full harness output to the compact schema without recomputing metrics."""
    summary = {k: v for k, v in rq1.items() if k not in ("per_example", "arms")}
    metric_names = {
        "routing_rate": "routing_rate",
        "precision": "routed_hallucination_precision",
        "recall": "hallucination_recall",
        "f1": "f1", "specificity": "specificity",
        "balanced_accuracy": "balanced_accuracy",
    }
    summary["arms"] = {}
    for scale, arm in rq1["arms"].items():
        compact = {k: v for k, v in arm.items()
                   if k not in ("routing_quality_delta", "routing_quality_delta_ci")}
        for gate_name in ("transferred_gate", "native_gate"):
            gate = arm[gate_name]
            compact[gate_name] = {short: gate[long] for short, long in metric_names.items()}
            compact[gate_name].update({short: gate[long] for short, long in {
                "tp": "n_true_positive", "fp": "n_false_positive",
                "fn": "n_false_negative", "tn": "n_true_negative",
            }.items()})
        compact["delta_native_minus_transferred"] = {}
        for short, long in metric_names.items():
            ci = arm["routing_quality_delta_ci"][f"delta_{long}_ci"]
            compact["delta_native_minus_transferred"][short] = [
                ci[k] for k in ("point", "ci_low", "ci_high")
            ]
        summary["arms"][scale] = compact
    return summary


def load_pairs(split: str) -> list[dict]:
    infix = "_test" if split == "test" else ""
    prefix = f"rq1_transfer{infix}_halueval_"
    full_paths = {p.name[:-5]: p for p in RESULTS.glob(prefix + "*.json")
                  if not p.name.endswith(".console_summary.json")}
    paths = dict(full_paths)
    if split == "development":
        for p in RESULTS.glob(prefix + "*.console_summary.json"):
            paths.setdefault(p.name.removesuffix(".console_summary.json"), p)
    if not paths:
        raise ValueError(f"No {split} results found")
    pairs = []
    for key, path in sorted(paths.items()):
        result = json.loads(path.read_text())
        if result.get("eval_split") != split:
            raise ValueError(f"{path.name}: expected eval_split={split!r}, got {result.get('eval_split')!r}")
        if key in full_paths:
            result = summarize_full(result)
        pair = derive(result)
        pair["source_file"] = path.name
        pairs.append(pair)
    return pairs


def derive(rq1: dict) -> dict:
    expected = 1.0 - rq1["quantile"]
    arms = {}
    for scale, arm in rq1["arms"].items():
        fixed, adaptive = arm["transferred_gate"], arm["native_gate"]
        arms[scale] = {
            "fixed_threshold": arm["source_native_threshold"],
            "adaptive_threshold": arm["target_native_threshold"],
            "fixed_gate": fixed,
            "adaptive_gate": adaptive,
            "adaptive_minus_fixed": arm["delta_native_minus_transferred"],
            "fixed_calibration_fidelity_gap": abs(fixed["routing_rate"] - expected),
            "adaptive_calibration_fidelity_gap": abs(adaptive["routing_rate"] - expected),
            "f1_verdict": f1_verdict(arm["delta_native_minus_transferred"]["f1"]),
        }
    return {
        "source": rq1["source_model"]["name"],
        "target": rq1["target_model"]["name"],
        "n_eval_examples": rq1["n_eval_examples"],
        "eval_split": rq1.get("eval_split"),
        "expected_routing_rate": expected,
        "arms": arms,
    }


def main() -> None:
    if sys.argv[1:] not in ([], ["test"]):
        raise SystemExit("Usage: derive_rq2_from_rq1.py [test]")
    split = "test" if sys.argv[1:] == ["test"] else "development"
    infix = "_test" if split == "test" else ""
    pairs = load_pairs(split)
    out = {
        "NOTE": f"Derived from the RQ1 saved results (source_file recorded per pair); RQ1 and RQ2 share one experiment (see derive_rq2_from_rq1.py). {split.capitalize()} split, binary judge at 0.7. Delta = adaptive minus fixed = RQ1's native minus transferred; intervals are RQ1's paired bootstrap. CAUTION: with precision near 0.96 routing more raises recall and F1 mechanically, so 'fixed better' on the Qwen3 scale-ladder pairs (where the fixed gate routes more, 13-21% vs 8-12%) reflects routing rate, not gate quality; read it with the routing rates and fidelity gaps.",
        "pairs": pairs,
    }
    (RESULTS / f"rq2_from_rq1{infix}.json").write_text(json.dumps(out, indent=1))
    print(f"{'pair':28s} {'scale':10s} {'fixed rr':>8s} {'adapt rr':>8s} {'dF1 [95% CI]':>24s}  verdict")
    for p in pairs:
        for scale, a in p["arms"].items():
            d = a["adaptive_minus_fixed"]["f1"]
            print(
                f"{p['source'] + '->' + p['target']:28s} {scale:10s} {a['fixed_gate']['routing_rate']:8.3f} "
                f"{a['adaptive_gate']['routing_rate']:8.3f} {d[0]:+7.3f} [{d[1]:+.3f},{d[2]:+.3f}]  {a['f1_verdict']}"
            )


if __name__ == "__main__":
    main()
