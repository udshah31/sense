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

Usage (from experiments/):  python derive_rq2_from_rq1.py
Reads results/rq1_transfer_halueval_*.console_summary.json; writes results/rq2_from_rq1.json.
"""

import json
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"


def f1_verdict(f1_delta: list[float]) -> str:
    """f1_delta is [point, ci_low, ci_high] of adaptive minus fixed."""
    if f1_delta[1] > 0:
        return "adaptive better"
    if f1_delta[2] < 0:
        return "fixed better"
    return "no clear difference"


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
        "expected_routing_rate": expected,
        "arms": arms,
    }


def main() -> None:
    pairs = [derive(json.loads(p.read_text())) for p in sorted(RESULTS.glob("rq1_transfer_halueval_*.console_summary.json"))]
    out = {
        "NOTE": "Derived from the RQ1 summaries; RQ1 and RQ2 share one experiment (see derive_rq2_from_rq1.py). Development split, binary judge at 0.7. Delta = adaptive minus fixed = RQ1's native minus transferred; intervals are RQ1's paired bootstrap. CAUTION: with precision near 0.96 routing more raises recall and F1 mechanically, so 'fixed better' on the Qwen3 scale-ladder pairs (where the fixed gate routes more, 13-21% vs 8-12%) reflects routing rate, not gate quality; read it with the routing rates and fidelity gaps.",
        "pairs": pairs,
    }
    (RESULTS / "rq2_from_rq1.json").write_text(json.dumps(out, indent=1))
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
