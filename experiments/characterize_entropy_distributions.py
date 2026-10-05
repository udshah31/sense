"""Pre-flight diagnostic: do these checkpoints' entropy distributions actually differ?

**This is not an RQ1 result.** It uses no factuality labels, scores nothing, and makes
no claim about routing quality. It answers one narrower question, cheaply, before the
full five-checkpoint run is paid for:

    If you calibrated a gate on checkpoint A and applied that threshold unchanged to
    checkpoint B, how far off would B's routing rate land — and does the answer depend
    on which entropy scale the threshold lives on?

Why this exists. RQ1's premise is that entropy is not comparable across models, so a
fixed threshold should misroute. The first pilot
(`results/rq1_transfer_truthful_qa_llama3_to_mistral.json`) returned
`transfer_agreement_rate: 1.0` — perfect transfer — and because only the normalized
scale was ever recorded, that run could not distinguish two very different
explanations:

  1. The premise is wrong and entropy thresholds transfer fine.
  2. The pipeline removed the effect before the gate saw it: dividing by
     `ln(vocab_size)` cancels the cross-tokenizer difference, and quantile calibration
     then cancels any residual difference in distribution location.

Reading the two scales side by side separates those. On the raw scale, a threshold is
a bare number in nats and nothing has been normalized away; on the normalized scale it
is a fraction of each model's own maximum entropy. If raw thresholds scatter and
normalized ones don't, explanation 2 holds and RQ1's answer is "normalization is what
makes the policy portable" — a real finding, but a different paper from the one the
proposal currently sets up.

What it reports, per entropy scale:
  - each checkpoint's distribution: mean, stdev, min/max, and quantiles, plus the
    gate threshold that `configs/gate.yaml`'s quantile would produce
  - `threshold_spread` — max threshold over min threshold across checkpoints. This is
    the single headline number. Near 1.0 means thresholds are interchangeable; large
    means they are not.
  - `transfer_matrix` — for every ordered pair, the routing rate the target would get
    under the source's threshold, and how far that is from the rate the gate was
    designed to produce (`1 - quantile`). This is the practical cost of transferring.
  - `summary_by_axis` — the same, split into cross-family and same-family (scale-ladder)
    pairs, because those are RQ1's two axes and published work suggests they behave
    differently. AdaDec (arXiv:2506.08980, Table VI) reports learned raw-entropy
    thresholds spanning 0.6153 to 1.9353 nats across eight checkpoints, but only
    0.6153-0.7134 within the Qwen3 family — i.e. wide across families, tight across
    scale. Different task and a different fitting procedure, so that is a hypothesis
    to check here, not a result to lean on.

Full per-example entropy values are written to the result file under `entropy_values`
so the distribution plots for the paper can be made locally. No plotting library is
pulled into the GPU environment for that.

Generation uses `example_signals`, exactly the path RQ1 and RQ2 use, so these are the
distributions the real gate will see — not a separate approximation of them.

Evaluated on development, never test (CLAUDE.md's data split discipline). Checkpoints
are loaded and freed one at a time, so peak memory holds one model regardless of how
many are listed.
"""

import gc
import statistics
import sys

import torch

from _common import (
    entropies_on_scale,
    example_signals,
    load_halueval_examples_and_splits,
    load_model,
    load_model_registry,
    load_yaml_config,
    seed_everything,
    write_results,
)
from sense_orchestrator.gate import quantile_threshold

ENTROPY_SCALES = ("normalized", "raw")
REPORTED_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

# Below this many examples, quantile estimates are too noisy to draw a conclusion from.
# The run is not blocked — a small smoke pass is legitimate — but the result file is
# stamped so the numbers can never be read as solid.
MIN_EXAMPLES_FOR_STABLE_QUANTILES = 50


def summarize_distribution(values: list[float], gate_quantile: float) -> dict:
    """Shape of one checkpoint's entropy distribution on one scale."""
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "quantiles": {f"q{int(q * 100):02d}": quantile_threshold(values, q) for q in REPORTED_QUANTILES},
        # The threshold configs/gate.yaml's quantile would produce on this scale.
        "gate_threshold": quantile_threshold(values, gate_quantile),
    }


def ascii_histogram(values: list[float], bins: int = 24, width: int = 48) -> list[str]:
    """Fixed-width histogram for reading distribution shape over an SSH session.

    Not a figure for the paper — the full values go in the result file for that.
    """
    low, high = min(values), max(values)
    if high == low:
        return [f"  all {len(values)} values at {low:.4f}"]

    counts = [0] * bins
    span = high - low
    for value in values:
        index = min(int((value - low) / span * bins), bins - 1)
        counts[index] += 1

    peak = max(counts)
    lines = []
    for index, count in enumerate(counts):
        edge = low + span * index / bins
        bar = "#" * int(count / peak * width) if peak else ""
        lines.append(f"  {edge:7.4f} | {bar:<{width}} {count}")
    return lines


def threshold_spread(thresholds: dict[str, float]) -> dict:
    """How far apart the checkpoints' own thresholds are, on one scale.

    `max_over_min_ratio` is the headline: 1.0 means every checkpoint calibrates to the
    same number and a fixed threshold is trivially portable; a large value means a
    transferred threshold is landing somewhere the target model never calibrated to.
    """
    values = list(thresholds.values())
    smallest, largest = min(values), max(values)
    return {
        "per_model": thresholds,
        "min": smallest,
        "max": largest,
        # Guarded: a zero threshold would mean a model with no entropy anywhere, which
        # is a bug worth seeing as None rather than an inf.
        "max_over_min_ratio": (largest / smallest) if smallest > 0 else None,
        "absolute_range": largest - smallest,
    }


def transfer_matrix(
    thresholds: dict[str, float],
    entropies: dict[str, list[float]],
    families: dict[str, str],
    quantile: float,
) -> list[dict]:
    """For every ordered source->target pair, what a transferred threshold would do.

    No gate object is involved: the decision rule is `entropy >= threshold`, the same
    comparison GatePolicy.decide makes, applied directly. Nothing here is calibrated,
    so no split-leakage surface is opened.
    """
    expected_rate = 1.0 - quantile
    rows = []
    for source in thresholds:
        for target in thresholds:
            if source == target:
                continue
            target_entropies = entropies[target]
            transferred = thresholds[source]
            rate_under_transferred = sum(h >= transferred for h in target_entropies) / len(target_entropies)
            native_rate = sum(h >= thresholds[target] for h in target_entropies) / len(target_entropies)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "same_family": families[source] == families[target],
                    "source_threshold": transferred,
                    "target_native_threshold": thresholds[target],
                    "threshold_ratio": (transferred / thresholds[target]) if thresholds[target] > 0 else None,
                    "target_rate_under_transferred_threshold": rate_under_transferred,
                    "target_rate_under_native_threshold": native_rate,
                    "expected_routing_rate": expected_rate,
                    # The practical cost of transferring: how far the target's routing
                    # rate drifts from what the gate was designed to produce.
                    "abs_deviation_from_expected": abs(rate_under_transferred - expected_rate),
                }
            )
    return rows


def summarize_by_axis(rows: list[dict]) -> dict:
    """Split the transfer matrix into RQ1's two axes and summarize each.

    Returns None for an axis with no pairs rather than a zero, so a single-family
    session cannot be misread as evidence about cross-family transfer.
    """
    summary = {}
    for axis, same_family in (("cross_family", False), ("same_family", True)):
        subset = [row for row in rows if row["same_family"] is same_family]
        if not subset:
            summary[axis] = None
            continue
        ratios = [row["threshold_ratio"] for row in subset if row["threshold_ratio"] is not None]
        summary[axis] = {
            "n_pairs": len(subset),
            "mean_abs_deviation_from_expected": statistics.fmean(
                row["abs_deviation_from_expected"] for row in subset
            ),
            "max_abs_deviation_from_expected": max(row["abs_deviation_from_expected"] for row in subset),
            "max_threshold_ratio": max(ratios) if ratios else None,
            "min_threshold_ratio": min(ratios) if ratios else None,
        }
    return summary


def characterize_model(model_name: str, model_cfg: dict, examples, eval_indices, decoding_cfg, gate_quantile) -> dict:
    """Load one checkpoint, run the eval indices through it, free it."""
    model, tokenizer = load_model(model_cfg)
    try:
        signals = example_signals(model, tokenizer, examples, eval_indices, decoding_cfg, model_cfg)
        scales = {scale: entropies_on_scale(signals, scale) for scale in ENTROPY_SCALES}
        return {
            "metadata": {
                "name": model_name,
                "hf_repo": model_cfg["hf_repo"],
                "revision": model_cfg["revision"],
                "quantization": model_cfg.get("quantization", "none"),
                # Read at load time, never hard-coded (CLAUDE.md constraint #4).
                "vocab_size": tokenizer.vocab_size,
                "n_examples": len(signals),
            },
            "distributions": {
                scale: summarize_distribution(values, gate_quantile) for scale, values in scales.items()
            },
            "values": scales,
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run() -> dict:
    seed_everything()
    models_registry = load_model_registry()
    gate_cfg = load_yaml_config("gate.yaml")
    cfg = load_yaml_config("characterize.yaml")

    eval_split = cfg["eval_split"]
    if eval_split == "test":
        raise ValueError(
            "characterize.yaml sets eval_split: test — the test split is touched once, at "
            "the end, for the reported numbers (CLAUDE.md data split discipline). Use "
            "development."
        )

    decoding_cfg = gate_cfg["decoding"]
    gate_quantile = gate_cfg["quantile"]
    model_names = cfg["models"]
    families = cfg["families"]

    missing_family = [name for name in model_names if name not in families]
    if missing_family:
        raise ValueError(
            f"characterize.yaml lists {missing_family} under `models` but gives them no entry "
            "under `families` — add one rather than letting the pair grouping guess"
        )

    examples, splits = load_halueval_examples_and_splits()
    eval_indices = getattr(splits, eval_split)[: cfg["n_examples"]]

    if len(eval_indices) < MIN_EXAMPLES_FOR_STABLE_QUANTILES:
        print(
            f"WARNING: only {len(eval_indices)} examples — below "
            f"{MIN_EXAMPLES_FOR_STABLE_QUANTILES}, quantile estimates are noisy and the "
            "threshold-spread numbers below should not be used to draw a conclusion.",
            file=sys.stderr,
        )

    per_model = {}
    for model_name in model_names:
        print(f"[characterize] {model_name} — {len(eval_indices)} examples", file=sys.stderr)
        per_model[model_name] = characterize_model(
            model_name, models_registry[model_name], examples, eval_indices, decoding_cfg, gate_quantile
        )

    spreads, matrices, by_axis = {}, {}, {}
    for scale in ENTROPY_SCALES:
        thresholds = {
            name: data["distributions"][scale]["gate_threshold"] for name, data in per_model.items()
        }
        entropies = {name: data["values"][scale] for name, data in per_model.items()}
        spreads[scale] = threshold_spread(thresholds)
        matrices[scale] = transfer_matrix(thresholds, entropies, families, gate_quantile)
        by_axis[scale] = summarize_by_axis(matrices[scale])

    for scale in ENTROPY_SCALES:
        print(f"\n=== {scale} entropy ===", file=sys.stderr)
        for name, data in per_model.items():
            summary = data["distributions"][scale]
            print(
                f"{name}: mean={summary['mean']:.4f} "
                f"gate_threshold(q={gate_quantile})={summary['gate_threshold']:.4f} "
                f"vocab={data['metadata']['vocab_size']}",
                file=sys.stderr,
            )
            for line in ascii_histogram(data["values"][scale]):
                print(line, file=sys.stderr)

    result = {
        "diagnostic": "entropy_distribution_characterization",
        "not_an_rq_result": (
            "Pre-flight diagnostic only. No factuality labels are used and no routing-quality "
            "claim is made. RQ1's answer comes from experiments/transfer_threshold_halueval.py."
        ),
        "dataset": "halueval",
        "eval_split": eval_split,
        "n_examples": len(eval_indices),
        "quantile_estimates_may_be_noisy": len(eval_indices) < MIN_EXAMPLES_FOR_STABLE_QUANTILES,
        "gate_quantile": gate_quantile,
        "decoding": decoding_cfg,
        "per_model": {name: data["metadata"] | {"distributions": data["distributions"]} for name, data in per_model.items()},
        "threshold_spread": spreads,
        "transfer_matrix": matrices,
        "summary_by_axis": by_axis,
        # Full per-example values, for making the distribution figures locally.
        "entropy_values": {name: data["values"] for name, data in per_model.items()},
    }
    write_results(
        "entropy_distribution_characterization.json",
        result,
        print_exclude_keys=frozenset({"entropy_values", "transfer_matrix"}),
    )
    return result


if __name__ == "__main__":
    run()
