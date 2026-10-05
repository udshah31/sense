"""RQ1 harness: does a fixed entropy-gating threshold, calibrated on one model,
transfer to a model from a different family or scale?

Runs once per (source_model, target_model) pair in configs/rq1.yaml's
transfer_pairs list, writing one result file per pair. Each checkpoint calibrates
on its own disjoint calibration subset (data/splits/halueval.json's per-checkpoint
shape — CLAUDE.md's "genuine held-out calibration splits per checkpoint"
requirement) rather than a shared calibration list.

Procedure per pair:
  1. Calibrate a gate natively on the source model's own calibration-split
     entropies — this is the threshold being tested for transfer.
  2. Calibrate a second gate natively on the target model's own calibration-split
     entropies — this is the target's "ground truth" threshold, used only as a
     comparison baseline, never as what actually gets applied.
  3. Apply the source's threshold directly to the target model (no refitting) via
     GatePolicy.set_threshold, and compare the two gates' behavior on the
     development split.

Two things changed on 2026-10-04, both from the proposal-v5 review
(docs/research/sense-proposal-v5-review.md, issues C1 and C3).

**Two transfer arms, not one.** Steps 1-3 run twice, on both entropy scales:

  - `normalized` — entropy divided by ln(vocab_size). This was the only arm before.
  - `raw` — entropy in nats, un-normalized.

The reason is that the RQ1 premise is specifically that raw entropy is not comparable
across tokenizers. Normalizing by ln(V) is designed to remove exactly that
incomparability, and quantile calibration then removes any residual difference in
distribution location, so the normalized arm asks whether a threshold transfers
*after* both sources of non-transfer have been engineered out. That is a real and
reportable question, but it is a weaker notion of "fixed threshold" than the one the
RQ is about. The raw arm transfers a threshold as a bare number on the native scale —
the naive thing a practitioner would do — and is the arm RQ1's premise predicts should
fail. Reporting both separates "the signal doesn't transfer" from "normalization fixes
it," which are different findings.

**Routing quality is measured as detection performance.** Agreement between the
transferred and native gates (`transfer_agreement_rate`, retained below) is
*concordance*: it reads 1.0 whenever both gates route the same examples, whether or
not those were the right examples to route. It is kept as a diagnostic, but the
primary evidence is now each gate's detection performance against the ungated model's
own correctness — does the gate fire on the examples the model actually got wrong —
plus the threshold-free AUROC of the entropy signal itself. See
eval/src/sense_eval/routing_quality.py for the argument and the metric definitions.

The target model's eval-split generation is produced once and scored once by the NLI
judge; both arms reuse those verdicts, since the generation does not depend on which
scale the gate reads. So the added cost over the previous entropy-only version is the
NLI judge pass, not a second round of generation.

Evaluated on development, never test (CLAUDE.md's data split discipline — test is
touched once, at the end, for the reported numbers).

Models are loaded and explicitly freed around each pair so peak GPU memory never
holds more than the current pair's two checkpoints, regardless of how many pairs
the config lists (some checkpoints, e.g. qwen3_8b, appear in more than one pair).
"""

import gc

import torch

from _common import (
    bootstrap_config,
    derived_seed,
    entropies_on_scale,
    example_signals,
    load_halueval_examples_and_splits,
    load_model,
    load_model_registry,
    load_yaml_config,
    seed_everything,
    write_results,
)
from sense_eval.bootstrap import percentile_bootstrap
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, load_nli_model, nli_verdict_short_answer
from sense_eval.routing_quality import (
    RoutingOutcome,
    detection_metrics_with_ci,
    routing_quality_delta,
    routing_quality_delta_with_ci,
)
from sense_orchestrator.gate import GatePolicy, quantile_threshold

# Both entropy scales TokenEntropyMonitor records. Each gets its own full transfer arm.
ENTROPY_SCALES = ("normalized", "raw")


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rq1": load_yaml_config("rq1.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }


def threshold_confidence_interval(
    calibration_entropies: list[float],
    quantile: float,
    bootstrap_cfg: dict,
    seed_label: tuple[str, ...],
) -> dict:
    """Bootstrap interval for the calibrated threshold itself.

    Resamples the calibration entropies and refits the quantile, so the interval
    answers "how much would this threshold move on another calibration draw from the
    same model". That is the dominant source of uncertainty in RQ1 and RQ2, since every
    downstream number is a function of the threshold. Costs no GPU: the entropies are
    already computed.

    `quantile_threshold` is the gate's own function, so the resampled thresholds are
    fit exactly the way the real one was.
    """
    return percentile_bootstrap(
        calibration_entropies,
        lambda values: quantile_threshold(list(values), quantile),
        n_resamples=bootstrap_cfg["n_resamples"],
        confidence=bootstrap_cfg["confidence"],
        seed=derived_seed(bootstrap_cfg["seed"], *seed_label, "threshold"),
    ).as_dict()


def transfer_arm(
    scale: str,
    source_signals,
    target_cal_signals,
    target_eval_signals,
    source_view,
    target_view,
    quantile: float,
    source_name: str,
    target_name: str,
    verdict_labels: list[str],
    bootstrap_cfg: dict,
) -> dict:
    """One full calibrate -> transfer -> compare cycle on a single entropy scale.

    `verdict_labels` are the NLI judge's per-example labels for the target model's
    eval-split generations, index-aligned with `target_eval_signals`. They do not
    depend on `scale` — the generation is the same either way — so both arms are
    scored against the same labels.
    """
    source_cal_entropies = entropies_on_scale(source_signals, scale)
    target_cal_entropies = entropies_on_scale(target_cal_signals, scale)
    target_eval_entropies = entropies_on_scale(target_eval_signals, scale)

    # 1. Native source calibration — this threshold is what gets transferred.
    source_gate = GatePolicy()
    source_threshold = source_gate.calibrate(
        calibration_entropies=source_cal_entropies,
        calibration_indices=source_view.calibration,
        splits=source_view,
        quantile=quantile,
        source=f"{source_name}-{scale}",
    )

    # 2. Native target calibration — comparison baseline only, never applied.
    target_native_gate = GatePolicy()
    target_native_threshold = target_native_gate.calibrate(
        calibration_entropies=target_cal_entropies,
        calibration_indices=target_view.calibration,
        splits=target_view,
        quantile=quantile,
        source=f"{target_name}-{scale}",
    )

    # 3. Transferred gate: source's threshold applied directly to the target model.
    transferred_gate = GatePolicy()
    transferred_gate.set_threshold(source_threshold, source=f"transferred-from-{source_name}-{scale}")

    transferred_decisions = [transferred_gate.decide(h) for h in target_eval_entropies]
    native_decisions = [target_native_gate.decide(h) for h in target_eval_entropies]

    n_eval = len(target_eval_entropies)
    agreement = sum(t == n for t, n in zip(transferred_decisions, native_decisions)) / n_eval

    def outcomes(decisions: list[bool]) -> list[RoutingOutcome]:
        return [
            RoutingOutcome(entropy=entropy, routed=routed, factuality=label)
            for entropy, routed, label in zip(target_eval_entropies, decisions, verdict_labels)
        ]

    seed_label = (source_name, target_name, scale)
    transferred_outcomes = outcomes(transferred_decisions)
    native_outcomes = outcomes(native_decisions)
    transferred_metrics = detection_metrics_with_ci(
        transferred_outcomes,
        n_resamples=bootstrap_cfg["n_resamples"],
        confidence=bootstrap_cfg["confidence"],
        seed=derived_seed(bootstrap_cfg["seed"], *seed_label, "transferred"),
    )
    native_metrics = detection_metrics_with_ci(
        native_outcomes,
        n_resamples=bootstrap_cfg["n_resamples"],
        confidence=bootstrap_cfg["confidence"],
        seed=derived_seed(bootstrap_cfg["seed"], *seed_label, "native"),
    )

    return {
        "entropy_scale": scale,
        "source_native_threshold": source_threshold,
        "target_native_threshold": target_native_threshold,
        "transferred_threshold": source_threshold,
        # Concordance diagnostic, not the primary evidence — see module docstring.
        "transfer_agreement_rate": agreement,
        "source_threshold_ci": threshold_confidence_interval(
            source_cal_entropies, quantile, bootstrap_cfg, (source_name, "source", scale)
        ),
        "target_native_threshold_ci": threshold_confidence_interval(
            target_cal_entropies, quantile, bootstrap_cfg, (target_name, "native", scale)
        ),
        "transferred_gate": transferred_metrics,
        "native_gate": native_metrics,
        # The proposal's "difference in routing quality": native minus transferred.
        # Positive means the transferred threshold cost something on this metric.
        "routing_quality_delta": routing_quality_delta(transferred_metrics, native_metrics),
        # Paired interval on that difference — the two gates are scored on the same
        # examples, so this is the interval that says whether the gap exceeds sampling
        # noise. An interval excluding zero is RQ1's evidence; one straddling zero is a
        # finding in its own right and must be reported as one.
        "routing_quality_delta_ci": routing_quality_delta_with_ci(
            transferred_outcomes,
            native_outcomes,
            n_resamples=bootstrap_cfg["n_resamples"],
            confidence=bootstrap_cfg["confidence"],
            seed=derived_seed(bootstrap_cfg["seed"], *seed_label, "delta"),
        ),
    }


def run_pair(
    models_registry,
    examples,
    splits,
    decoding_cfg,
    quantile,
    eval_split,
    source_name,
    target_name,
    nli_model,
    nli_tokenizer,
    nli_cfg,
    bootstrap_cfg,
) -> dict:
    source_cfg = models_registry[source_name]
    target_cfg = models_registry[target_name]

    source_model, source_tokenizer = load_model(source_cfg)
    target_model, target_tokenizer = load_model(target_cfg)
    try:
        source_view = splits.for_checkpoint(source_name)
        target_view = splits.for_checkpoint(target_name)
        eval_indices = getattr(splits, eval_split)

        # One generation pass per example per model; both entropy scales and the
        # generated text come back together (see _common.ExampleSignal).
        source_signals = example_signals(
            source_model, source_tokenizer, examples, source_view.calibration, decoding_cfg, source_cfg
        )
        target_cal_signals = example_signals(
            target_model, target_tokenizer, examples, target_view.calibration, decoding_cfg, target_cfg
        )
        target_eval_signals = example_signals(
            target_model, target_tokenizer, examples, eval_indices, decoding_cfg, target_cfg
        )

        # Score the target's eval-split generations once. These labels are what the
        # gate's detection performance is measured against, and they are scale-
        # independent, so both arms share them.
        verdicts = [
            nli_verdict_short_answer(
                nli_model,
                nli_tokenizer,
                signal.generated_text,
                examples[index].right_answer,
                examples[index].hallucinated_answer,
                nli_cfg["short_answer_entailment_threshold"],
            )
            for index, signal in zip(eval_indices, target_eval_signals)
        ]
        verdict_labels = [v.label for v in verdicts]

        arms = {
            scale: transfer_arm(
                scale,
                source_signals,
                target_cal_signals,
                target_eval_signals,
                source_view,
                target_view,
                quantile,
                source_name,
                target_name,
                verdict_labels,
                bootstrap_cfg,
            )
            for scale in ENTROPY_SCALES
        }

        # The target model's ungated factuality, reported as the required triple.
        # RQ1 applies no merge-back (it compares routing decisions, it does not act on
        # them), so nothing abstains and n_abstained is explicitly 0 rather than
        # defaulted — summarize_factuality requires the caller to state this.
        ungated = summarize_factuality([FactualityVerdict(label=lbl) for lbl in verdict_labels], n_abstained=0)

        result = {
            "research_question": "RQ1",
            "dataset": "halueval",
            "eval_split": eval_split,
            "n_eval_examples": len(eval_indices),
            "quantile": quantile,
            "decoding": decoding_cfg,
            "bootstrap": bootstrap_cfg,
            "factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
                hf_repo=nli_cfg["hf_repo"], revision=nli_cfg["revision"]
            ),
            "source_model": {
                "name": source_name,
                "hf_repo": source_cfg["hf_repo"],
                "revision": source_cfg["revision"],
            },
            "target_model": {
                "name": target_name,
                "hf_repo": target_cfg["hf_repo"],
                "revision": target_cfg["revision"],
            },
            "arms": arms,
            "per_example": [
                {
                    "index": index,
                    "normalized_entropy": signal.normalized_entropy,
                    "raw_entropy": signal.raw_entropy,
                    "factuality": label,
                }
                for index, signal, label in zip(eval_indices, target_eval_signals, verdict_labels)
            ],
        }
        # Prefixed so the always-together check treats this as its own triple,
        # independent of any other factuality numbers in the file.
        result.update({f"ungated_{key}": value for key, value in ungated.items()})
        return result
    finally:
        del source_model, target_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run() -> list[dict]:
    seed_everything()
    config = load_config()
    examples, splits = load_halueval_examples_and_splits()
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    eval_split = config["rq1"]["eval_split"]
    nli_cfg = config["nli_judge"]
    nli_model, nli_tokenizer = load_nli_model(nli_cfg["hf_repo"], nli_cfg["revision"])
    bootstrap_cfg = bootstrap_config()

    results = []
    for pair in config["rq1"]["transfer_pairs"]:
        source_name, target_name = pair["source_model"], pair["target_model"]
        result = run_pair(
            config["models"],
            examples,
            splits,
            decoding_cfg,
            quantile,
            eval_split,
            source_name,
            target_name,
            nli_model,
            nli_tokenizer,
            nli_cfg,
            bootstrap_cfg,
        )
        write_results(
            f"rq1_transfer_halueval_{source_name}_to_{target_name}.json",
            result,
            print_exclude_keys=frozenset({"per_example"}),
        )
        results.append(result)
    return results


if __name__ == "__main__":
    run()
