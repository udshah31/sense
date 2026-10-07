"""Post-hoc verification baseline: SelfCheckGPT-NLI on HaluEval.

The proposal's §5 promises a comparison against post-hoc verification alongside the
ungated base model and RAG. Until 2026-10-04 that baseline did not exist in the repo
(proposal-v5 review issue M2), which left the proposal's central framing — the symbolic
component is "neither always on, as in retrieval, nor applied after the fact, as in
post-hoc verification, but invoked selectively during generation" — with nothing
quantitative behind the second half of it.

Per example, per checkpoint:
  1. Generate the main answer greedily, with gate.yaml's decoding config — identical to
     every other condition in the study, so the thing being scored is comparable.
  2. Generate `n_samples` sampled continuations for the same prompt.
  3. Score the main answer's inconsistency against those samples with
     SelfCheckGPT-NLI (sense_eval.selfcheck): mean probability that a sample
     contradicts each sentence of the main answer.
  4. Flag the example if the inconsistency score crosses a threshold calibrated on
     this checkpoint's own calibration split.

**Matched budget.** The flag threshold is calibrated at the same quantile the entropy
gate uses, so this detector and the gate flag the same fraction of examples and their
precision, recall and AUROC are directly comparable. Comparing two detectors operating
at different rates would confound signal quality with intervention rate.

GatePolicy is reused for the thresholding rather than reimplemented. It is named for
the entropy gate, but what it provides is exactly what is needed here — a
quantile-calibrated threshold over a scalar signal, with the split-leakage guard and
the refuse-if-uncalibrated rule — and reusing it keeps one implementation of those
rules instead of two. The signal differs; the thresholding discipline should not.

**Cost is the point, so it is measured.** This method needs n_samples + 1 full
generations per example. The entropy gate needs a logarithm over a distribution the
forward pass already produced. Per-stage latency is recorded so that gap is a reported
number rather than an assertion.

Evaluated on development, never test (CLAUDE.md's data split discipline).
"""

import gc

import torch

from _common import (
    bootstrap_config,
    build_generation_inputs,
    derived_seed,
    load_halueval_examples_and_splits,
    load_model,
    load_model_registry,
    load_yaml_config,
    seed_everything,
    write_results,
)
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, load_nli_model, nli_verdict_short_answer, split_into_atomic_claims
from sense_eval.routing_quality import RoutingOutcome, detection_metrics_with_ci
from sense_eval.selfcheck import make_nli_contradiction_scorer, selfcheck_inconsistency
from sense_neural.latency import generate_with_latency
from sense_orchestrator.gate import GatePolicy


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "selfcheck": load_yaml_config("selfcheck.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }


def sampling_decoding_cfg(decoding_cfg: dict, temperature: float) -> dict:
    """The sample passes' decoding config: the greedy one, switched to sampling.

    Only `do_sample` and `temperature` differ from the main answer's config, so the
    sample length and every other generation parameter stay matched.
    """
    return {**decoding_cfg, "do_sample": True, "temperature": temperature}


def score_example(
    model,
    tokenizer,
    model_cfg: dict,
    question: str,
    decoding_cfg: dict,
    sample_cfg: dict,
    n_samples: int,
    contradiction_scorer,
) -> dict:
    """One example: main answer, samples, inconsistency score, and the latency of each."""
    inputs = build_generation_inputs(tokenizer, question, model_cfg).to(model.device)

    main = generate_with_latency(model, tokenizer, inputs, decoding_cfg)

    sampling_ms = 0.0
    samples = []
    for _ in range(n_samples):
        sampled = generate_with_latency(model, tokenizer, inputs, sample_cfg)
        samples.append(sampled["text"])
        sampling_ms += sampled["total_generation_ms"]

    sentences = split_into_atomic_claims(main["text"])
    score = selfcheck_inconsistency(sentences, samples, contradiction_scorer)

    return {
        "main_text": main["text"],
        "main_ttft_ms": main["ttft_ms"],
        "main_total_ms": main["total_generation_ms"],
        "sampling_total_ms": sampling_ms,
        "selfcheck": score,
    }


def run_for_model(
    config: dict, model_name: str, examples, splits, nli_model, nli_tokenizer, bootstrap_cfg: dict
) -> dict:
    model_cfg = config["models"][model_name]
    selfcheck_cfg = config["selfcheck"]
    decoding_cfg = config["gate"]["decoding"]
    quantile = config["gate"]["quantile"]
    sample_cfg = sampling_decoding_cfg(decoding_cfg, selfcheck_cfg["temperature"])
    n_samples = selfcheck_cfg["n_samples"]
    eval_split = selfcheck_cfg["eval_split"]

    view = splits.for_checkpoint(model_name)
    calibration_indices = view.calibration[: selfcheck_cfg["n_calibration_examples"]]
    eval_indices = getattr(splits, eval_split)[: selfcheck_cfg["n_eval_examples"]]

    model, tokenizer = load_model(model_cfg)
    try:
        contradiction_scorer = make_nli_contradiction_scorer(nli_model, nli_tokenizer)

        # Calibration pass: inconsistency scores on this checkpoint's own calibration
        # split. Examples the scorer cannot score (an empty generation) are dropped
        # rather than coerced to 0.0, which would read as perfectly consistent and drag
        # the calibrated threshold downward.
        calibration_scores = []
        for index in calibration_indices:
            scored = score_example(
                model, tokenizer, model_cfg, examples[index].question,
                decoding_cfg, sample_cfg, n_samples, contradiction_scorer,
            )
            if scored["selfcheck"].score is not None:
                calibration_scores.append(scored["selfcheck"].score)

        if not calibration_scores:
            raise ValueError(
                f"{model_name}: no calibration example produced a scorable generation — "
                "cannot calibrate a flag threshold"
            )

        # Matched-budget threshold: same quantile as the entropy gate, so this detector
        # flags the same fraction of examples the gate routes.
        flag_gate = GatePolicy()
        flag_threshold = flag_gate.calibrate(
            calibration_entropies=calibration_scores,
            calibration_indices=calibration_indices[: len(calibration_scores)],
            splits=view,
            quantile=quantile,
            source=f"selfcheck-{model_name}",
        )

        per_example = []
        for index in eval_indices:
            example = examples[index]
            scored = score_example(
                model, tokenizer, model_cfg, example.question,
                decoding_cfg, sample_cfg, n_samples, contradiction_scorer,
            )
            score = scored["selfcheck"].score
            verdict = nli_verdict_short_answer(
                nli_model,
                nli_tokenizer,
                scored["main_text"],
                example.right_answer,
                example.hallucinated_answer,
                config["nli_judge"]["short_answer_entailment_threshold"],
                question=example.question,
            )
            per_example.append(
                {
                    "index": index,
                    # Unscorable examples are not flagged, and are reported as such
                    # rather than silently counted as consistent.
                    "flagged": (score is not None and flag_gate.decide(score)),
                    "scorable": score is not None,
                    "selfcheck_inconsistency": score,
                    "factuality": verdict.label,
                    "main_total_ms": scored["main_total_ms"],
                    "main_ttft_ms": scored["main_ttft_ms"],
                    "sampling_total_ms": scored["sampling_total_ms"],
                }
            )

        n = len(per_example)
        scorable = [row for row in per_example if row["scorable"]]
        verdicts = [FactualityVerdict(label=row["factuality"]) for row in per_example]
        # Annotate-only, matching SENSE's merge-back policy, so nothing abstains and the
        # comparison is like-for-like.
        factuality_report = summarize_factuality(verdicts, n_abstained=0)

        # Detection metrics over the scorable subset, so the flag rate and the
        # unscorable count can't be conflated. `entropy` carries the inconsistency
        # score here — the field name is the gate's, the signal is this detector's,
        # which is what makes the two directly comparable.
        outcomes = [
            RoutingOutcome(
                entropy=row["selfcheck_inconsistency"], routed=row["flagged"], factuality=row["factuality"]
            )
            for row in scorable
        ]

        return {
            "baseline": "selfcheckgpt_nli",
            "dataset": "halueval",
            "model_name": model_name,
            "hf_repo": model_cfg["hf_repo"],
            "revision": model_cfg["revision"],
            "eval_split": eval_split,
            "n_eval_examples": n,
            "n_scorable": len(scorable),
            "n_unscorable": n - len(scorable),
            "n_calibration_examples": len(calibration_scores),
            "n_samples_per_example": n_samples,
            "sample_temperature": selfcheck_cfg["temperature"],
            "quantile": quantile,
            "flag_threshold": flag_threshold,
            "decoding": decoding_cfg,
            "sample_decoding": sample_cfg,
            "bootstrap": bootstrap_cfg,
            "factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
                hf_repo=config["nli_judge"]["hf_repo"], revision=config["nli_judge"]["revision"]
            ),
            "flag_rate": (sum(row["flagged"] for row in per_example) / n) if n else None,
            "task_accuracy": factuality_report["task_accuracy"],
            "hallucination_rate": factuality_report["hallucination_rate"],
            "abstention_rate": factuality_report["abstention_rate"],
            "factuality_report": factuality_report,
            # Comparable, at matched flag rate, with RQ1/RQ2's routing_quality and
            # RQ3's routing_detection.
            "detection": detection_metrics_with_ci(
                outcomes,
                n_resamples=bootstrap_cfg["n_resamples"],
                confidence=bootstrap_cfg["confidence"],
                seed=derived_seed(bootstrap_cfg["seed"], model_name, "selfcheck", "detection"),
            ),
            # The cost side of the comparison: what post-hoc verification charges per
            # example, against the gate's near-free logarithm.
            "mean_main_total_ms": sum(row["main_total_ms"] for row in per_example) / n if n else None,
            "mean_sampling_total_ms": sum(row["sampling_total_ms"] for row in per_example) / n if n else None,
            "mean_posthoc_overhead_ratio": (
                sum(row["sampling_total_ms"] for row in per_example)
                / sum(row["main_total_ms"] for row in per_example)
                if n and sum(row["main_total_ms"] for row in per_example) > 0
                else None
            ),
            "per_example": per_example,
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run() -> list[dict]:
    seed_everything()
    config = load_config()
    examples, splits = load_halueval_examples_and_splits()
    nli_model, nli_tokenizer = load_nli_model(
        config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"]
    )
    bootstrap_cfg = bootstrap_config()

    results = []
    for model_name in config["selfcheck"]["models"]:
        result = run_for_model(
            config, model_name, examples, splits, nli_model, nli_tokenizer, bootstrap_cfg
        )
        write_results(
            f"selfcheck_baseline_halueval_{model_name}.json",
            result,
            print_exclude_keys=frozenset({"per_example"}),
        )
        results.append(result)
    return results


if __name__ == "__main__":
    run()
