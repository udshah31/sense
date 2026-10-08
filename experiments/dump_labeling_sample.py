"""Dump real generations, with the judge's entailment scores, for hand labeling.

**Not a result.** The NLI threshold (nli_judge.yaml) was fit on canonical answer text as
a stand-in for model output, and the real outputs it must judge are 20-token hedged
fragments — so it needs checking against real generations with human-meaning labels.
This script produces the rows to label; the labels are added afterwards, outside this
file, so nothing here can see them.

Rows are drawn only from the calibration pool (calibrate_nli_short_answer_threshold.py's
union of the five calibration splits, never development or test), using the same
deterministic 80/20 fit/holdout split, so a threshold chosen on the `fit` rows can be
checked on the untouched `holdout` rows. Scores are the question-prefixed entailment
of the right and the hallucinated answer — independent of any verdict rule, so the
rule and threshold can be chosen afterwards from the labels.

Usage (from experiments/):  uv run python dump_labeling_sample.py [n_fit] [n_holdout] [model_key ...]
Writes results/judge_label_sample_<model>.json.
"""

import gc
import random
import sys

import torch

from _common import (
    example_signal,
    load_halueval_examples_and_splits,
    load_model,
    load_model_registry,
    load_yaml_config,
    seed_everything,
    write_results,
)
from calibrate_nli_short_answer_threshold import build_calibration_pool, split_fit_holdout
from sense_eval.nli_judge import entailment_scores, load_nli_model, short_answer_nli_pair

DEFAULT_MODELS = ["mistral", "qwen3_1_7b"]  # hedged-fragment style vs. terse style


def sample_rows(fit: list[int], holdout: list[int], n_fit: int, n_holdout: int, model_key: str) -> list[tuple[int, str]]:
    """Deterministic per model, so a re-run labels the same questions."""
    rng = random.Random(f"judge-label-sample:{model_key}")
    return [(i, "fit") for i in sorted(rng.sample(fit, n_fit))] + [
        (i, "holdout") for i in sorted(rng.sample(holdout, n_holdout))
    ]


def main(n_fit: int, n_holdout: int, model_keys: list[str]) -> None:
    seed_everything()
    registry = load_model_registry()
    decoding_cfg = load_yaml_config("gate.yaml")["decoding"]
    nli_cfg = load_yaml_config("nli_judge.yaml")
    examples, splits = load_halueval_examples_and_splits()
    fit, holdout = split_fit_holdout(build_calibration_pool(splits))
    nli_model, nli_tokenizer = load_nli_model(nli_cfg["hf_repo"], nli_cfg["revision"])

    for model_key in model_keys:
        model_cfg = registry[model_key]
        model, tokenizer = load_model(model_cfg)
        try:
            rows = []
            for index, tier in sample_rows(fit, holdout, n_fit, n_holdout, model_key):
                example = examples[index]
                text = example_signal(model, tokenizer, example.question, decoding_cfg, model_cfg).generated_text
                score = lambda answer: entailment_scores(  # noqa: E731
                    nli_model, nli_tokenizer, *short_answer_nli_pair(example.question, text, answer)
                )["entailment"]
                rows.append(
                    {
                        "index": index,
                        "tier": tier,
                        "question": example.question,
                        "right_answer": example.right_answer,
                        "hallucinated_answer": example.hallucinated_answer,
                        "generated_text": text,
                        "right_entailment": score(example.right_answer),
                        "hallucinated_entailment": score(example.hallucinated_answer),
                    }
                )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_results(
            f"judge_label_sample_{model_key}.json",
            {
                "research_question": "judge labeling sample (not a result)",
                "model_name": model_key,
                "hf_repo": model_cfg["hf_repo"],
                "revision": model_cfg["revision"],
                "decoding": decoding_cfg,
                "nli_judge": {"hf_repo": nli_cfg["hf_repo"], "revision": nli_cfg["revision"]},
                "n_rows": len(rows),
                # The three-together rule applies to reported factuality; this file reports
                # none — it is raw material for labeling — so the keys are absent, not null.
                "rows": rows,
            },
            print_exclude_keys=frozenset({"rows"}),
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        int(args[0]) if args else 40,
        int(args[1]) if len(args) > 1 else 20,
        args[2:] or DEFAULT_MODELS,
    )
