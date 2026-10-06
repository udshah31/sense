"""Diagnostic: why does the NLI judge label most generations "unknown"?

**This is not a result.** It exists because the first real RQ1 pair (llama3 -> mistral,
1,000 development examples) had the judge label 946 of 1,000 target generations
"unknown", leaving 54 scored examples — too few for routing quality to mean anything.

`nli_verdict_short_answer` returns "unknown" when neither the right nor the
hallucinated answer is entailed at or above `short_answer_entailment_threshold`
(or both are). This script generates for a small sample with the same code path RQ1
uses, then prints the generation, both answers, and both entailment scores, so the
cause can be read off directly instead of guessed. It records only the first
`n_examples` of the **development** split (never test) and writes nothing to results/.

Usage (from experiments/):  uv run python diagnose_judge_unknowns.py [model_key] [n_examples]
Output: one line per example to stdout; per-formulation summary at the end.
"""

import statistics
import sys

from _common import (
    example_signal,
    load_halueval_examples_and_splits,
    load_model,
    load_model_registry,
    load_yaml_config,
    seed_everything,
)
from sense_eval.nli_judge import entailment_scores, load_nli_model


# Judge formulations compared on the SAME generations. `current` is what the harness uses.
# The others change only how premise and hypothesis are built; the threshold stays fixed
# at nli_judge.yaml's value so nothing is tuned on this sample (CLAUDE.md: thresholds are
# fit on a calibration split, never on the data being inspected).
VARIANTS = {
    "current": lambda q, gen, ans: (gen, ans),
    "q_prefixed": lambda q, gen, ans: (f"{q} {gen}", f"{q} {ans}"),
    "statement": lambda q, gen, ans: (gen, f'The answer to "{q}" is {ans}.'),
    "q_prefixed_statement": lambda q, gen, ans: (f"{q} {gen}", f'The answer to "{q}" is {ans}.'),
}


def main(model_key: str, n_examples: int) -> None:
    seed_everything()
    registry = load_model_registry()
    gate_cfg = load_yaml_config("gate.yaml")
    nli_cfg = load_yaml_config("nli_judge.yaml")
    threshold = nli_cfg["short_answer_entailment_threshold"]

    examples, splits = load_halueval_examples_and_splits()
    indices = splits.development[:n_examples]

    model_cfg = registry[model_key]
    model, tokenizer = load_model(model_cfg)
    nli_model, nli_tokenizer = load_nli_model(nli_cfg["hf_repo"], nli_cfg["revision"])
    print(f"model={model_key} threshold={threshold} max_new_tokens={gate_cfg['decoding']['max_new_tokens']}")

    rows = []
    for index in indices:
        example = examples[index]
        text = example_signal(model, tokenizer, example.question, gate_cfg["decoding"], model_cfg).generated_text
        contains_right = example.right_answer.strip().lower() in text.lower()
        scores = {}
        for name, build in VARIANTS.items():
            pr, hr = build(example.question, text, example.right_answer)
            pw, hw = build(example.question, text, example.hallucinated_answer)
            scores[name] = (
                entailment_scores(nli_model, nli_tokenizer, pr, hr)["entailment"],
                entailment_scores(nli_model, nli_tokenizer, pw, hw)["entailment"],
            )
        rows.append((scores, contains_right))
        print(f"{index} has_right={int(contains_right)} gen={text[:60]!r}")

    n = len(rows)
    print("--- summary (threshold fixed at", threshold, ") ---")
    for name in VARIANTS:
        r = [x[0][name][0] for x in rows]
        w = [x[0][name][1] for x in rows]
        r_has = [x[0][name][0] for x in rows if x[1]]
        r_not = [x[0][name][0] for x in rows if not x[1]]
        verdicts = {"correct": 0, "incorrect": 0, "unknown": 0}
        for a_, b_ in zip(r, w):
            rc, wc = a_ >= threshold, b_ >= threshold
            verdicts["correct" if rc and not wc else "incorrect" if wc and not rc else "unknown"] += 1
        sep = (statistics.median(r_has) - statistics.median(r_not)) if r_has and r_not else float("nan")
        print(
            f"{name}: n={n} verdicts={verdicts} medR={statistics.median(r):.3f} medW={statistics.median(w):.3f} "
            f"medR(has_right)={statistics.median(r_has) if r_has else float('nan'):.3f} "
            f"medR(not)={statistics.median(r_not) if r_not else float('nan'):.3f} sep={sep:.3f}"
        )
    print(f"has_right_total={sum(x[1] for x in rows)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mistral", int(sys.argv[2]) if len(sys.argv) > 2 else 30)
