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
Output: one line per example to stdout; a summary at the end.
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
        e_right = entailment_scores(nli_model, nli_tokenizer, text, example.right_answer)["entailment"]
        e_wrong = entailment_scores(nli_model, nli_tokenizer, text, example.hallucinated_answer)["entailment"]
        # Lexical hint only, to tell "the model said the right thing but the judge missed
        # it" from "the model said something else". Not used as a label anywhere.
        contains_right = example.right_answer.strip().lower() in text.lower()
        rows.append((e_right, e_wrong, contains_right, len(text.split())))
        print(
            f"{index} R={e_right:.2f} W={e_wrong:.2f} has_right={int(contains_right)} "
            f"gen={text[:70]!r} right={example.right_answer[:25]!r} wrong={example.hallucinated_answer[:25]!r}"
        )

    n = len(rows)
    right_clears = sum(r[0] >= threshold for r in rows)
    wrong_clears = sum(r[1] >= threshold for r in rows)
    print("--- summary ---")
    print(f"n={n} right_clears={right_clears} wrong_clears={wrong_clears} "
          f"neither={sum(r[0] < threshold and r[1] < threshold for r in rows)}")
    print(f"median R={statistics.median(r[0] for r in rows):.3f} median W={statistics.median(r[1] for r in rows):.3f} "
          f"max R={max(r[0] for r in rows):.3f}")
    print(f"has_right={sum(r[2] for r in rows)} median_words={statistics.median(r[3] for r in rows)}")
    # What does the model produce when the right answer IS literally in the text?
    contained = [r for r in rows if r[2]]
    if contained:
        print(f"when has_right: median R={statistics.median(r[0] for r in contained):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "mistral", int(sys.argv[2]) if len(sys.argv) > 2 else 30)
