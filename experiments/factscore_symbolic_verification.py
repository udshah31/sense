"""FActScore symbolic-verification pipeline check: exercises the decomposition
front-end -> Z3 backend round-trip (CLAUDE.md's design decisions) against
FActScore's biography-prompt entities.

Not an RQ1-RQ3 harness. This project has no free-text-to-triple extractor yet
(documented in configs/rq3.yaml's symbolic_probe: out of scope, tied to the
undecided merge-back "replace" policy) — so, like RQ3's symbolic round-trip, this
does not extract claims from the generated biography and does not read the
generated text as a factuality signal. Instead, for each FActScore entity that
resolves against the Z3 backend's fixed domain KB
(services/symbolic/src/sense_symbolic/domain.py), it builds a *known ground-truth*
probe claim from that KB (e.g. "Albert Einstein" born "1879"), decomposes it, and
checks that the Z3 backend correctly verifies its own fact as true. Entities that
don't resolve against the small fixed KB (most of FActScore's 500, by design —
CLAUDE.md's design decisions call for containing the symbolic backend's scope to a
fixed constraint set) are recorded as abstained, not guessed at.

This validates decomposition+Z3 pipeline mechanics end-to-end on real FActScore
data, not the factual accuracy of the generated biography — task_accuracy here
means "fraction of examples whose entity resolved and whose ground-truth probe
claim the backend verified correctly," not "fraction of generations that were
non-hallucinatory." Still reports task_accuracy/hallucination_rate/abstention_rate
together (write_results enforces this), per CLAUDE.md's reporting requirement.

The biography itself is still generated (from `factscore_prompt`, same decoding
config as every other harness) and included per-example for inspection, exercising
the CPU-testable generation path this pipeline will eventually need for a real
extractor — it just isn't scored.
"""

from _common import build_generation_inputs, load_factscore_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_symbolic.decomposition import decompose_claim
from sense_symbolic.domain import facts_for, resolve_entity
from sense_symbolic.z3_verifier import verify_claim


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "factscore_symbolic": load_yaml_config("factscore_symbolic.yaml"),
    }


def build_probe_claim(entity_key: str) -> tuple[str, str]:
    """A (predicate_pid, object_label) pair built from the fixed domain KB's own
    ground-truth facts for `entity_key` — birth_year preferred (covers both the
    biographical entries and the two founding-year entries), nationality as the
    fallback. Raises KeyError if `entity_key` isn't in the KB (facts_for's
    contract) and ValueError if it's in the KB but has neither fact recorded.
    """
    facts = facts_for(entity_key)
    if facts.birth_year is not None:
        return "P569", str(facts.birth_year)
    if facts.nationality is not None:
        return "P27", facts.nationality
    raise ValueError(f"entity {entity_key!r} has no probe-able fact (birth_year or nationality) in the fixed domain KB")


async def run_experiment(config: dict, examples=None, eval_indices=None) -> dict:
    model_name = config["factscore_symbolic"]["model"]
    model_cfg = config["models"][model_name]
    decoding_cfg = config["factscore_symbolic"]["decoding"]

    if examples is None or eval_indices is None:
        examples, splits = load_factscore_examples_and_splits()
        all_eval_indices = getattr(splits, config["factscore_symbolic"]["eval_split"])
        n_requested = config["factscore_symbolic"]["n_eval_examples"]
        eval_indices = all_eval_indices[:n_requested]

    model, tokenizer = load_model(model_cfg)

    per_example = []
    for index_i in eval_indices:
        example = examples[index_i]

        inputs = build_generation_inputs(tokenizer, example.factscore_prompt, model_cfg).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=decoding_cfg["max_new_tokens"],
            do_sample=decoding_cfg["do_sample"],
        )
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        resolved_key = resolve_entity(example.entity)
        if resolved_key is None:
            per_example.append(
                {
                    "index": index_i,
                    "entity": example.entity,
                    "generated_text": generated_text,
                    "resolved": False,
                    "factuality": None,
                }
            )
            continue

        predicate_pid, probe_object_label = build_probe_claim(resolved_key)
        claim = decompose_claim(example.entity, predicate_pid, probe_object_label)
        verified = verify_claim(claim)
        label = "correct" if verified is True else "incorrect" if verified is False else "unknown"

        per_example.append(
            {
                "index": index_i,
                "entity": example.entity,
                "generated_text": generated_text,
                "resolved": True,
                "predicate_pid": predicate_pid,
                "probe_object_label": probe_object_label,
                "factuality": label,
            }
        )

    resolved_records = [r for r in per_example if r["resolved"]]
    n_abstained = len(per_example) - len(resolved_records)
    verdicts = [FactualityVerdict(label=r["factuality"]) for r in resolved_records]
    factuality_report = summarize_factuality(verdicts, n_abstained=n_abstained)

    return {
        "research_question": "FActScore symbolic-verification pipeline check (not RQ1-RQ3)",
        "dataset": "factscore",
        "model_name": model_name,
        "hf_repo": model_cfg["hf_repo"],
        "revision": model_cfg["revision"],
        "n_eval_examples": len(per_example),
        "decoding": decoding_cfg,
        "task_accuracy": factuality_report["task_accuracy"],
        "hallucination_rate": factuality_report["hallucination_rate"],
        "abstention_rate": factuality_report["abstention_rate"],
        "factuality_report": factuality_report,
        "factuality_metric": (
            "decomposition_z3_roundtrip_against_known_probe_claim "
            "(NOT a factuality judgment of the generated biography — see module docstring)"
        ),
        "per_example": per_example,
    }


async def main() -> dict:
    config = load_config()
    result = await run_experiment(config)

    write_results(
        f"factscore_symbolic_verification_{config['factscore_symbolic']['model']}.json",
        result,
        print_exclude_keys=frozenset({"per_example"}),
    )
    return result


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
