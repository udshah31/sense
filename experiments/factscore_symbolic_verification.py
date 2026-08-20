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

Separately, every example's generated biography IS now scored for real against
its own FActScore reference text (`example.wikipedia_text`), using the NLI
judge's atomic-decomposition-based verdict (`sense_eval.nli_judge.factscore_style_verdict`)
— reported under the `factscore_*`-prefixed keys, kept distinct from the
Z3-round-trip keys above so the two different things this script measures are
never confused with each other. This is this project's first real (non-placeholder)
FActScore factuality number, though its threshold calibration is a named,
unvalidated first cut (see `docs/superpowers/specs/2026-08-16-nli-judge-design.md`).

A third, independent measurement now also extracts claims from the generated
biography's own text (not a hand-built probe claim) via a rule-based extractor
(`sense_symbolic.extraction.extract_claims`) and verifies each through the same
Z3 backend as the round-trip check above — reported under the
`extracted_*`-prefixed keys. This is this project's first real symbolic
verification of the model's *own* generated claims (as opposed to a
known-ground-truth probe or an NLI entailment score), though the extractor
itself is a first cut: precision-biased fixed patterns over five relation
kinds, no negation handling, no coreference resolution (see
`docs/superpowers/specs/2026-08-20-freetext-claim-extraction-design.md`).

The biography itself is still generated (from `factscore_prompt`, same decoding
config as every other harness) and included per-example for inspection, exercising
the CPU-testable generation path this pipeline will eventually need for a real
extractor — and, as described above, it IS now scored against the reference text
via the NLI judge, under the `factscore_*`-prefixed keys.
"""

from _common import build_generation_inputs, load_factscore_examples_and_splits, load_model, load_model_registry, load_yaml_config, write_results
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, factscore_style_verdict, load_nli_model, split_into_atomic_claims
from sense_symbolic.decomposition import decompose_claim
from sense_symbolic.domain import facts_for, resolve_entity
from sense_symbolic.extraction import extract_claims
from sense_symbolic.z3_verifier import verify_claim


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "factscore_symbolic": load_yaml_config("factscore_symbolic.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
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


def extracted_claims_verdict(subject_label: str, generated_text: str) -> tuple[FactualityVerdict, int]:
    """Splits generated_text into sentences, extracts AtomicClaims from each
    via extract_claims (services/symbolic's rule-based extractor), and
    verifies each through the existing Z3 verify_claim. Aggregates: any
    verified-False claim makes the whole verdict "incorrect"; else any
    verified-True claim makes it "correct"; else "unknown" — including when
    zero claims were extracted, or every extracted claim's subject failed to
    resolve against the fixed KB. Returns the verdict plus the number of
    claims extracted (for per_example transparency, regardless of whether
    they verified) — see
    docs/superpowers/specs/2026-08-20-freetext-claim-extraction-design.md.
    """
    claims = [
        claim
        for sentence in split_into_atomic_claims(generated_text)
        for claim in extract_claims(subject_label, sentence)
    ]
    results = [verify_claim(claim) for claim in claims]

    if any(result is False for result in results):
        label = "incorrect"
    elif any(result is True for result in results):
        label = "correct"
    else:
        label = "unknown"

    return FactualityVerdict(label=label), len(claims)


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
    nli_model, nli_tokenizer = load_nli_model(config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"])

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

        factscore_verdict, factscore_detail = factscore_style_verdict(
            nli_model,
            nli_tokenizer,
            generated_text,
            example.wikipedia_text,
            config["nli_judge"]["claim_supported_threshold"],
            config["nli_judge"]["fraction_correct_threshold"],
            config["nli_judge"]["fraction_incorrect_threshold"],
        )

        extracted_verdict, extracted_claims_count = extracted_claims_verdict(example.entity, generated_text)

        resolved_key = resolve_entity(example.entity)
        if resolved_key is None:
            per_example.append(
                {
                    "index": index_i,
                    "entity": example.entity,
                    "generated_text": generated_text,
                    "resolved": False,
                    "factuality": None,
                    "factscore_factuality": factscore_verdict.label,
                    "factscore_supported_fraction": factscore_detail.supported_fraction,
                    "extracted_factuality": extracted_verdict.label,
                    "extracted_claims_count": extracted_claims_count,
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
                "factscore_factuality": factscore_verdict.label,
                "factscore_supported_fraction": factscore_detail.supported_fraction,
                "extracted_factuality": extracted_verdict.label,
                "extracted_claims_count": extracted_claims_count,
            }
        )

    resolved_records = [r for r in per_example if r["resolved"]]
    n_abstained = len(per_example) - len(resolved_records)
    verdicts = [FactualityVerdict(label=r["factuality"]) for r in resolved_records]
    factuality_report = summarize_factuality(verdicts, n_abstained=n_abstained)

    factscore_verdicts = [FactualityVerdict(label=r["factscore_factuality"]) for r in per_example]
    factscore_factuality_report = summarize_factuality(factscore_verdicts, n_abstained=0)

    extracted_verdicts = [FactualityVerdict(label=r["extracted_factuality"]) for r in per_example]
    extracted_factuality_report = summarize_factuality(extracted_verdicts, n_abstained=0)

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
        "factscore_task_accuracy": factscore_factuality_report["task_accuracy"],
        "factscore_hallucination_rate": factscore_factuality_report["hallucination_rate"],
        "factscore_abstention_rate": factscore_factuality_report["abstention_rate"],
        "factscore_factuality_report": factscore_factuality_report,
        "factscore_factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
            hf_repo=config["nli_judge"]["hf_repo"], revision=config["nli_judge"]["revision"]
        ),
        "extracted_task_accuracy": extracted_factuality_report["task_accuracy"],
        "extracted_hallucination_rate": extracted_factuality_report["hallucination_rate"],
        "extracted_abstention_rate": extracted_factuality_report["abstention_rate"],
        "extracted_factuality_report": extracted_factuality_report,
        "extracted_factuality_metric": (
            "z3_verified_claims_extracted_from_generated_text (rule-based extraction; "
            "see docs/superpowers/specs/2026-08-20-freetext-claim-extraction-design.md)"
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
