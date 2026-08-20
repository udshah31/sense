import pytest

from factscore_symbolic_verification import build_probe_claim, run_experiment
from sense_data.factscore import FActScoreExample

TINY_GPT2 = {"hf_repo": "sshleifer/tiny-gpt2", "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"}
NLI_MODEL_CFG = {
    "hf_repo": "cliang1453/deberta-v3-xsmall-mnli",
    "revision": "d1ca70f9ece4d8afd33015893a69df9a6e45a672",
    "claim_supported_threshold": 0.5,
    "fraction_correct_threshold": 0.8,
    "fraction_incorrect_threshold": 0.2,
}


def test_build_probe_claim_uses_birth_year_when_known():
    predicate_pid, object_label = build_probe_claim("albert einstein")

    assert predicate_pid == "P569"
    assert object_label == "1879"


def test_build_probe_claim_falls_back_to_nationality_without_birth_year():
    with pytest.raises(KeyError):
        build_probe_claim("not a real entity")


@pytest.mark.asyncio
async def test_run_experiment_verifies_known_entity_as_correct():
    examples = [
        FActScoreExample(
            index=0,
            entity="Albert Einstein",
            one_fact_prompt="Tell me a fact about Albert Einstein.",
            factscore_prompt="Tell me about Albert Einstein.",
            hundredw_prompt="Write 100 words about Albert Einstein.",
            around_100="",
            wikipedia_text="Albert Einstein was a theoretical physicist.",
        ),
        FActScoreExample(
            index=1,
            entity="Some Unresolvable Person",
            one_fact_prompt="Tell me a fact about Some Unresolvable Person.",
            factscore_prompt="Tell me about Some Unresolvable Person.",
            hundredw_prompt="Write 100 words about Some Unresolvable Person.",
            around_100="",
            wikipedia_text="Some Unresolvable Person was a fictional test fixture.",
        ),
    ]

    config = {
        "models": {"cpu_test": TINY_GPT2},
        "factscore_symbolic": {
            "model": "cpu_test",
            "decoding": {"do_sample": False, "max_new_tokens": 5},
        },
        "nli_judge": NLI_MODEL_CFG,
    }

    result = await run_experiment(config, examples, list(range(len(examples))))

    assert result["dataset"] == "factscore"
    assert result["n_eval_examples"] == 2
    # Z3 round-trip numbers (against known entities only) — unchanged behavior.
    assert result["factuality_report"]["n_correct"] == 1
    assert result["factuality_report"]["n_abstained"] == 1
    assert result["task_accuracy"] == 0.5
    assert result["hallucination_rate"] == 0.0
    assert result["abstention_rate"] == 0.5

    # NLI-judge numbers (against every example's own reference text) — new.
    assert "factscore_task_accuracy" in result
    assert "factscore_hallucination_rate" in result
    assert "factscore_abstention_rate" in result
    assert result["factscore_factuality_report"]["n_examples"] == 2

    resolved_record = next(r for r in result["per_example"] if r["entity"] == "Albert Einstein")
    assert resolved_record["resolved"] is True
    assert resolved_record["factuality"] == "correct"
    assert "factscore_factuality" in resolved_record
    assert "factscore_supported_fraction" in resolved_record

    unresolved_record = next(r for r in result["per_example"] if r["entity"] == "Some Unresolvable Person")
    assert unresolved_record["resolved"] is False
    assert unresolved_record["factuality"] is None
    # NLI scoring runs regardless of Z3 entity resolution — it's independent.
    assert "factscore_factuality" in unresolved_record

    # Extracted-claims numbers (from the generated biography's own text) — new,
    # independent of both the Z3-probe-claim trio and the factscore_* NLI trio.
    assert "extracted_task_accuracy" in result
    assert "extracted_hallucination_rate" in result
    assert "extracted_abstention_rate" in result
    assert result["extracted_factuality_report"]["n_examples"] == 2
    assert "extracted_claims_count" in resolved_record
    assert "extracted_factuality" in resolved_record
    assert "extracted_claims_count" in unresolved_record
    assert "extracted_factuality" in unresolved_record


from factscore_symbolic_verification import extracted_claims_verdict


def test_extracted_claims_verdict_correct_when_claim_verifies_true():
    verdict, count = extracted_claims_verdict("Albert Einstein", "Albert Einstein was born in 1879.")
    assert verdict.label == "correct"
    assert count == 1


def test_extracted_claims_verdict_incorrect_when_claim_verifies_false():
    verdict, count = extracted_claims_verdict("Albert Einstein", "Albert Einstein was born in 1900.")
    assert verdict.label == "incorrect"
    assert count == 1


def test_extracted_claims_verdict_unknown_when_no_claims_extracted():
    verdict, count = extracted_claims_verdict("Albert Einstein", "Albert Einstein enjoyed music.")
    assert verdict.label == "unknown"
    assert count == 0


def test_extracted_claims_verdict_unknown_when_subject_unresolved():
    verdict, count = extracted_claims_verdict("Some Unresolvable Person", "Some Unresolvable Person was born in 1900.")
    assert verdict.label == "unknown"
    assert count == 1


def test_extracted_claims_verdict_prioritizes_incorrect_over_correct():
    # Two sentences: one true claim, one false claim about the same subject —
    # any False must make the whole verdict "incorrect".
    verdict, count = extracted_claims_verdict(
        "Albert Einstein", "Albert Einstein was born in 1879. Albert Einstein was born in 1900."
    )
    assert verdict.label == "incorrect"
    assert count == 2
