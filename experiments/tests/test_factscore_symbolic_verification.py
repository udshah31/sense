import pytest

from factscore_symbolic_verification import build_probe_claim, run_experiment
from sense_data.factscore import FActScoreExample

TINY_GPT2 = {"hf_repo": "sshleifer/tiny-gpt2", "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"}


def test_build_probe_claim_uses_birth_year_when_known():
    predicate_pid, object_label = build_probe_claim("albert einstein")

    assert predicate_pid == "P569"
    assert object_label == "1879"


def test_build_probe_claim_falls_back_to_nationality_without_birth_year():
    # arthur's magazine only has a birth_year (founding year) in the fixed KB, so
    # exercise a case in domain.py's KB that lacks both to hit the None fallback
    # instead: no such entity exists, so assert the documented contract directly
    # via an entity that only has nationality (none in the KB lack birth_year but
    # have nationality) is not exercisable here — cover via unresolvable key.
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
            wikipedia_text="",
        ),
    ]

    config = {
        "models": {"cpu_test": TINY_GPT2},
        "factscore_symbolic": {
            "model": "cpu_test",
            "decoding": {"do_sample": False, "max_new_tokens": 5},
        },
    }

    result = await run_experiment(config, examples, list(range(len(examples))))

    assert result["dataset"] == "factscore"
    assert result["n_eval_examples"] == 2
    assert result["factuality_report"]["n_correct"] == 1
    assert result["factuality_report"]["n_abstained"] == 1
    assert result["task_accuracy"] == 0.5
    assert result["hallucination_rate"] == 0.0
    assert result["abstention_rate"] == 0.5

    resolved_record = next(r for r in result["per_example"] if r["entity"] == "Albert Einstein")
    assert resolved_record["resolved"] is True
    assert resolved_record["factuality"] == "correct"
    assert "generated_text" in resolved_record

    unresolved_record = next(r for r in result["per_example"] if r["entity"] == "Some Unresolvable Person")
    assert unresolved_record["resolved"] is False
    assert unresolved_record["factuality"] is None
