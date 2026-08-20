import pytest

from sense_eval.factuality import (
    FactualityVerdict,
    IncompleteFactualityReportError,
    assert_factuality_metrics_reported_together,
    summarize_factuality,
)


def test_summarize_factuality_computes_all_three_together():
    verdicts = [
        FactualityVerdict(label="correct"),
        FactualityVerdict(label="correct"),
        FactualityVerdict(label="incorrect"),
        FactualityVerdict(label="unknown"),
    ]

    report = summarize_factuality(verdicts, n_abstained=1)

    assert report["n_examples"] == 5  # 4 verdicts + 1 abstained
    assert report["n_abstained"] == 1
    assert report["n_correct"] == 2
    assert report["n_incorrect"] == 1
    assert report["n_unknown"] == 1
    assert report["task_accuracy"] == pytest.approx(2 / 5)
    assert report["hallucination_rate"] == pytest.approx(1 / 5)
    assert report["abstention_rate"] == pytest.approx(1 / 5)


def test_summarize_factuality_denominator_includes_abstained_examples():
    # An all-abstain policy must not show hallucination_rate == 0 via an empty
    # verdicts list dividing by a shrunk denominator — n_examples has to include
    # the abstentions, or abstaining would trivially "solve" hallucination.
    report = summarize_factuality([], n_abstained=10)

    assert report["n_examples"] == 10
    assert report["hallucination_rate"] == 0.0
    assert report["abstention_rate"] == 1.0
    assert report["task_accuracy"] == 0.0


def test_summarize_factuality_handles_zero_examples():
    report = summarize_factuality([], n_abstained=0)

    assert report["n_examples"] == 0
    assert report["task_accuracy"] is None
    assert report["hallucination_rate"] is None
    assert report["abstention_rate"] is None


def test_assert_factuality_metrics_reported_together_passes_when_all_present():
    assert_factuality_metrics_reported_together(
        {"task_accuracy": 0.5, "hallucination_rate": 0.1, "abstention_rate": 0.0}
    )  # must not raise


def test_assert_factuality_metrics_reported_together_passes_when_none_present():
    assert_factuality_metrics_reported_together({"some_other_key": 1})  # must not raise


def test_assert_factuality_metrics_reported_together_raises_on_partial_subset():
    with pytest.raises(IncompleteFactualityReportError, match="hallucination_rate"):
        assert_factuality_metrics_reported_together({"task_accuracy": 0.5})


def test_assert_factuality_metrics_reported_together_raises_when_only_hallucination_rate_present():
    # The specific misleading case CLAUDE.md calls out: hallucination rate reported
    # alone, with no accuracy or abstention figure next to it.
    with pytest.raises(IncompleteFactualityReportError):
        assert_factuality_metrics_reported_together({"hallucination_rate": 0.02})


def test_assert_factuality_metrics_reported_together_passes_when_prefixed_triple_present():
    # FActScore's factscore_symbolic_verification.py reports a SECOND, independent
    # triple under factscore_-prefixed keys alongside the unprefixed one.
    assert_factuality_metrics_reported_together(
        {
            "task_accuracy": 0.5,
            "hallucination_rate": 0.1,
            "abstention_rate": 0.0,
            "factscore_task_accuracy": 0.7,
            "factscore_hallucination_rate": 0.2,
            "factscore_abstention_rate": 0.1,
        }
    )  # must not raise


def test_assert_factuality_metrics_reported_together_raises_on_prefixed_partial_subset():
    with pytest.raises(IncompleteFactualityReportError, match="factscore_hallucination_rate"):
        assert_factuality_metrics_reported_together(
            {
                "task_accuracy": 0.5,
                "hallucination_rate": 0.1,
                "abstention_rate": 0.0,
                "factscore_task_accuracy": 0.7,
            }
        )


def test_assert_factuality_metrics_reported_together_passes_when_extracted_triple_present():
    result = {
        "task_accuracy": 1.0,
        "hallucination_rate": 0.0,
        "abstention_rate": 0.0,
        "extracted_task_accuracy": 0.5,
        "extracted_hallucination_rate": 0.5,
        "extracted_abstention_rate": 0.0,
    }
    assert_factuality_metrics_reported_together(result)  # must not raise


def test_assert_factuality_metrics_reported_together_raises_on_extracted_partial_subset():
    result = {
        "task_accuracy": 1.0,
        "hallucination_rate": 0.0,
        "abstention_rate": 0.0,
        "extracted_task_accuracy": 0.5,
        "extracted_hallucination_rate": 0.5,
        # extracted_abstention_rate missing
    }
    with pytest.raises(IncompleteFactualityReportError):
        assert_factuality_metrics_reported_together(result)
