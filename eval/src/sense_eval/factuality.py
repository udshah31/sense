"""Factuality metrics: the reporting layer that turns per-example verdicts into
the three numbers CLAUDE.md/the proposal require to always be reported together —
task accuracy, hallucination rate, and abstention rate. Reporting hallucination
rate alone is misleading: abstaining trivially drives it toward zero at the cost
of accuracy, so a hallucination-reduction number with no accuracy or abstention
figure next to it hides exactly the trade-off this project studies.

The verdict scorers that produce FactualityVerdict instances live in
sense_eval.nli_judge (an NLI-based semantic judge) — this module is agnostic to
which scorer produced them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FactualityVerdict:
    label: str  # "correct" | "incorrect" | "unknown"


# The three keys CLAUDE.md/the proposal require to always be reported together.
# assert_factuality_metrics_reported_together (called from every experiment's
# write_results) fails loudly if a result carries some but not all of them.
REQUIRED_TOGETHER_KEYS = frozenset({"task_accuracy", "hallucination_rate", "abstention_rate"})


class IncompleteFactualityReportError(ValueError):
    """Raised when a result reports some but not all of task_accuracy,
    hallucination_rate, and abstention_rate. Never fix this by dropping the
    offending keys — add the missing ones instead; the whole point of the check is
    that none of the three may be reported alone."""


def assert_factuality_metrics_reported_together(result: dict) -> None:
    present = REQUIRED_TOGETHER_KEYS & result.keys()
    if present and present != REQUIRED_TOGETHER_KEYS:
        missing = REQUIRED_TOGETHER_KEYS - present
        raise IncompleteFactualityReportError(
            f"result reports {sorted(present)} but is missing {sorted(missing)} — "
            "task_accuracy, hallucination_rate, and abstention_rate must always be "
            "reported together, never a subset (a hallucination-rate number alone "
            "is misleading: abstaining trivially drives it toward zero at the cost "
            "of accuracy)"
        )


def summarize_factuality(verdicts: list[FactualityVerdict], n_abstained: int) -> dict:
    """Roll up per-example verdicts (from generated, non-abstained examples) plus a
    count of abstained examples into the three required, always-together numbers.

    All three share one denominator: every example the harness attempted,
    including ones it abstained on. Computing hallucination_rate only over
    answered examples would let an abstain-heavy policy shrink its own
    denominator and hide the accuracy it gave up — exactly the trap CLAUDE.md
    warns about. n_abstained is a required, explicit argument (not defaulted to
    0) so a caller has to state whether its merge-back policy can abstain at all,
    rather than the reporting layer silently assuming it can't.
    """
    n_examples = len(verdicts) + n_abstained
    n_correct = sum(1 for v in verdicts if v.label == "correct")
    n_incorrect = sum(1 for v in verdicts if v.label == "incorrect")
    n_unknown = sum(1 for v in verdicts if v.label == "unknown")

    return {
        "n_examples": n_examples,
        "n_abstained": n_abstained,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_unknown": n_unknown,
        "task_accuracy": (n_correct / n_examples) if n_examples else None,
        "hallucination_rate": (n_incorrect / n_examples) if n_examples else None,
        "abstention_rate": (n_abstained / n_examples) if n_examples else None,
    }
