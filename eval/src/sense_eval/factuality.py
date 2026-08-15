"""Factuality metrics: a placeholder verdict scorer, and the reporting layer that
turns per-example verdicts into the three numbers CLAUDE.md/the proposal require to
always be reported together — task accuracy, hallucination rate, and abstention
rate. Reporting hallucination rate alone is misleading: abstaining trivially drives
it toward zero at the cost of accuracy, so a hallucination-reduction number with no
accuracy or abstention figure next to it hides exactly the trade-off this project
studies.
"""

from dataclasses import dataclass

# No fine-tuned judge model or LLM-as-judge API is wired into this project yet — that
# is real, unfinished work, not a design choice, so lexical_containment_verdict below
# stays the only scorer for now. Every result produced with it MUST carry this label
# verbatim in its "factuality_metric" field (write_results checks for the
# "placeholder" substring and prints a loud warning if it's missing) so a
# pipeline-mechanics number can never quietly read as a paper-grade one.
PLACEHOLDER_FACTUALITY_METRIC_LABEL = "lexical_containment_placeholder (NOT a judge — see eval/README.md)"


@dataclass(frozen=True)
class FactualityVerdict:
    label: str  # "correct" | "incorrect" | "unknown"


def lexical_containment_verdict(
    generated_text: str,
    best_answer: str,
    correct_answers: tuple[str, ...],
    incorrect_answers: tuple[str, ...],
) -> FactualityVerdict:
    """"correct" if the generated text contains the best/any correct answer,
    "incorrect" if it contains a known incorrect answer (and no correct one),
    "unknown" otherwise — the generated text didn't clearly match either list.

    This is a simplified lexical-containment placeholder, NOT the project's
    eventual factuality judge — real TruthfulQA/HaluEval evaluation typically uses
    a fine-tuned judge model or human annotation, which is out of scope for
    exercising the pipeline on a tiny CPU model. Any number this function's
    verdicts roll up into measures whether the harness plumbing works end-to-end,
    not paper-grade accuracy. Replace before any number derived from it is
    reported as a result (see PLACEHOLDER_FACTUALITY_METRIC_LABEL above).
    """
    text = generated_text.lower()

    correct_candidates = [a for a in (best_answer, *correct_answers) if a]
    if any(candidate.lower() in text for candidate in correct_candidates):
        return FactualityVerdict(label="correct")

    incorrect_candidates = [a for a in incorrect_answers if a]
    if any(candidate.lower() in text for candidate in incorrect_candidates):
        return FactualityVerdict(label="incorrect")

    return FactualityVerdict(label="unknown")


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
