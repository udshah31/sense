"""Routing quality: does the entropy gate fire on the examples the model gets wrong?

This module exists because "routing quality" — the phrase the proposal uses for the
primary evidence behind RQ1 and RQ2 — was previously operationalized two different
ways, neither of which measured quality:

  - RQ1 measured `transfer_agreement_rate`, the fraction of examples on which a
    transferred gate and a natively-calibrated gate make the same decision. That is
    *concordance*. It reads 1.0 when both gates route the same examples, whether or
    not those were the right examples to route.
  - RQ2 measured calibration fidelity, |observed_routing_rate - (1 - quantile)|.
    That measures whether a threshold hits its own target firing rate. A gate that
    fires on exactly (1 - quantile) of examples chosen at random scores perfectly.

Neither relates a routing decision to a factuality outcome, so both can look healthy
while providing no evidence that routing catches hallucinations.

The gate's actual job is a detection problem: fire on examples that would otherwise be
hallucinated, stay quiet on examples that would be answered correctly. Labels for that
are already available — HaluEval ships a right_answer/hallucinated_answer pair per item
and sense_eval.nli_judge turns a generation into a correct/incorrect/unknown verdict —
so routing quality is measured here as detection performance of the gate against the
*ungated* model's own correctness.

Two complementary numbers are reported, and the distinction between them is the point:

  - Threshold-dependent (precision / recall / F1): how good is *this particular
    threshold* on *this* model. This is what changes when a threshold transfers badly.
  - Threshold-free (AUROC over the raw entropy score): how informative is the entropy
    *signal* about hallucination on this model at all, independent of where the cutoff
    sits. A transferred threshold can score badly on precision/recall while AUROC stays
    high — that is the signature of a mis-placed cutoff over a still-informative signal,
    which is a different finding from the signal itself failing to transfer.

Conventions:
  - Positive class is "incorrect" (a hallucination the gate should have caught).
  - "unknown" verdicts are excluded from every detection metric and counted
    separately. They carry no ground truth to score against, and silently folding
    them into either class would bias the result in a direction chosen by accident.
  - Routing rate is reported over *all* outcomes including unknowns, because it is a
    cost figure (it drives RQ3's latency overhead) rather than a detection figure.
"""

from dataclasses import dataclass

CORRECT = "correct"
INCORRECT = "incorrect"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class RoutingOutcome:
    """One evaluation example's gate decision paired with how the ungated generation
    actually scored.

    `entropy` is the scalar the gate decided on, kept so threshold-free metrics can be
    computed over the same values the gate saw. `factuality` is a
    sense_eval.factuality.FactualityVerdict label.
    """

    entropy: float
    routed: bool
    factuality: str


class UnknownFactualityLabelError(ValueError):
    """Raised when an outcome carries a factuality label this module doesn't know.

    Loud rather than lenient: a typo'd or newly-added label silently dropped from the
    denominator would quietly change every rate this module reports.
    """


def _validate(outcomes: list[RoutingOutcome]) -> None:
    allowed = {CORRECT, INCORRECT, UNKNOWN}
    offending = sorted({o.factuality for o in outcomes} - allowed)
    if offending:
        raise UnknownFactualityLabelError(
            f"unrecognized factuality labels {offending} — expected one of {sorted(allowed)}"
        )


def _auroc(positive_scores: list[float], negative_scores: list[float]) -> float | None:
    """AUROC via the rank-sum (Mann-Whitney U) identity, with midranks for ties.

    Equivalent to the probability that a randomly chosen positive example scores above
    a randomly chosen negative one, with ties counting a half. Implemented directly
    rather than pulled from sklearn to keep this package dependency-light, matching
    GatePolicy's hand-rolled quantile.

    Returns None when either class is empty — AUROC is undefined there, and returning
    0.5 or 0.0 would be a fabricated number.
    """
    n_pos, n_neg = len(positive_scores), len(negative_scores)
    if n_pos == 0 or n_neg == 0:
        return None

    combined = sorted(
        [(s, 1) for s in positive_scores] + [(s, 0) for s in negative_scores],
        key=lambda pair: pair[0],
    )

    # Midranks: every member of a tied block receives the mean of the ranks that block
    # spans, so ties contribute exactly 0.5 rather than depending on sort order.
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[k] = midrank
        i = j + 1

    positive_rank_sum = sum(rank for rank, (_, label) in zip(ranks, combined) if label == 1)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def routing_detection_metrics(outcomes: list[RoutingOutcome]) -> dict:
    """Detection performance of the gate against the ungated model's own correctness.

    Positive class is "incorrect". Keys:

      routing_rate                 — fraction of ALL outcomes routed (cost figure)
      n_outcomes                   — every outcome, unknowns included
      n_scored                     — outcomes with a correct/incorrect label
      n_unknown_excluded           — outcomes dropped from detection metrics
      n_true_positive              — routed and incorrect (a catch)
      n_false_positive             — routed and correct (wasted symbolic call)
      n_false_negative             — not routed and incorrect (a miss)
      n_true_negative              — not routed and correct
      routed_hallucination_precision — of routed examples, fraction actually incorrect
      hallucination_recall         — of incorrect examples, fraction routed
      f1                           — harmonic mean of the two
      specificity                  — of correct examples, fraction NOT routed
      balanced_accuracy            — mean of recall and specificity
      entropy_auroc                — threshold-free; see module docstring

    Rate keys are None, never 0.0, where the denominator is empty. A precision of 0.0
    on zero routed examples would be a fabricated measurement.
    """
    _validate(outcomes)

    n_outcomes = len(outcomes)
    scored = [o for o in outcomes if o.factuality != UNKNOWN]

    tp = sum(1 for o in scored if o.routed and o.factuality == INCORRECT)
    fp = sum(1 for o in scored if o.routed and o.factuality == CORRECT)
    fn = sum(1 for o in scored if not o.routed and o.factuality == INCORRECT)
    tn = sum(1 for o in scored if not o.routed and o.factuality == CORRECT)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    specificity = tn / (tn + fp) if (tn + fp) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2.0 if recall is not None and specificity is not None else None
    )

    return {
        "routing_rate": (sum(1 for o in outcomes if o.routed) / n_outcomes) if n_outcomes else None,
        "n_outcomes": n_outcomes,
        "n_scored": len(scored),
        "n_unknown_excluded": n_outcomes - len(scored),
        "n_true_positive": tp,
        "n_false_positive": fp,
        "n_false_negative": fn,
        "n_true_negative": tn,
        "routed_hallucination_precision": precision,
        "hallucination_recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "entropy_auroc": _auroc(
            [o.entropy for o in scored if o.factuality == INCORRECT],
            [o.entropy for o in scored if o.factuality == CORRECT],
        ),
    }


def routing_quality_delta(transferred: dict, native: dict) -> dict:
    """native minus transferred, for the metrics where a difference is meaningful.

    This is the quantity the proposal calls "the difference in routing quality between
    the transferred fixed threshold and the self-adaptive gate operating natively on
    each model" — the primary evidence for RQ1 and RQ2. A positive delta means the
    native gate did better, i.e. the transferred threshold cost something.

    `entropy_auroc` is deliberately NOT differenced: it is threshold-free, so both
    gates see the same score distribution on the same model and the delta is always
    zero by construction. It belongs in each arm's metrics as a signal-quality figure,
    not in the comparison.

    A metric is None in the output when it is missing on either side, rather than
    being treated as zero.
    """
    differenced = (
        "routed_hallucination_precision",
        "hallucination_recall",
        "f1",
        "specificity",
        "balanced_accuracy",
        "routing_rate",
    )
    return {
        f"delta_{key}": (
            native[key] - transferred[key]
            if native.get(key) is not None and transferred.get(key) is not None
            else None
        )
        for key in differenced
    }
