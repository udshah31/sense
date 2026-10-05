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

    Rate keys are None, never 0.0, where the denominator is empty: a precision of 0.0
    on zero routed examples would be a fabricated measurement. A rate of 0.0 on a
    non-empty denominator is a real measurement and is reported as such — so f1 is 0.0
    when precision and recall are both genuinely zero, and None only when one of them
    could not be measured at all.
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
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        # Both are real measurements that came out zero: examples were routed and none
        # were hallucinations, and hallucinations existed and none were caught. F1 is
        # 0.0, not undefined — returning None here would hide a gate that failed
        # completely, and would make the delta against a working gate unreportable.
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
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


# Metrics that get a bootstrap confidence interval. Count keys (n_true_positive and
# friends) are excluded: they are properties of this sample, not estimates of a
# population quantity, so an interval on them would be meaningless.
CI_METRICS = (
    "routing_rate",
    "routed_hallucination_precision",
    "hallucination_recall",
    "f1",
    "specificity",
    "balanced_accuracy",
    "entropy_auroc",
)

# Metrics whose transferred-vs-native difference carries a paired interval. Same list
# minus entropy_auroc, which is threshold-free and therefore identical for both gates
# on the same model — its delta is zero by construction (see routing_quality_delta).
DELTA_CI_METRICS = tuple(key for key in CI_METRICS if key != "entropy_auroc")


def detection_metrics_with_ci(
    outcomes: list[RoutingOutcome],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict:
    """`routing_detection_metrics` plus a bootstrap interval for each CI_METRICS key.

    The interval covers **evaluation sampling** only — how much each metric would move
    on another draw of examples from the same model. Uncertainty in the threshold
    itself is a separate question, bootstrapped from the calibration sample by the
    harness; see sense_eval.bootstrap for why these are different sources.

    One resampling pass computes the whole metric dict per resample, so adding a metric
    to CI_METRICS costs nothing extra.
    """
    import random as _random

    from sense_eval.bootstrap import interval_from_estimates

    point_metrics = routing_detection_metrics(outcomes)

    if not outcomes:
        return point_metrics | {
            f"{key}_ci": interval_from_estimates(
                point_metrics.get(key), [], n_resamples=n_resamples, confidence=confidence
            ).as_dict()
            for key in CI_METRICS
        }

    rng = _random.Random(seed)
    n = len(outcomes)
    collected: dict[str, list[float]] = {key: [] for key in CI_METRICS}
    for _ in range(n_resamples):
        resample = [outcomes[rng.randrange(n)] for _ in range(n)]
        resampled_metrics = routing_detection_metrics(resample)
        for key in CI_METRICS:
            value = resampled_metrics.get(key)
            if value is not None:
                collected[key].append(value)

    return point_metrics | {
        f"{key}_ci": interval_from_estimates(
            point_metrics.get(key), collected[key], n_resamples=n_resamples, confidence=confidence
        ).as_dict()
        for key in CI_METRICS
    }


def routing_quality_delta_with_ci(
    transferred_outcomes: list[RoutingOutcome],
    native_outcomes: list[RoutingOutcome],
    *,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> dict:
    """native minus transferred, with a PAIRED bootstrap interval per metric.

    The two outcome lists must be index-aligned — same examples, same generations, only
    the routing decisions differ. One index set is drawn per resample and both gates are
    scored on it, because the two conditions share examples and their errors are
    correlated; resampling them independently would widen the interval and understate a
    real gap.

    **An interval excluding zero is the evidence RQ1 and RQ2 are after.** It says the
    gap between a transferred threshold and a natively-calibrated one is larger than
    sampling noise on this model. An interval straddling zero says the data does not
    distinguish them, which on these two research questions is itself a finding and must
    be reported as one rather than read as a null.
    """
    import random as _random

    from sense_eval.bootstrap import interval_from_estimates

    if len(transferred_outcomes) != len(native_outcomes):
        raise ValueError(
            "paired delta needs index-aligned outcome lists, got "
            f"{len(transferred_outcomes)} and {len(native_outcomes)}"
        )

    transferred_point = routing_detection_metrics(transferred_outcomes)
    native_point = routing_detection_metrics(native_outcomes)

    def point_delta(key: str) -> float | None:
        a, b = transferred_point.get(key), native_point.get(key)
        return (b - a) if (a is not None and b is not None) else None

    if not transferred_outcomes:
        return {
            f"delta_{key}_ci": interval_from_estimates(
                point_delta(key), [], n_resamples=n_resamples, confidence=confidence
            ).as_dict()
            for key in DELTA_CI_METRICS
        }

    rng = _random.Random(seed)
    n = len(transferred_outcomes)
    collected: dict[str, list[float]] = {key: [] for key in DELTA_CI_METRICS}
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        resampled_transferred = routing_detection_metrics([transferred_outcomes[i] for i in indices])
        resampled_native = routing_detection_metrics([native_outcomes[i] for i in indices])
        for key in DELTA_CI_METRICS:
            a, b = resampled_transferred.get(key), resampled_native.get(key)
            if a is not None and b is not None:
                collected[key].append(b - a)

    return {
        f"delta_{key}_ci": interval_from_estimates(
            point_delta(key), collected[key], n_resamples=n_resamples, confidence=confidence
        ).as_dict()
        for key in DELTA_CI_METRICS
    }
