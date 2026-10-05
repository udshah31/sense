"""Bootstrap confidence intervals for the routing-quality numbers.

**Why this, and not "multiple seeds."** The writing guide's §4.6 asks for multiple
seeds and reported variance, which is the right instinct for a stochastic pipeline.
This pipeline is not stochastic: `configs/gate.yaml` sets `do_sample: false`, so
re-running a checkpoint on the same examples produces byte-identical generations and
the seed-to-seed variance is exactly zero. Reporting that zero as "variance" would be
worse than reporting nothing — it would claim a stability result the experiment never
tested.

The variance that genuinely exists here comes from sampling, not from seeds:

  1. **Calibration sampling.** The gate threshold is a quantile of a finite calibration
     sample. A different calibration draw from the same model gives a different
     threshold. This is the dominant source of uncertainty in RQ1 and RQ2, since
     everything downstream is a function of that threshold.
  2. **Evaluation sampling.** Precision, recall, F1 and AUROC are estimated on a finite
     development split. Another draw of examples gives different values.

Both are addressable by resampling data already in hand, at **zero additional GPU
cost** — no extra generation, no extra judge calls. That is what this module does.

**Paired resampling for differences.** RQ1 and RQ2 both rest on a difference between
two gates evaluated on *the same* examples, so their errors are correlated. Resampling
the two independently would produce an interval that is too wide and would understate
a real effect. `paired_delta_bootstrap` draws one index set per resample and scores
both gates on it, which is the correct interval for a paired difference.

Percentile intervals are used rather than BCa: they are simple enough to verify by
hand, and at these sample sizes the refinement BCa buys is small relative to the other
uncertainties in the study. State that choice in the methods rather than leaving a
reader to assume something fancier.
"""

import random
from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

# 1,000 resamples is the usual floor for a stable 95% percentile interval: the interval
# endpoints are the 25th and 975th order statistics, so each is estimated from enough
# resamples to be steady to about the third decimal. Raising it costs linear time and
# buys little; the number is configurable in configs/run.yaml.
DEFAULT_N_RESAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95

# If more than this fraction of resamples leave the statistic undefined (a rate with an
# empty denominator, an AUROC with only one class present), the interval is not
# reported. An interval computed from the surviving minority would silently describe a
# different population than the point estimate.
MAX_DEGENERATE_FRACTION = 0.1


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with a percentile bootstrap interval around it.

    `low`/`high` are None when the statistic was undefined on the full sample, or when
    too many resamples were degenerate (see MAX_DEGENERATE_FRACTION). None means "not
    estimable from this data", never zero — a zero-width interval is a claim.
    """

    point: float | None
    low: float | None
    high: float | None
    n_resamples: int
    confidence: float
    n_degenerate: int

    def as_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "ci_confidence": self.confidence,
            "n_resamples": self.n_resamples,
            "n_degenerate_resamples": self.n_degenerate,
        }


def _percentile(ordered: list[float], q: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _interval(
    point: float | None,
    estimates: list[float],
    n_resamples: int,
    confidence: float,
) -> ConfidenceInterval:
    n_degenerate = n_resamples - len(estimates)
    too_many_degenerate = n_degenerate > MAX_DEGENERATE_FRACTION * n_resamples

    if point is None or not estimates or too_many_degenerate:
        return ConfidenceInterval(point, None, None, n_resamples, confidence, n_degenerate)

    estimates.sort()
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        point=point,
        low=_percentile(estimates, tail),
        high=_percentile(estimates, 1.0 - tail),
        n_resamples=n_resamples,
        confidence=confidence,
        n_degenerate=n_degenerate,
    )


def interval_from_estimates(
    point: float | None,
    estimates: list[float],
    *,
    n_resamples: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> ConfidenceInterval:
    """Build an interval from resample estimates gathered by the caller.

    For callers computing many statistics over the same resamples: recomputing a whole
    metric dict once per metric would multiply the work by the number of metrics, so
    they run one resampling pass, collect each statistic's estimates, and hand them
    here. `estimates` may be shorter than `n_resamples` — the shortfall is the
    degenerate count.
    """
    return _interval(point, list(estimates), n_resamples, confidence)


def percentile_bootstrap(
    data: Sequence[T],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for `statistic` over `data`.

    `statistic` may return None for a resample where it is undefined; those resamples
    are counted and excluded rather than coerced to a number. `seed` is required, not
    defaulted — an unreproducible interval is not a reportable one.
    """
    point = statistic(data)
    if not data:
        return ConfidenceInterval(point, None, None, n_resamples, confidence, n_resamples)

    rng = random.Random(seed)
    n = len(data)
    estimates: list[float] = []
    for _ in range(n_resamples):
        resample = [data[rng.randrange(n)] for _ in range(n)]
        value = statistic(resample)
        if value is not None:
            estimates.append(value)

    return _interval(point, estimates, n_resamples, confidence)


def paired_delta_bootstrap(
    data_a: Sequence[T],
    data_b: Sequence[T],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int,
) -> ConfidenceInterval:
    """Interval for `statistic(b) - statistic(a)` where a and b are index-aligned.

    One index set is drawn per resample and used for BOTH sequences, which is what
    makes the interval correct for a paired difference: the two conditions are
    evaluated on the same examples, so their errors are correlated and independent
    resampling would overstate the uncertainty.

    An interval excluding zero is the evidence RQ1 and RQ2 need — it says the gap
    between the transferred and native gates is larger than sampling noise.
    """
    if len(data_a) != len(data_b):
        raise ValueError(
            f"paired bootstrap needs index-aligned sequences, got {len(data_a)} and {len(data_b)}"
        )

    value_a, value_b = statistic(data_a), statistic(data_b)
    point = (value_b - value_a) if (value_a is not None and value_b is not None) else None

    if not data_a:
        return ConfidenceInterval(point, None, None, n_resamples, confidence, n_resamples)

    rng = random.Random(seed)
    n = len(data_a)
    estimates: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        resample_a = [data_a[i] for i in indices]
        resample_b = [data_b[i] for i in indices]
        a, b = statistic(resample_a), statistic(resample_b)
        if a is not None and b is not None:
            estimates.append(b - a)

    return _interval(point, estimates, n_resamples, confidence)
