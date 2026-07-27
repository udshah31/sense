"""Entropy-gating policy: routes generation to the symbolic stream when token
entropy crosses a per-model calibrated threshold.

Merge-back policy (accept/replace/abstain/annotate) is an open decision — see
CLAUDE.md — so this module only produces a route/no-route decision; the harness
annotates only, until that decision is made.
"""

from sense_data.splits import SplitIndices, assert_no_test_leakage


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile, no numpy dependency."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class UncalibratedGateError(Exception):
    """Raised when a routing decision is requested before the gate is calibrated."""


class GatePolicy:
    """Per-model entropy-gating threshold.

    There is no default threshold (CLAUDE.md: never add a default threshold value as
    a convenience fallback). The gate must be explicitly calibrated — either fit from
    calibration-split data via `calibrate`, or set to a threshold transferred from
    another model's calibration via `set_threshold` — before `decide` will run.
    """

    def __init__(self):
        self._threshold: float | None = None
        self._calibration_source: str | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._threshold is not None

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise UncalibratedGateError("gate has no threshold; call calibrate() or set_threshold() first")
        return self._threshold

    @property
    def calibration_source(self) -> str | None:
        return self._calibration_source

    def calibrate(
        self,
        calibration_entropies: list[float],
        calibration_indices: list[int],
        splits: SplitIndices,
        quantile: float,
        source: str = "native",
    ) -> float:
        """Fit the threshold as a quantile of calibration-split entropy values.

        `calibration_indices` — the dataset indices the entropy values came from —
        are checked against `splits.test` before fitting anything; this call raises
        rather than fitting a threshold if any test index is present.
        """
        assert_no_test_leakage(calibration_indices, splits)
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if not calibration_entropies:
            raise ValueError("calibration_entropies must be non-empty")

        self._threshold = _quantile(calibration_entropies, quantile)
        self._calibration_source = source
        return self._threshold

    def set_threshold(self, threshold: float, source: str) -> None:
        """Explicitly set an already-fit threshold, e.g. one transferred from another
        model's calibration (RQ1: does a fixed threshold transfer across families).

        `source` must identify where the threshold came from — there is no anonymous
        or default setting.
        """
        self._threshold = threshold
        self._calibration_source = source

    def decide(self, entropy: float) -> bool:
        """Return True if `entropy` crosses the calibrated threshold and generation
        should route to the symbolic verification stream."""
        return entropy >= self.threshold  # raises UncalibratedGateError if unset
