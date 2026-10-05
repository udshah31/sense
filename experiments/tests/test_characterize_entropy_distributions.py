"""Tests for the entropy-characterization diagnostic.

Pure logic only — no models, no generation. The GPU-only part of the harness is
`characterize_model`, which is a load/generate/free wrapper; everything that computes a
reported number is tested here.
"""

import pytest

from characterize_entropy_distributions import (
    ascii_histogram,
    summarize_by_axis,
    summarize_distribution,
    threshold_spread,
    transfer_matrix,
)
from sense_orchestrator.gate import quantile_threshold as gate_quantile_fn


# --- the threshold the gate would produce --------------------------------------


@pytest.mark.parametrize("values", [
    [0.1],
    [0.1, 0.2],
    [0.5, 0.1, 0.9, 0.3, 0.7],
    [1.0, 1.0, 1.0, 1.0],
    [0.61, 0.71, 0.70, 1.13, 1.94],  # AdaDec-shaped raw thresholds
])
@pytest.mark.parametrize("q", [0.05, 0.5, 0.9, 0.95])
def test_reported_gate_threshold_is_exactly_what_the_gate_would_produce(values, q):
    """The diagnostic's whole purpose is to report the threshold a gate WOULD fit, so
    it calls the gate's own `quantile_threshold` rather than reimplementing it. This
    pins the property that matters: same input, same number the gate would calibrate
    to. (Until 2026-10-04 the quantile was private to the gate and this module kept a
    copy guarded by an equality test; making it public removed the need for the copy.)
    """
    assert summarize_distribution(values, gate_quantile=q)["gate_threshold"] == gate_quantile_fn(values, q)


# --- distribution summary --------------------------------------------------------


def test_summarize_distribution_reports_the_gate_threshold_at_the_given_quantile():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    summary = summarize_distribution(values, gate_quantile=0.9)

    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(0.3)
    assert (summary["min"], summary["max"]) == (0.1, 0.5)
    assert summary["gate_threshold"] == pytest.approx(gate_quantile_fn(values, 0.9))
    assert set(summary["quantiles"]) == {"q05", "q10", "q25", "q50", "q75", "q90", "q95"}


def test_stdev_is_none_for_a_single_value_rather_than_zero():
    # Zero would assert a spread measurement that one sample cannot support.
    assert summarize_distribution([0.42], gate_quantile=0.9)["stdev"] is None


# --- threshold spread ------------------------------------------------------------


def test_threshold_spread_headline_ratio():
    # AdaDec's reported raw thresholds: 0.6153 (QW-1.7B) to 1.9353 (CL-7B).
    spread = threshold_spread({"a": 0.6153, "b": 0.7060, "c": 1.9353})
    assert spread["min"] == 0.6153
    assert spread["max"] == 1.9353
    assert spread["max_over_min_ratio"] == pytest.approx(1.9353 / 0.6153)
    assert spread["absolute_range"] == pytest.approx(1.3200, abs=1e-4)


def test_identical_thresholds_give_a_ratio_of_one():
    # What the normalized scale should look like if normalization does its job.
    spread = threshold_spread({"a": 0.28, "b": 0.28})
    assert spread["max_over_min_ratio"] == 1.0
    assert spread["absolute_range"] == 0.0


def test_zero_threshold_yields_none_ratio_not_infinity():
    assert threshold_spread({"a": 0.0, "b": 0.5})["max_over_min_ratio"] is None


# --- transfer matrix -------------------------------------------------------------

THRESHOLDS = {"llama3": 0.50, "qwen3_8b": 0.80, "qwen3_1_7b": 0.82}
ENTROPIES = {
    "llama3": [0.1, 0.3, 0.5, 0.7, 0.9],
    "qwen3_8b": [0.2, 0.4, 0.6, 0.81, 0.95],
    "qwen3_1_7b": [0.25, 0.45, 0.65, 0.83, 0.97],
}
FAMILIES = {"llama3": "llama", "qwen3_8b": "qwen3", "qwen3_1_7b": "qwen3"}


def test_transfer_matrix_covers_every_ordered_pair_and_no_self_pairs():
    rows = transfer_matrix(THRESHOLDS, ENTROPIES, FAMILIES, quantile=0.9)
    assert len(rows) == 3 * 2
    assert all(row["source"] != row["target"] for row in rows)
    pairs = {(row["source"], row["target"]) for row in rows}
    assert ("llama3", "qwen3_8b") in pairs and ("qwen3_8b", "llama3") in pairs


def test_same_family_flag_tracks_the_configured_grouping():
    rows = transfer_matrix(THRESHOLDS, ENTROPIES, FAMILIES, quantile=0.9)
    by_pair = {(r["source"], r["target"]): r["same_family"] for r in rows}
    assert by_pair[("qwen3_8b", "qwen3_1_7b")] is True
    assert by_pair[("llama3", "qwen3_8b")] is False


def test_transferred_rate_uses_the_same_ge_comparison_the_gate_does():
    rows = transfer_matrix(THRESHOLDS, ENTROPIES, FAMILIES, quantile=0.9)
    row = next(r for r in rows if r["source"] == "llama3" and r["target"] == "qwen3_8b")

    # llama3's low threshold (0.50) applied to qwen3_8b's [0.2,0.4,0.6,0.81,0.95]
    # routes the three values >= 0.50 — a transferred threshold over-routing badly.
    assert row["target_rate_under_transferred_threshold"] == pytest.approx(3 / 5)
    # qwen3_8b's own threshold (0.80) routes two: 0.81 and 0.95.
    assert row["target_rate_under_native_threshold"] == pytest.approx(2 / 5)
    assert row["expected_routing_rate"] == pytest.approx(0.1)
    assert row["abs_deviation_from_expected"] == pytest.approx(abs(0.6 - 0.1))
    assert row["threshold_ratio"] == pytest.approx(0.50 / 0.80)


# --- axis summary ----------------------------------------------------------------


def test_summarize_by_axis_splits_cross_family_from_scale_pairs():
    rows = transfer_matrix(THRESHOLDS, ENTROPIES, FAMILIES, quantile=0.9)
    summary = summarize_by_axis(rows)

    assert summary["cross_family"]["n_pairs"] == 4
    assert summary["same_family"]["n_pairs"] == 2
    # The qwen3 pair's thresholds are close (0.80 vs 0.82), so its threshold ratios sit
    # near 1 while the cross-family ones do not — the AdaDec pattern, if it holds.
    assert summary["same_family"]["max_threshold_ratio"] == pytest.approx(0.82 / 0.80)
    assert summary["cross_family"]["max_threshold_ratio"] > summary["same_family"]["max_threshold_ratio"]


def test_an_axis_with_no_pairs_is_none_not_zero():
    """A single-family session must not look like evidence about cross-family transfer."""
    thresholds = {"qwen3_8b": 0.80, "qwen3_1_7b": 0.82}
    entropies = {k: ENTROPIES[k] for k in thresholds}
    families = {k: "qwen3" for k in thresholds}

    summary = summarize_by_axis(transfer_matrix(thresholds, entropies, families, quantile=0.9))
    assert summary["cross_family"] is None
    assert summary["same_family"]["n_pairs"] == 2


# --- histogram -------------------------------------------------------------------


def test_histogram_handles_a_degenerate_all_equal_distribution():
    lines = ascii_histogram([0.5] * 10)
    assert len(lines) == 1
    assert "all 10 values at 0.5000" in lines[0]


def test_histogram_emits_one_line_per_bin_and_counts_every_value():
    values = [i / 100 for i in range(100)]
    lines = ascii_histogram(values, bins=10)
    assert len(lines) == 10
    assert sum(int(line.rsplit(" ", 1)[1]) for line in lines) == 100


# --- the test-split guard --------------------------------------------------------


def test_run_refuses_to_characterize_on_the_test_split(monkeypatch):
    """CLAUDE.md's data split discipline: test is touched once, at the end."""
    import characterize_entropy_distributions as mod

    monkeypatch.setattr(mod, "load_model_registry", lambda: {})
    monkeypatch.setattr(
        mod,
        "load_yaml_config",
        lambda name: (
            {"decoding": {"do_sample": False, "max_new_tokens": 20}, "quantile": 0.9}
            if name == "gate.yaml"
            else {"models": [], "families": {}, "eval_split": "test", "n_examples": 10}
        ),
    )
    monkeypatch.setattr(mod, "seed_everything", lambda: 42)

    with pytest.raises(ValueError, match="test split is touched once"):
        mod.run()


def test_run_refuses_a_model_with_no_configured_family(monkeypatch):
    import characterize_entropy_distributions as mod

    monkeypatch.setattr(mod, "load_model_registry", lambda: {"llama3": {}})
    monkeypatch.setattr(
        mod,
        "load_yaml_config",
        lambda name: (
            {"decoding": {"do_sample": False, "max_new_tokens": 20}, "quantile": 0.9}
            if name == "gate.yaml"
            else {"models": ["llama3"], "families": {}, "eval_split": "development", "n_examples": 10}
        ),
    )
    monkeypatch.setattr(mod, "seed_everything", lambda: 42)

    with pytest.raises(ValueError, match="no entry"):
        mod.run()
