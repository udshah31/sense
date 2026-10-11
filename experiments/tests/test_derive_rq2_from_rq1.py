from derive_rq2_from_rq1 import derive, f1_verdict


def _rq1(f1, fixed_rr, adaptive_rr):
    gate = lambda rr: {"routing_rate": rr}  # noqa: E731
    return {
        "quantile": 0.9, "n_eval_examples": 1000,
        "source_model": {"name": "a"}, "target_model": {"name": "b"},
        "arms": {"normalized": {
            "source_native_threshold": 0.1, "target_native_threshold": 0.2,
            "transferred_gate": gate(fixed_rr), "native_gate": gate(adaptive_rr),
            "delta_native_minus_transferred": {"f1": f1},
        }},
    }


def test_verdict_uses_the_interval_not_the_point_estimate():
    assert f1_verdict([0.2, 0.1, 0.3]) == "adaptive better"
    assert f1_verdict([-0.2, -0.3, -0.1]) == "fixed better"
    assert f1_verdict([0.05, -0.01, 0.1]) == "no clear difference"


def test_derive_maps_transferred_to_fixed_native_to_adaptive_and_computes_fidelity_gaps():
    out = derive(_rq1([0.2, 0.1, 0.3], fixed_rr=0.006, adaptive_rr=0.114))["arms"]["normalized"]

    assert out["fixed_threshold"] == 0.1 and out["adaptive_threshold"] == 0.2
    assert abs(out["fixed_calibration_fidelity_gap"] - 0.094) < 1e-9
    assert abs(out["adaptive_calibration_fidelity_gap"] - 0.014) < 1e-9
    assert out["f1_verdict"] == "adaptive better"
