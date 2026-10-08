from dump_labeling_sample import sample_rows


def test_sample_rows_is_deterministic_disjoint_and_stays_in_its_tier():
    fit, holdout = list(range(0, 100)), list(range(100, 130))

    a = sample_rows(fit, holdout, 5, 3, "mistral")

    assert a == sample_rows(fit, holdout, 5, 3, "mistral")
    assert a != sample_rows(fit, holdout, 5, 3, "qwen3_1_7b")
    assert [i in fit for i, t in a if t == "fit"] == [True] * 5
    assert [i in holdout for i, t in a if t == "holdout"] == [True] * 3
