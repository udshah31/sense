from analyze_judge_labels import best_per_family, confusion


def _row(label, r, w, tier="fit"):
    return {"label": label, "right_entailment": r, "hallucinated_entailment": w, "tier": tier}


def test_confusion_counts_and_balanced_accuracy():
    rows = [_row("correct", 0.9, 0.1), _row("correct", 0.2, 0.1), _row("incorrect", 0.8, 0.1), _row("incorrect", 0.1, 0.1)]

    c = confusion(rows, lambda r, w: r >= 0.7)

    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (1, 1, 1, 1)
    assert c["balanced_accuracy"] == 0.5


def test_best_per_family_finds_the_separating_threshold():
    rows = [_row("correct", 0.9, 0.0)] * 3 + [_row("incorrect", 0.3, 0.0)] * 3

    best = best_per_family(rows)

    assert best["right-only"][1] == 1.0
    assert 0.3 < best["right-only"][0] <= 0.9
