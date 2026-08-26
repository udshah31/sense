from pathlib import Path

from sense_data.factscore_labeled import FActScoreLabeledExample, MODELS, load_factscore_labeled

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "factscore_labeled"


def test_loads_all_three_models_183_topics_each():
    examples = load_factscore_labeled(FIXTURES_DIR)
    assert len(examples) == 549
    for model in MODELS:
        assert sum(1 for e in examples if e.model == model) == 183


def test_examples_are_well_formed():
    examples = load_factscore_labeled(FIXTURES_DIR)
    first = examples[0]
    assert isinstance(first, FActScoreLabeledExample)
    assert first.model in MODELS
    assert first.topic
    assert isinstance(first.generated_text, str)


def test_human_atomic_fact_labels_are_only_s_ns_ir():
    examples = load_factscore_labeled(FIXTURES_DIR)
    labels = {fact.label for e in examples for fact in e.human_atomic_facts}
    assert labels <= {"S", "NS", "IR"}


def test_handles_null_annotations_and_null_atomic_facts_without_raising():
    # Some rows have annotations: null (empty generation) and some annotations
    # have human-atomic-facts: null (is-relevant: false sentences) — both must
    # produce an empty human_atomic_facts tuple, not raise.
    examples = load_factscore_labeled(FIXTURES_DIR)
    assert any(e.generated_text == "" and e.human_atomic_facts == () for e in examples)


def test_most_topics_have_resolved_reference_text():
    examples = load_factscore_labeled(FIXTURES_DIR)
    topics = {e.topic for e in examples}
    resolved = {e.topic for e in examples if e.reference_text is not None}
    # Wikipedia fetch (data/scripts/fetch_factscore_labeled_reference_text.py) is
    # a best-effort live fetch — a handful of topics can fail to resolve, but the
    # large majority must succeed for the fixture to be usable for calibration.
    assert len(resolved) / len(topics) > 0.9


def test_load_is_deterministic_across_calls():
    a = load_factscore_labeled(FIXTURES_DIR)
    b = load_factscore_labeled(FIXTURES_DIR)
    assert [e.topic for e in a] == [e.topic for e in b]
