from sense_data.halueval import HaluEvalExample, load_halueval


def test_loads_expected_count():
    examples = load_halueval()
    assert len(examples) == 10_000


def test_examples_are_well_formed():
    examples = load_halueval()
    first = examples[0]
    assert isinstance(first, HaluEvalExample)
    assert first.index == 0
    assert first.knowledge
    assert first.question
    assert first.right_answer
    assert first.hallucinated_answer


def test_index_matches_position():
    examples = load_halueval()
    assert [e.index for e in examples] == list(range(len(examples)))


def test_load_is_deterministic_across_calls():
    a = load_halueval()
    b = load_halueval()
    assert [e.question for e in a] == [e.question for e in b]
