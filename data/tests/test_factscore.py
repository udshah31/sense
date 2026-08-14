from sense_data.factscore import FActScoreExample, load_factscore


def test_loads_expected_count():
    examples = load_factscore()
    assert len(examples) == 500


def test_examples_are_well_formed():
    examples = load_factscore()
    first = examples[0]
    assert isinstance(first, FActScoreExample)
    assert first.index == 0
    assert first.entity
    assert first.one_fact_prompt
    assert first.factscore_prompt
    assert first.hundredw_prompt
    assert first.around_100
    assert first.wikipedia_text


def test_index_matches_position():
    examples = load_factscore()
    assert [e.index for e in examples] == list(range(len(examples)))


def test_load_is_deterministic_across_calls():
    a = load_factscore()
    b = load_factscore()
    assert [e.entity for e in a] == [e.entity for e in b]
