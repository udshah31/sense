from sense_data.truthful_qa import TruthfulQAExample, load_truthful_qa


def test_loads_expected_count():
    examples = load_truthful_qa()
    assert len(examples) == 817


def test_examples_are_well_formed():
    examples = load_truthful_qa()
    first = examples[0]
    assert isinstance(first, TruthfulQAExample)
    assert first.index == 0
    assert first.question
    assert first.best_answer
    assert len(first.correct_answers) >= 1
    assert len(first.incorrect_answers) >= 1


def test_index_matches_position():
    examples = load_truthful_qa()
    assert [e.index for e in examples] == list(range(len(examples)))


def test_load_is_deterministic_across_calls():
    a = load_truthful_qa()
    b = load_truthful_qa()
    assert [e.question for e in a] == [e.question for e in b]
