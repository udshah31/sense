from sense_symbolic.decomposition import AtomicClaim
from sense_symbolic.extraction import extract_claims


def test_extracts_birth_year_claim():
    claims = extract_claims("Albert Einstein", "Albert Einstein was born in 1879.")
    assert AtomicClaim(subject_label="Albert Einstein", relation_kind="birth_year", object_label="1879") in claims


def test_birth_year_pattern_requires_born_keyword():
    # A year present without "born" nearby must not be mistaken for a birth claim.
    claims = extract_claims("Albert Einstein", "Albert Einstein worked on relativity in 1915.")
    assert not any(c.relation_kind == "birth_year" for c in claims)


def test_extracts_death_year_claim():
    claims = extract_claims("Albert Einstein", "Albert Einstein died in 1955.")
    assert AtomicClaim(subject_label="Albert Einstein", relation_kind="death_year", object_label="1955") in claims


def test_death_year_pattern_requires_died_keyword():
    claims = extract_claims("Albert Einstein", "Albert Einstein published a paper in 1955.")
    assert not any(c.relation_kind == "death_year" for c in claims)


def test_extracts_nationality_claim():
    claims = extract_claims("Albert Einstein", "Albert Einstein was a German scientist.")
    assert AtomicClaim(subject_label="Albert Einstein", relation_kind="nationality", object_label="german") in claims


def test_nationality_pattern_does_not_match_unrelated_words():
    # "Germany" contains "German" as a prefix but is not the whole-word adjective.
    claims = extract_claims("Albert Einstein", "Albert Einstein lived in Germany.")
    assert not any(c.relation_kind == "nationality" for c in claims)


def test_extracts_occupation_claim():
    claims = extract_claims("Albert Einstein", "Albert Einstein was a physicist.")
    assert AtomicClaim(subject_label="Albert Einstein", relation_kind="occupation", object_label="physicist") in claims


def test_occupation_pattern_does_not_match_unrelated_sentence():
    claims = extract_claims("Albert Einstein", "Albert Einstein enjoyed sailing.")
    assert not any(c.relation_kind == "occupation" for c in claims)


def test_extracts_employer_claim():
    claims = extract_claims("Alan Turing", "Alan Turing worked at the Government Code and Cypher School.")
    assert AtomicClaim(
        subject_label="Alan Turing",
        relation_kind="employer",
        object_label="the Government Code and Cypher School",
    ) in claims


def test_employer_pattern_requires_worked_at_or_for():
    claims = extract_claims("Alan Turing", "Alan Turing studied at Cambridge.")
    assert not any(c.relation_kind == "employer" for c in claims)


def test_extracts_multiple_claims_from_one_sentence():
    claims = extract_claims("Albert Einstein", "Albert Einstein was a German physicist born in 1879.")
    kinds = {c.relation_kind for c in claims}
    assert kinds == {"nationality", "occupation", "birth_year"}


def test_no_match_returns_empty_list():
    assert extract_claims("Albert Einstein", "Albert Einstein enjoyed playing the violin.") == []


def test_negation_is_not_handled_named_limitation():
    """Documented limitation (design doc's 'Out of scope' section): negation
    is not detected, so a negated sentence still yields the same claim a
    positive one would. This asserts the current, unsolved behavior
    explicitly rather than leaving it silently uncovered."""
    claims = extract_claims("Albert Einstein", "Albert Einstein was not born in 1879.")
    assert AtomicClaim(subject_label="Albert Einstein", relation_kind="birth_year", object_label="1879") in claims


def test_subject_label_is_never_inferred_from_sentence():
    # Caller-supplied subject_label is used verbatim, even when the sentence
    # names a different entity — extraction never re-derives the subject.
    claims = extract_claims("Marie Curie", "Albert Einstein was born in 1879.")
    assert all(c.subject_label == "Marie Curie" for c in claims)
