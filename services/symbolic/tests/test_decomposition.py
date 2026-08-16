from sense_symbolic.decomposition import AtomicClaim, decompose_claim


def test_decompose_known_predicate_returns_atomic_claim():
    claim = decompose_claim("Albert Einstein", "P106", "physicist")
    assert claim == AtomicClaim(subject_label="Albert Einstein", relation_kind="occupation", object_label="physicist")


def test_decompose_unknown_predicate_returns_none():
    assert decompose_claim("Albert Einstein", "P9999999", "physicist") is None


def test_decompose_before_year_pseudo_predicate():
    claim = decompose_claim("Arthur's Magazine", "BEFORE_YEAR", "First for Women")
    assert claim.relation_kind == "before_year"
