import z3

from sense_symbolic.decomposition import AtomicClaim
from sense_symbolic.z3_verifier import verify_claim


def test_verify_claim_true_via_z3_equality_entailment():
    claim = AtomicClaim(subject_label="Isaac Newton", relation_kind="birth_year", object_label="1643")
    assert verify_claim(claim) is True


def test_verify_claim_false_via_z3_equality_entailment():
    claim = AtomicClaim(subject_label="Isaac Newton", relation_kind="birth_year", object_label="1900")
    assert verify_claim(claim) is False


def test_verify_claim_death_after_birth_ordering_holds_for_every_known_entity():
    # Not a claim kind exposed over HTTP — exercises declare_solver_with_facts'
    # death >= birth constraint directly, since a violation there would silently
    # make every death_year/before_year check built on top of it unsound.
    from sense_symbolic.domain import KNOWN_ENTITIES, declare_solver_with_facts

    solver, years = declare_solver_with_facts()
    assert solver.check() == z3.sat  # the fixed fact base is itself consistent
    for key, facts in KNOWN_ENTITIES.items():
        if facts.birth_year is not None and facts.death_year is not None:
            assert facts.death_year >= facts.birth_year, f"{key}: death before birth in the fixed KB itself"


def test_verify_claim_before_year_true():
    claim = AtomicClaim(subject_label="Arthur's Magazine", relation_kind="before_year", object_label="First for Women")
    assert verify_claim(claim) is True


def test_verify_claim_before_year_false():
    claim = AtomicClaim(subject_label="First for Women", relation_kind="before_year", object_label="Arthur's Magazine")
    assert verify_claim(claim) is False


def test_verify_claim_unresolved_subject_yields_none():
    claim = AtomicClaim(subject_label="not a real person", relation_kind="birth_year", object_label="1900")
    assert verify_claim(claim) is None


def test_verify_claim_unparsable_year_yields_none():
    claim = AtomicClaim(subject_label="Isaac Newton", relation_kind="birth_year", object_label="a long time ago")
    assert verify_claim(claim) is None


def test_verify_claim_nationality_true():
    claim = AtomicClaim(subject_label="Marie Curie", relation_kind="nationality", object_label="Polish")
    assert verify_claim(claim) is True


def test_verify_claim_nationality_false():
    claim = AtomicClaim(subject_label="Marie Curie", relation_kind="nationality", object_label="French")
    assert verify_claim(claim) is False


def test_verify_claim_occupation_membership():
    claim = AtomicClaim(subject_label="Alan Turing", relation_kind="occupation", object_label="mathematician")
    assert verify_claim(claim) is True


def test_verify_claim_occupation_non_membership_is_false_not_none():
    # Distinct from the unknown-fact case below: the KB DOES have Newton's
    # occupations, "politician" just isn't among them — a determined False, not
    # an unresolved None.
    claim = AtomicClaim(subject_label="Isaac Newton", relation_kind="occupation", object_label="politician")
    assert verify_claim(claim) is False


def test_verify_claim_employer_unknown_fact_yields_none():
    claim = AtomicClaim(subject_label="Isaac Newton", relation_kind="employer", object_label="royal society")
    assert verify_claim(claim) is None


def test_verify_claim_death_year_true_via_z3_equality_entailment():
    claim = AtomicClaim(subject_label="Isaac Newton", relation_kind="death_year", object_label="1727")
    assert verify_claim(claim) is True


def test_verify_claim_death_year_false_via_z3_equality_entailment():
    claim = AtomicClaim(subject_label="Isaac Newton", relation_kind="death_year", object_label="1600")
    assert verify_claim(claim) is False


def test_verify_claim_before_year_unresolved_object_yields_none():
    claim = AtomicClaim(subject_label="Arthur's Magazine", relation_kind="before_year", object_label="not a real entity")
    assert verify_claim(claim) is None


def test_verify_claim_nationality_unknown_fact_yields_none():
    # Arthur's Magazine is in the KB but has no nationality recorded.
    claim = AtomicClaim(subject_label="Arthur's Magazine", relation_kind="nationality", object_label="american")
    assert verify_claim(claim) is None
