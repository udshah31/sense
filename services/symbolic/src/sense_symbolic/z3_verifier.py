"""Autoformalization + Z3 check: turn an AtomicClaim (already decomposed and
restricted to a known relation kind by decomposition.py) into a Z3 query against
domain.py's fixed fact base, and check it.

Numeric relations (birth_year/death_year/before_year) go through a real Z3
entailment check — push the claim's negation, check unsat (facts force the claim
true), then push the claim itself, check unsat (facts force it false); if neither
is unsat the fixed KB doesn't determine the claim either way. String-valued
relations (nationality/occupation/employer) are plain set-membership checks in
Python — Z3 offers no reasoning leverage over opaque label equality that a
Python `in` doesn't already give, so formalizing those into the solver would be
theater, not verification.

Tri-state throughout, matching wikidata_client's ask_triple/search_entity_qid
contract: True/False only when the fixed facts actually determine the claim,
None whenever they don't (unresolved entity, unparsable year, or a fact the KB
simply doesn't record) — never a guess standing in for "unknown".
"""

import z3

from sense_symbolic.decomposition import AtomicClaim
from sense_symbolic.domain import declare_solver_with_facts, facts_for, resolve_entity


def _entails(solver: z3.Solver, expr: z3.BoolRef) -> bool | None:
    solver.push()
    solver.add(z3.Not(expr))
    negation_satisfiable = solver.check() == z3.sat
    solver.pop()

    solver.push()
    solver.add(expr)
    claim_satisfiable = solver.check() == z3.sat
    solver.pop()

    if claim_satisfiable and not negation_satisfiable:
        return True
    if negation_satisfiable and not claim_satisfiable:
        return False
    return None  # facts don't pin this down either way


def verify_claim(claim: AtomicClaim) -> bool | None:
    subject_key = resolve_entity(claim.subject_label)
    if subject_key is None:
        return None
    subject_facts = facts_for(subject_key)

    if claim.relation_kind == "before_year":
        object_key = resolve_entity(claim.object_label)
        if object_key is None:
            return None
        object_facts = facts_for(object_key)
        if subject_facts.birth_year is None or object_facts.birth_year is None:
            return None
        solver, years = declare_solver_with_facts()
        return _entails(solver, years[f"birth::{subject_key}"] < years[f"birth::{object_key}"])

    if claim.relation_kind in ("birth_year", "death_year"):
        try:
            claimed_year = int(claim.object_label.strip())
        except ValueError:
            return None
        actual_year = subject_facts.birth_year if claim.relation_kind == "birth_year" else subject_facts.death_year
        if actual_year is None:
            return None
        solver, years = declare_solver_with_facts()
        fact_key = f"{'birth' if claim.relation_kind == 'birth_year' else 'death'}::{subject_key}"
        return _entails(solver, years[fact_key] == claimed_year)

    if claim.relation_kind == "nationality":
        if subject_facts.nationality is None:
            return None
        return claim.object_label.strip().lower() == subject_facts.nationality

    if claim.relation_kind in ("occupation", "employer"):
        member_set = subject_facts.occupations if claim.relation_kind == "occupation" else subject_facts.employers
        if not member_set:
            return None
        return claim.object_label.strip().lower() in member_set

    return None
