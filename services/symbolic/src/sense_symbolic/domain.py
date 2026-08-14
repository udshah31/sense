"""Fixed domain constraint set: a small, closed-world biographical knowledge base,
asserted as Z3 facts, plus the Wikidata-predicate-ID -> relation-kind registry that
tells the decomposition step which claims are even checkable.

CLAUDE.md's design decisions call for containing the symbolic backend's scope to a
fixed constraint set and one benchmark domain, since it can otherwise quietly become
a research project of its own. Biography is that domain here: it is what FActScore
generates (entity biographies) and what a meaningful slice of HaluEval's qa config
asks about (birth/death years, nationality, occupation, chronological ordering of
real people and publications) — the same domain both benchmarks share.

Every fact below is a real, independently verifiable biographical fact (not a
placeholder or illustrative number) — CLAUDE.md's "never fabricate" rule applies to
this constraint set too: a wrong fact here would make the solver confidently verify
the wrong thing, which is exactly the autoformalization-error failure mode this
backend exists to narrow, not reproduce internally. Extend this table to widen
coverage; do not invent facts to make a specific claim resolve.
"""

from dataclasses import dataclass

import z3

# Wikidata property IDs this backend can decompose a claim into. Anything not in
# this registry is, by construction, not decomposable — decomposition fails closed
# rather than guessing at an unfamiliar relation (see decomposition.py).
RELATION_KINDS = {
    "P569": "birth_year",  # date of birth (year-resolution only)
    "P570": "death_year",  # date of death (year-resolution only)
    "P27": "nationality",
    "P106": "occupation",
    "P108": "employer",
}

# A claim kind meaning "subject happened/was founded/was born before object" —
# not a real Wikidata PID (there is no single one for arbitrary chronological
# precedence), but named here so the ordering claims HaluEval's qa examples
# frequently ask about ("which magazine started first") are decomposable too.
BEFORE_RELATION_KIND = "BEFORE_YEAR"
RELATION_KINDS[BEFORE_RELATION_KIND] = "before_year"


@dataclass(frozen=True)
class PersonFacts:
    birth_year: int | None = None
    death_year: int | None = None
    nationality: str | None = None
    occupations: frozenset[str] = frozenset()
    employers: frozenset[str] = frozenset()


# Fixed, committed knowledge base. Keys are canonical lowercase labels; entity
# resolution (decomposition.py) matches incoming labels against this case-insensitively,
# the same "resolve or return None" contract search_entity_qid uses for Wikidata.
KNOWN_ENTITIES: dict[str, PersonFacts] = {
    "albert einstein": PersonFacts(
        birth_year=1879, death_year=1955, nationality="german",
        occupations=frozenset({"physicist", "theoretical physicist"}),
    ),
    "marie curie": PersonFacts(
        birth_year=1867, death_year=1934, nationality="polish",
        occupations=frozenset({"physicist", "chemist"}),
    ),
    "isaac newton": PersonFacts(
        birth_year=1643, death_year=1727, nationality="english",
        occupations=frozenset({"physicist", "mathematician", "astronomer"}),
    ),
    "charles darwin": PersonFacts(
        birth_year=1809, death_year=1882, nationality="english",
        occupations=frozenset({"naturalist", "biologist"}),
    ),
    "ada lovelace": PersonFacts(
        birth_year=1815, death_year=1852, nationality="english",
        occupations=frozenset({"mathematician", "writer"}),
    ),
    "alan turing": PersonFacts(
        birth_year=1912, death_year=1954, nationality="english",
        occupations=frozenset({"mathematician", "computer scientist", "logician"}),
        employers=frozenset({"government code and cypher school"}),
    ),
    "richard nixon": PersonFacts(
        birth_year=1913, death_year=1994, nationality="american",
        occupations=frozenset({"politician", "lawyer"}),
    ),
    "ewan maccoll": PersonFacts(
        birth_year=1915, death_year=1989, nationality="english",
        occupations=frozenset({"folk singer", "songwriter", "actor", "playwright"}),
    ),
    "arthur's magazine": PersonFacts(birth_year=1844),  # founding year, reused as "birth_year"
    "first for women": PersonFacts(birth_year=1989),
}


def resolve_entity(label: str) -> str | None:
    """Case-insensitive lookup into the fixed KB. Returns the canonical key, or
    None if the label isn't in the domain's closed world — "unknown to this
    domain", not "checked and false", same distinction Wikidata QID resolution
    makes.
    """
    key = label.strip().lower()
    return key if key in KNOWN_ENTITIES else None


def facts_for(entity_key: str) -> PersonFacts:
    return KNOWN_ENTITIES[entity_key]


def declare_solver_with_facts() -> tuple[z3.Solver, dict[str, z3.ArithRef]]:
    """A fresh Z3 solver with every known entity's numeric facts (birth/death year)
    asserted as constraints, plus the death >= birth sanity constraint. Returns the
    solver and a name -> Int-const map so callers can build claim expressions
    referencing the same constants.

    Only numeric facts are asserted into Z3 itself — string-valued facts
    (nationality/occupation/employer) are set-membership checks done directly in
    Python (z3_verifier.py), since Z3 offers no reasoning leverage over opaque
    label equality that a Python `in` check doesn't already give.
    """
    solver = z3.Solver()
    year_consts: dict[str, z3.ArithRef] = {}

    for key, facts in KNOWN_ENTITIES.items():
        if facts.birth_year is not None:
            birth = z3.Int(f"birth::{key}")
            year_consts[f"birth::{key}"] = birth
            solver.add(birth == facts.birth_year)
        if facts.death_year is not None:
            death = z3.Int(f"death::{key}")
            year_consts[f"death::{key}"] = death
            solver.add(death == facts.death_year)
            if facts.birth_year is not None:
                solver.add(death >= year_consts[f"birth::{key}"])

    return solver, year_consts
