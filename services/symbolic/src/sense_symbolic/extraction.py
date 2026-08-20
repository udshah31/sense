"""Free-text claim extraction: rule-based patterns that turn a caller-supplied
subject label plus a free-text sentence into zero or more AtomicClaims,
restricted to the same relation kinds domain.py's fixed KB can ever verify.

See docs/superpowers/specs/2026-08-20-freetext-claim-extraction-design.md.

Deliberately rule-based, not NER or an LLM (design's "Approach" section) —
biased toward precision over recall: a sentence yielding no claims is
preferred over a sentence yielding a wrong one, the same "refuse rather than
guess" contract decompose_claim already has for the structured path.

`before_year` is out of scope here — a two-entity relation that doesn't arise
from a single free-text sentence about one subject; it stays reachable only
via the existing structured/probe path.

Subject resolution: subject_label is always supplied by the caller (the
biography's title entity, per the design doc's "Subject resolution" section)
and used verbatim — extraction never infers a subject from the sentence, so
no coreference/pronoun resolution exists here.

Named limitation: negation is not handled. "He was not born in 1879" still
extracts a birth_year claim, identical to the positive form — see the design
doc's "Out of scope" section. Not fixed by this module.
"""

import re

from sense_symbolic.decomposition import AtomicClaim
from sense_symbolic.domain import KNOWN_ENTITIES

_BIRTH_YEAR_PATTERN = re.compile(r"\bborn\b.*?\b(1[0-9]{3}|20[0-9]{2})\b", re.IGNORECASE)
_DEATH_YEAR_PATTERN = re.compile(r"\bdied\b.*?\b(1[0-9]{3}|20[0-9]{2})\b", re.IGNORECASE)

# Closed vocabularies seeded from domain.py's own KNOWN_ENTITIES — extraction
# only ever needs to recognize a nationality/occupation/employer the fixed KB
# could actually confirm or refute, not general adjective/noun/span detection.
_NATIONALITIES = sorted({facts.nationality for facts in KNOWN_ENTITIES.values() if facts.nationality})
_OCCUPATIONS = sorted({occupation for facts in KNOWN_ENTITIES.values() for occupation in facts.occupations})
_EMPLOYERS = sorted({employer for facts in KNOWN_ENTITIES.values() for employer in facts.employers})


def _extract_birth_year(sentence: str) -> str | None:
    match = _BIRTH_YEAR_PATTERN.search(sentence)
    return match.group(1) if match else None


def _extract_death_year(sentence: str) -> str | None:
    match = _DEATH_YEAR_PATTERN.search(sentence)
    return match.group(1) if match else None


def _extract_nationalities(sentence: str) -> list[str]:
    lowered = sentence.lower()
    return [nationality for nationality in _NATIONALITIES if re.search(rf"\b{re.escape(nationality)}\b", lowered)]


def _extract_occupations(sentence: str) -> list[str]:
    lowered = sentence.lower()
    return [occupation for occupation in _OCCUPATIONS if re.search(rf"\b{re.escape(occupation)}\b", lowered)]


def _extract_employers(sentence: str) -> list[str]:
    lowered = sentence.lower()
    return [employer for employer in _EMPLOYERS if re.search(rf"\b{re.escape(employer)}\b", lowered)]


def extract_claims(subject_label: str, sentence: str) -> list[AtomicClaim]:
    """Runs each relation-kind pattern against `sentence` independently and
    returns one AtomicClaim per match — a sentence can yield 0, 1, or several
    claims (e.g. a sentence naming a nationality, an occupation, and a birth
    year all yields three). subject_label is used verbatim for every claim
    returned, never re-derived from the sentence."""
    claims: list[AtomicClaim] = []

    birth_year = _extract_birth_year(sentence)
    if birth_year is not None:
        claims.append(AtomicClaim(subject_label=subject_label, relation_kind="birth_year", object_label=birth_year))

    death_year = _extract_death_year(sentence)
    if death_year is not None:
        claims.append(AtomicClaim(subject_label=subject_label, relation_kind="death_year", object_label=death_year))

    for nationality in _extract_nationalities(sentence):
        claims.append(
            AtomicClaim(subject_label=subject_label, relation_kind="nationality", object_label=nationality)
        )

    for occupation in _extract_occupations(sentence):
        claims.append(AtomicClaim(subject_label=subject_label, relation_kind="occupation", object_label=occupation))

    for employer in _extract_employers(sentence):
        claims.append(AtomicClaim(subject_label=subject_label, relation_kind="employer", object_label=employer))

    return claims
