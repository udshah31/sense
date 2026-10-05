# Provenance: uncertainty-gated intervention and symbolic verification

**Run date:** 2026-10-05
**Workflow:** `feynman-literature-review` (adapted from Feynman, MIT License, Companion Inc.)
**Model:** Claude Opus 5
**Scope:** proposal §2.1–§2.4 and the novelty claim; advisor's standing instruction to
expand the literature review.
**Mode:** direct search, no subagents — the topic was focused enough that decomposition
would have cost more than it returned.

## Files

| File | Role |
|---|---|
| `.plans/uncertainty-gated-symbolic-verification.md` | Scope, five key questions, task ledger |
| `uncertainty-gated-symbolic-verification.md` | The review |
| `uncertainty-gated-symbolic-verification.provenance.md` | This file |

No `.drafts/` files: with no subagents there were no intermediate research notes to
collect, and the review was written directly from verified sources.

## Task ledger — final state

| # | Question | Status |
|---|---|---|
| Q1 | Who else gates an intervention on the model's own uncertainty? | **done** — FLARE, DRAGIN, plus the already-cited Varshney/AdaDec/UnCert-CoT; cascades as an adjacent family |
| Q2 | Has anyone studied threshold transfer across models? | **done, negative** — no designed cross-family study found; AdaDec remains the closest. See Open Questions: one search pass is weak evidence of absence |
| Q3 | State of solver-backed verification | **done** — Logic-LM found as a precedent predating VERGE, with explicit autoformalization-error handling |
| Q4 | What the semantic-entropy line establishes and costs | **superseded** — the 2026-10-04 proposal revision already reframed §2.1 on this; nothing new found that would change it, so it was not expanded further |
| Q5 | Reporting conventions | **done** — gated-intervention papers report downstream task metrics; the uncertainty line reports detection AUROC. SENSE reporting both is unusual and claimable |

## Sources consulted, accepted, rejected

**Accepted (5 fetches, all verified against the publisher or arXiv):** ACL Anthology pages
for FLARE (2023.emnlp-main.495), DRAGIN (2024.acl-long.702) and Logic-LM
(2023.findings-emnlp.248); the arXiv HTML full text of DRAGIN for its mechanism; the arXiv
abstract for Language Model Cascades (2404.10136).

**Consulted and not used:** four WebSearch passes (FLARE, DRAGIN, solver-backed reasoning,
cascade/deferral routing) plus one extended search on calibration-threshold transfer. The
calibration search returned general confidence-calibration surveys and several post-2026
preprints whose relevance could not be established from abstracts; none were cited.

**Not pursued:** SatLM and LINC appeared in search results as further LLM-plus-solver work
but were not fetched. Logic-LM was sufficient to establish the point §4.3 needs, and
adding more solver papers would pad the reference list without changing a decision.

## Verification status

**Verified:** every citation entering the review had its title, author order, venue and
page range checked against the ACL Anthology or arXiv in this session. DRAGIN's trigger
formula was read from the paper's own HTML, not from a secondary description.

**Unverified, flagged inline:** reference [31]'s published venue. The arXiv page carries no
journal reference. It is plausibly ICLR 2024; this was not confirmed and the review says so
rather than asserting it.

**Known limitation:** Q2's negative result rests on a single extended search pass. The
review states this. Treat "nobody has asked RQ1 directly" as a working belief to re-check
before it appears in print, not as an established fact.

## Relationship to the earlier review

This run follows `sense-proposal-v5-review.md` (2026-10-04), which raised M1 — the novelty
claim contradicted by uncited prior art. That issue was closed the same day by adding
three references and a new §2.4. This run tests whether the *surrounding* literature was
surveyed, and finds that it was not: FLARE and DRAGIN break the "always on, as in
retrieval" contrast that the narrowed §2.4 still rests on. So M1 is reopened in a weaker
form, with a concrete and stronger replacement framing proposed in §2 of the review.
