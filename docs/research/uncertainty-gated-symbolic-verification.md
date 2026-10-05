# Literature review: uncertainty-gated intervention and symbolic verification

Run 2026-10-05 via the `feynman-literature-review` workflow (adapted from Feynman, MIT
License, Companion Inc.). Scope and questions: `.plans/uncertainty-gated-symbolic-verification.md`.
Serves the advisor's instruction to expand the literature review, and targets proposal
§2.1–§2.4 and the novelty claim.

---

## Summary

Four findings, in descending order of consequence for the proposal.

1. **"Always on, as in retrieval" is wrong, and it is wrong in the one direction that
   costs SENSE the most.** Retrieval has had an uncertainty-gated branch since 2023.
   FLARE triggers retrieval only when the next sentence contains low-confidence tokens;
   DRAGIN triggers on a score built from *token entropy* and attention. The proposal's
   distinguishing sentence contrasts SENSE against a characterization of retrieval that
   the retrieval literature abandoned three years ago.
2. **DRAGIN gates on token entropy against a predefined threshold.** That is precisely
   the naive-fixed-threshold condition RQ1 is built to attack, in a main-conference ACL
   paper. This cuts both ways: another uncited precedent for entropy-gated intervention,
   and strong independent motivation for RQ1.
3. **Mean-aggregated sequence uncertainty has a known failure mode the proposal never
   mentions.** Gupta et al. identify length bias in sequence-level uncertainty and argue
   for token-level aggregation with learned deferral rules. SENSE averages token entropy
   over the generated span; the proposal does not justify that choice or name the
   alternatives.
4. **Logic-LM already handles autoformalization error by feeding solver error messages
   back for self-refinement.** §4.3 argues decomposition-first narrows the
   autoformalization surface, which is true and different, but the proposal presents the
   problem as largely unaddressed. It is not.

None of this sinks the contribution. All of it changes what §2 has to say.

---

## 1. Uncertainty-gated intervention is a populated field

The useful way to organize it is by two axes: **what the gate reads**, and **what it
routes to**. Arranged that way, SENSE's position is specific and defensible — and
visibly adjacent to work the proposal does not cite.

| Work | Gate signal | Threshold | Routes to |
|---|---|---|---|
| FLARE [28] | Token confidence in a lookahead sentence | Fixed | Retrieval + regeneration |
| DRAGIN [29] | Token entropy x max attention x semantic significance | **Predefined** | Retrieval + regeneration |
| Varshney et al. [25] | Low-confidence span detection | Fixed | Validation + repair |
| AdaDec [26] | Token Shannon entropy | **Learned per model** | Lookahead reranking |
| UnCert-CoT [27] | Token uncertainty | Fixed | Multi-path reasoning |
| LM cascades [31] | Aggregated token uncertainty | Learned deferral rule | A larger model |
| **SENSE** | **Token entropy** | **Calibrated per model** | **Symbolic verification** |

Two things follow. The *routing target* is where SENSE is actually alone — no one in
this table routes to a solver. And the *threshold* column is the real research question:
only AdaDec and the cascade line treat the threshold as something to be learned rather
than set, and neither studies whether a learned threshold moves across model families.

**FLARE** (Jiang et al., EMNLP 2023, pp. 7969–7992) [28] predicts the upcoming sentence,
and if it contains low-confidence tokens, uses it as a retrieval query and regenerates.
Retrieval is explicitly conditional, "activated dynamically during generation when
uncertainty is detected."

**DRAGIN** (Su et al., ACL 2024, pp. 12991–13013) [29] splits the problem into RIND
(*when* to retrieve) and QFS (*what* to retrieve). RIND scores each token as
`H_i · a_max(i) · s_i` — entropy times maximum attention from subsequent tokens times a
stopword-filtering significance indicator — and retrieves when that score crosses a
**predefined threshold**. The entropy term makes this the closest published neighbour to
SENSE's gate signal, and the predefined threshold makes it a live example of the practice
RQ1 questions.

---

## 2. The sentence in §2.3 that has to change

> "What distinguishes SENSE from this prior work is that the symbolic component is
> neither always on, as in retrieval, nor applied after the fact, as in post-hoc
> verification, but invoked selectively during generation on the basis of the model's own
> uncertainty."

The 2026-10-04 revision already moved this argument into a new §2.4 and narrowed it. The
narrowed version still rests on "always on, as in retrieval" as the contrast for the
first clause. Given [28] and [29], that clause should go.

**What survives, and is defensible:** retrieval and post-hoc verification have *both*
been given uncertainty gates; what has not been done is routing the gated call to a
**formal verifier** that returns a verdict rather than evidence. FLARE and DRAGIN retrieve
*documents* and regenerate; the model remains free to ignore them — which is the standard
criticism of RAG, and it applies with equal force to gated RAG. A solver returns
satisfied / violated / unknown. That is a categorically different kind of answer, and it
is the distinction worth building the novelty claim on.

That reframing is *stronger* than the current one, because it no longer depends on
mischaracterizing a neighbouring field.

---

## 3. Threshold transfer: thinner than expected, which is good for RQ1

Direct study of whether an uncertainty threshold transfers across models remains sparse.
AdaDec [26] is still the strongest single data point — a learned per-model threshold
across eight checkpoints, with the explicit finding that one fixed value does not
generalize — and it is cross-*scale* and cross-family within code models, not a designed
transfer study.

The cascade literature [31] arrives at a related conclusion from the cost side: naive
sequence-level uncertainty is a poor deferral rule, and the fix is a *learned* rule rather
than a tuned constant. Nobody in the reachable literature appears to have asked SENSE's
question directly: calibrate on model A, apply unchanged to model B, and measure what
routing quality is lost.

**This is good news, and it should be stated plainly in §2.4.** RQ1 is not a crowded
question. It is an unasked one, adjacent to several fields that each assume their own
answer.

---

## 4. A methodological gap the proposal should close

Gupta et al. [31] identify **length bias** in sequence-level uncertainty: aggregate
measures "over- or under-emphasize outputs based on their lengths," and they argue for
token-level aggregation with learned post-hoc deferral rules.

SENSE aggregates by taking the **mean** token entropy over the generated span. Averaging
is length-normalizing, and `max_new_tokens` is held constant across conditions, so the
exposure is smaller than in the cascade setting — but the proposal never states that mean
aggregation *is* the choice, never justifies it, and never names the alternatives (max,
last-token, top-k mean, length-corrected). A reader who knows [31] will ask.

**Concrete recommendation.** Add one sentence to §4.1 naming mean-over-span as the
aggregator and noting the length-bias literature, and add the aggregator to the ablation
list in §5 if budget allows. It is a cheap ablation — the per-example entropy values are
already recorded on both scales, so alternative aggregators can be computed from existing
runs without regenerating anything.

---

## 5. Solver-backed verification: VERGE is not the only precedent

**Logic-LM** (Pan et al., Findings of EMNLP 2023, pp. 3806–3824) [30] is the canonical
LLM-plus-solver pipeline: formalize with the LLM, solve deterministically, then
**self-refine using the solver's own error messages**. That last stage is a direct
response to autoformalization error, and it predates VERGE by over two years.

§4.3's argument is that decomposition into short atomic claims *narrows the surface* on
which autoformalization error can occur. That remains a distinct and defensible position —
Logic-LM detects and repairs translation errors after the fact, whereas decomposition-first
tries to make them less likely. But §4.3 currently reads as if the problem were open.
Citing [30] and drawing that contrast explicitly is strictly better: it shows the problem
is known, that others attack it downstream, and that SENSE attacks it upstream.

---

## 6. What this literature reports, and what it means for the detection framing

The gated-intervention papers report **downstream task metrics** — exact match, F1, Pass@1
— rather than detection quality of the gate itself. The uncertainty-estimation line
(semantic entropy and its descendants) reports **AUROC of the signal as a hallucination
detector**. SENSE now does both: detection performance of the gate, and the factuality
triple downstream.

That is unusual and worth claiming as a methodological contribution rather than leaving
implicit. Reporting only downstream metrics makes it impossible to tell whether a gain
came from a better signal or a better intervention — which is exactly the confound the
matched-budget comparison against SelfCheckGPT is designed to break.

---

## Recommended additions to the reference list

Numbered from [28], continuing the appended block. All four were verified this session
against the ACL Anthology or arXiv.

- **[28]** Z. Jiang, F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, and
  G. Neubig, "Active retrieval augmented generation," in *Proc. EMNLP*, Singapore, 2023,
  pp. 7969–7992.
- **[29]** W. Su, Y. Tang, Q. Ai, Z. Wu, and Y. Liu, "DRAGIN: Dynamic retrieval augmented
  generation based on the real-time information needs of large language models," in
  *Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics (Vol. 1: Long Papers)*, 2024,
  pp. 12991–13013.
- **[30]** L. Pan, A. Albalak, X. Wang, and W. Wang, "Logic-LM: Empowering large language
  models with symbolic solvers for faithful logical reasoning," in *Findings of the Assoc.
  Comput. Linguistics: EMNLP 2023*, 2023, pp. 3806–3824.
- **[31]** N. Gupta, H. Narasimhan, W. Jitkrittum, A. S. Rawat, A. K. Menon, and S. Kumar,
  "Language model cascades: Token-level uncertainty and beyond," arXiv:2404.10136, 2024.

Where each is cited: [28] and [29] in §2.4 (and [28] also in §2.3, where retrieval is
introduced); [30] in §2.2 and §4.3; [31] in §2.4 and §4.1.

---

## Open questions

- **[31]'s published venue is unconfirmed.** The arXiv page lists no journal reference.
  It is plausibly an ICLR 2024 paper; that was not verified here and must be checked
  before submission rather than assumed.
- **Is there a designed cross-family threshold-transfer study anywhere?** This review did
  not find one, which is RQ1's opportunity — but absence of evidence from one search pass
  is weak evidence of absence. Worth one targeted check of the calibration literature
  before claiming novelty in print.
- **Does gated retrieval beat gated verification on HaluEval?** FLARE and DRAGIN are
  arguably a *better* baseline for SENSE than plain RAG, because they share the gating
  mechanism and differ only in what the gate routes to. That would isolate the
  contribution precisely. It is also a second baseline implementation, which the credit
  budget may not support — worth raising with the advisor as a scoping decision rather
  than deciding unilaterally.
- **Aggregation ablation** — cheap to run from existing data, not currently planned.

---

## Sources

| Source | Status |
|---|---|
| [aclanthology.org/2023.emnlp-main.495](https://aclanthology.org/2023.emnlp-main.495) — FLARE | Verified: title, 9 authors, venue, pages, mechanism |
| [aclanthology.org/2024.acl-long.702](https://aclanthology.org/2024.acl-long.702) — DRAGIN | Verified: title, 5 authors, venue, pages |
| [arxiv.org/html/2403.10081v3](https://arxiv.org/html/2403.10081v3) — DRAGIN full text | Verified: RIND/QFS components and the entropy-attention trigger formula |
| [aclanthology.org/2023.findings-emnlp.248](https://aclanthology.org/2023.findings-emnlp.248) — Logic-LM | Verified: title, 4 authors, venue, pages, self-refinement mechanism |
| [arxiv.org/abs/2404.10136](https://arxiv.org/abs/2404.10136) — LM cascades | Verified: title, 6 authors, year. **Venue unconfirmed** |

Rejected or not pursued: general confidence-calibration surveys (too broad to change a
decision here); several post-2026 preprints surfaced by search whose relevance could not
be established from abstracts alone.
