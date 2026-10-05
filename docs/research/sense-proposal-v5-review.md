# Peer review: SENSE starred paper proposal v5

**Artifact:** `SENSE_starred_paper_proposal_v5.docx` (26 July 2026, revised 10 and 25 August 2026)
**Reviewed:** 2026-10-04
**Corroborating evidence:** `SENSE_starred_paper_writing_guide.md`; the implementation repo at `/Users/udaysah/projects/sense` @ `14ecd46`
**Evidence notes:** `docs/research/.drafts/sense-proposal-v5-review-evidence.md`
**Plan:** `docs/research/.plans/sense-proposal-v5-review-plan.md`

Produced by the `feynman-research-review` workflow (adapted from Feynman, MIT License,
Companion Inc. — github.com/Companion-Inc/feynman).

> **Action status, updated 2026-10-04.** Every code-side issue is addressed on branch
> `review-c1-c3-routing-quality` (see `CLAUDE.md`, "2026-10-04 — proposal-v5 review
> changes"): C1 (routing quality defined as detection performance), C3 (raw-scale
> transfer arm, raw entropy retained), M2 (SelfCheckGPT-NLI post-hoc baseline, run at
> matched budget), M3 (bootstrap confidence intervals — deliberately not the "multiple
> seeds" the guide asks for; greedy decoding makes seed variance exactly zero).
> C2, M1, M4, M5, M3's document side and most Minor items are also done, in the
> proposal and writing guide on disk (see `CLAUDE.md`). Still open: m2 (ref [10]
> initials), m3 (citation renumbering), m9 (AI Use Statement), and verification of
> the newly added citations. None of the code-side work has been executed on a GPU. This review is otherwise left as written —
> it is the point-in-time record the revision plan was built from.

---

## Summary Assessment

This is a well-scoped, honestly argued proposal, and it is unusual in a good way: the
design decisions are dated, the exclusions are justified rather than silent, and the
reporting discipline around abstention is the kind of thing most proposals at this level
get wrong. The advisor has approved it. Nothing below questions that approval.

What the review found is a gap between the document and the implementation that is wider
than a drafting lag, and it runs in a specific direction. On three points the repo has
made a decision the proposal does not reflect, and in each case the repo's decision
narrows what the experiments can show:

1. The proposal's primary evidence for RQ1 and RQ2 — "the difference in routing quality" —
   is never defined, and the two quantities the harnesses compute in its place measure
   properties of the threshold rather than the quality of the routing.
2. The proposal names semantic entropy and a probe-based approximation as the gating
   signal. The project builds normalized token entropy, which is the right engineering
   call and is what §3's motivation actually argues for — but §4.1, §6, and §7 all
   describe the other thing.
3. The gate normalizes by `ln(V)` *and* calibrates by quantile. Each of those removes a
   source of cross-model difference, and RQ1 exists to measure cross-model difference.
   The one pilot run available returns perfect transfer agreement, which is close to what
   that design should be expected to produce.

Taken together, these mean the five-checkpoint GPU run could complete successfully, pass
every mechanical constraint check, and still not answer RQ1 or RQ2. That is the finding I
would want surfaced before the compute is spent rather than after.

The remaining issues are ordinary and fixable: one of three promised baselines is not
built, §5 has no statistical treatment, and the novelty sentence in §2.3 is contradicted
by prior art the reference list does not contain.

---

## Strengths

These are real and should survive revision intact.

- **The reporting rule is enforced in code, not just stated.** §5's commitment that
  "hallucination rate, task accuracy, and abstention rate are reported together" is
  backed by `assert_factuality_metrics_reported_together`, called from every harness's
  `write_results`. Most proposals that make this promise break it by the results section.
- **Exclusions are argued, not omitted.** TruthfulQA's exclusion (§5) is the strongest
  passage in the document: it identifies the regime where the mechanism is expected to
  fail, explains why a weak number there would be less informative than naming the
  boundary, and converts the limitation into the most direct extension of the work. The
  same quality applies to the reasoning-distilled and MoE exclusions in §4.2.
- **Autoformalization error is identified as the symbolic stream's real failure mode**
  (§4.3), and decomposition-first is justified as narrowing that surface rather than
  asserted as good practice. The VERGE differentiation is specific and fair.
- **Scope discipline is genuine.** §6's framing — three credits, five checkpoints, clean
  over broad — is matched by the repo's scope-containment choices rather than contradicted
  by them.
- **Reproducibility infrastructure exceeds what the proposal claims.** Pinned revision
  SHAs recorded per result, mechanical quantization enforcement, committed split index
  lists with a leakage guard, and one-command reproduction with a failing preflight.
- **The reference notes already disclose two citation subtleties** — [15]'s author-version
  discrepancy and [16]'s v0.1-vs-Instruct-v0.3 mismatch — correctly and unprompted.
- **One calibration result was rejected for landing on its sweep boundary and re-run**
  with an extended range. That instinct is worth a sentence in the final paper's methods.

---

## Critical Issues

### C1 · The primary evidence for RQ1 and RQ2 is an undefined metric, and its two implementations do not measure routing quality

§5 states: *"The difference in routing quality between the transferred fixed threshold and
the self-adaptive gate operating natively on each model constitutes the primary evidence
for RQ1 and RQ2."*

"Routing quality" is not defined anywhere in the proposal. §5's own metric list is
hallucination reduction, task accuracy, and latency — none of which is routing quality.
The two harnesses supply two different stand-ins, neither stated in the document:

- **RQ1** computes `transfer_agreement_rate`: the fraction of evaluation examples on which
  the transferred and native gates make the same decision
  (`transfer_threshold_halueval.py:92`). This is **concordance**, not quality. If both
  gates route the same 15% of examples, agreement is 1.0 whether or not those were the
  right examples to route.
- **RQ2** operationalizes it as **calibration fidelity**:
  `abs(observed_routing_rate - (1 - quantile))` (`adaptive_threshold_halueval.py:104,107`).
  This measures whether a threshold hits its own target firing rate on held-out data. A
  gate that fires on exactly 10% of examples *chosen at random* scores perfectly.

Neither quantity connects a routing decision to a factuality outcome. As implemented, RQ1
and RQ2 can both resolve cleanly while providing no evidence that routing catches
hallucinations — which is the claim the paper exists to make.

**What routing quality has to mean for the claim to hold.** The gate's job is to fire on
examples that would otherwise be hallucinated and not fire on examples that would be
answered correctly. That is a detection problem with available labels: HaluEval supplies
`right_answer`/`hallucinated_answer` per item, and the NLI judge already produces a
per-example factuality verdict. Routing quality should therefore be a
detection-performance measure of the gate against the ungated model's own correctness —
precision and recall of routed-vs-hallucinated, or AUROC over the entropy score, with the
routing rate reported alongside. Agreement and calibration fidelity are then useful
secondary diagnostics, not the primary evidence.

This is the single change most likely to decide whether the paper has a result.

### C2 · The document specifies a gating signal the project is not building, and §3's motivation argues for the signal the project *is* building

Three sections name semantic entropy or a probe approximation as the gate's signal:

- §4.1: *"SENSE computes a per-generation uncertainty estimate — semantic entropy, or a
  probe-based approximation where latency matters."*
- §6: *"The probe-based approximation of [4] ... is therefore evaluated as the production
  configuration, with full sampled semantic entropy retained as an accuracy ceiling."*
- §7, Phase 1: *"...choosing between semantic entropy and a probe-based approximation."*

The repo decided otherwise on 2026-07-26 and `CLAUDE.md` records it as locked: *"**Uncertainty
signal: token entropy.** The sole online gating signal ... semantic entropy itself stays
out of the online gate; if used at all, it's a single offline correlation study, not part
of the pipeline."* `services/neural/src/sense_neural/entropy.py` computes normalized
Shannon entropy over the next-token distribution and nothing else. No multi-sample path
and no hidden-state probe exist in the repository.

The repo is right and the document is wrong — but the inconsistency is not only a stale
paragraph, because it propagates:

- **§6's entire compute argument is about the wrong signal.** *"The principal cost driver
  is the uncertainty estimate itself: semantic entropy as formulated in [3] requires
  multiple sampled generations per query."* Token entropy is a logarithm over a
  distribution the forward pass already produced. The real cost driver is the symbolic
  round-trip multiplied by the routing rate — which is a far better feasibility argument
  and the one RQ3 is built to measure.
- **§3's premise only works for token entropy.** *"Entropy is not directly comparable
  across models"* is a vocabulary-and-tokenizer argument. Semantic entropy is computed
  over meaning-equivalence clusters and is vocabulary-independent, so under the signal
  §4.1 names, the cross-model incomparability that motivates the entire project largely
  dissolves. The proposal currently motivates itself with token entropy and then specifies
  semantic entropy.
- **§2.1's framing follows from it.** Semantic entropy [3] and probes [4] are introduced as
  the project's signal. They should be introduced as the expensive alternative that
  motivates the choice of token entropy, with the correlation study named as optional
  future work.

### C3 · Normalization and quantile calibration together may make a negative RQ1 result unreachable

`CLAUDE.md` constraint #4 reads: *"Normalize by `ln(V)` **or** compare quantiles."* The
implementation does both:

1. `entropy.py:38-40` divides every entropy value by `ln(vocab_size)` and appends only the
   normalized value — **the raw entropy is computed and discarded.**
2. `configs/gate.yaml: quantile: 0.9` — each model's threshold is the 0.9 quantile of its
   own calibration distribution, so every gate routes its own top-10% most uncertain
   examples by construction.

Step 1 removes the vocabulary-size difference that §3 identifies as the reason thresholds
should not transfer. Step 2 removes any residual difference in distribution location. RQ1
then asks whether a threshold transfers across models after both sources of non-transfer
have been engineered out.

The one existing cross-model run behaves accordingly:
`results/rq1_transfer_truthful_qa_llama3_to_mistral.json` reports thresholds of 0.28134
and 0.28204, identical routing rates of 0.15337, and `transfer_agreement_rate: 1.0`.

That run is from the retired scope — TruthfulQA, base rather than Instruct checkpoints,
two models — so it is a pilot and not a result. But it is the only cross-model evidence in
existence, it points away from RQ1's premise, and the mechanism above predicts it.

Three consequences for the design:

- **Raw entropy must be retained.** `entropy.py` should record both raw and normalized
  values. The un-normalized cross-model comparison is the one §3's premise is about, and
  it currently cannot be reconstructed from any run.
- **RQ1 needs an absolute-threshold condition.** "Fixed threshold" should mean a threshold
  transferred as a number on the raw scale — the naive thing a practitioner would do.
  Transferring a *normalized* threshold between quantile-calibrated gates is a much weaker
  notion of fixed, and it is the one currently implemented.
- **The normalization result is itself publishable.** If `ln(V)` normalization is what
  makes thresholds portable, that is a clean, useful finding and a legitimate answer to
  RQ1 — but it is a different paper from the one §3 sets up, and it needs to be framed as
  the contribution rather than discovered in the discussion section.

---

## Major Issues

### M1 · The novelty sentence is contradicted by prior art absent from the reference list

§2.3 concludes: *"What distinguishes SENSE from this prior work is that the symbolic
component is neither always on, as in retrieval, nor applied after the fact, as in post-hoc
verification, but invoked selectively during generation on the basis of the model's own
uncertainty."*

Selective, uncertainty-triggered intervention during generation is an established line of
work, and none of it is cited:

- **AdaDec** (arXiv:2506.08980, 2025) pauses decoding when token-level Shannon entropy
  exceeds a **learned, model-specific** threshold, and reports explicitly that a fixed
  threshold does not generalize across models — demonstrated on eight checkpoints from
  0.6B to 8B. This is the closest published analogue to SENSE's gate and it pre-empts part
  of RQ1's premise. It must be cited and differentiated.
- **Varshney et al.** (arXiv:2307.03987, 2023), "A stitch in time saves nine," detects
  low-confidence spans during generation and validates only those — the selective-
  verification idea itself, predating the semantic-entropy line.
- **UnCert-CoT** (arXiv:2503.15341, 2025) triggers expensive multi-path reasoning on a
  **fixed** uncertainty threshold, which is precisely the naive-gating baseline RQ1 is
  built against.

The defensible residual claim is narrower and should be stated in these terms: an adaptive
entropy gate whose output is a route to a **symbolic verification stream** (not reranking,
not detection, not always-on constraint), evaluated for **cross-family and cross-scale
transfer of the routing policy** as the primary question, in **open-domain factuality**
rather than code generation. That is still a contribution. It is not the contribution the
current sentence claims.

Structurally, the writing guide designates §2.5 as the place novelty is defended — *"why
prior work doesn't do inference-time entropy-gated switching"* — and the proposal has no
§2.5. The argument is collapsed into one sentence at the end of §2.3. Restore the
subsection.

### M2 · The post-hoc verification baseline is promised but not implemented

§5: *"SENSE will be benchmarked against an ungated base model, retrieval-augmented
generation [11], and post-hoc verification [12], [13]."* The writing guide §4.3 repeats it.

`experiments/` contains `rag_baseline_halueval.py` and no post-hoc baseline. A recursive
search for `selfcheck`, `SelfCheck`, `chain_of_verification`, `chain-of-verification`, and
`CoVe` across `*.py`, `*.yaml`, and `*.md` returns no matches anywhere in the repo.

This is the most load-bearing of the three baselines. "Selectively during generation" is a
claim *relative to* after-the-fact checking; without a post-hoc comparison the central
framing has no quantitative support. Either build it — SelfCheckGPT against the existing
NLI judge is the cheaper of the two — or amend §5 and the contributions list to promise
base + RAG only, and say why.

### M3 · §5 contains no statistical treatment

The writing guide §4.6 requires *"Multiple seeds, report variance, significance where the
sample size supports it."* §5 specifies no seeds, no repeated runs, no variance, no
intervals, and no significance testing.

RQ1 and RQ2 both rest on a *difference between two numbers*. Without a variance estimate
that difference cannot be defended, and a committee reader will ask. Separately,
`CLAUDE.md`'s own reproducibility requirement — *"Fixed seeds, recorded per run"* — is
unmet: `seed` appears once in the entire experiment and config surface
(`configs/dataset.yaml:10`, governing split construction), and no harness records a
per-run generation seed. Greedy decoding makes this low-impact today, but any sampled
condition would silently become irreproducible.

Add a paragraph to §5 specifying how many evaluation runs per condition, what is varied
between them, and what dispersion statistic is reported.

### M4 · The symbolic stream's coverage limits are documented in the repo but not in the proposal

§4.3 describes claims being *"checked against a domain constraint set using an SMT
solver."* Two material facts are recorded in the repo and absent from the document:

- The constraint KB is deliberately small, and **most FActScore entities do not resolve
  against it**; non-resolving entities are recorded as abstained (`CLAUDE.md`, current
  status). The abstention discipline is exactly right. The coverage rate is a limitation
  the proposal should name.
- **RQ3 does not verify generated content at all.** `configs/rq3.yaml` routes a fixed
  always-resolvable probe triple (`Albert Einstein / P106 / physicist`) so that latency is
  measured against a real backend round-trip, with the comment *"Do not read
  symbolic_result content from this harness as a factuality signal."* The reasoning is
  sound and documented. But §5 implies RQ3 exercises verification of the generation, and a
  reader comparing the paper to the code will find otherwise.

Both belong in §4.3 and in the limitations section, stated as scope containment — which is
what they are.

### M5 · A retired claim still drives an argument in §5

§5 ends: *"Latency instrumentation will be built into the pipeline from the outset, since
the real-time claim depends entirely on those measurements."*

The real-time claim was retired with the title on 2026-08-10. `CLAUDE.md` is explicit:
*"the title no longer promises real-time ... so latency is reported as a **result**, not
asserted upfront."* The replacement argument is stronger and already written in the repo —
an unsupported trade-off claim is exactly as bad as an unsupported real-time claim, so the
instrumentation matters more, not less. Port that sentence into §5.

The same stale phrasing survives at §6 (*"difficult to reconcile with a real-time claim"*)
and in the writing guide at §A.3 and §C.2. Sweep all four.

---

## Minor Issues

- **m1 · [1] publication status is stale.** Listed as *"accepted, ACM Trans. Inf. Syst."*;
  it has appeared — vol. 43, no. 2, pp. 1–55, 2025. Confirm against the ACM DL record and
  update. `Verification: PARTIAL` (publisher record not fetched this run).
- **m2 · [10] lacks author initials** — *"Zhao, Zhou, Yang, Qin, and Zhou"* — inconsistent
  with every other entry and with IEEE style.
- **m3 · [22] and [23] are out of citation order.** Self-flagged in the reference notes;
  the renumbering pass is still outstanding and the writing guide §A.2 requires order of
  first appearance.
- **m4 · Kuhn, Gal & Farquhar (ICLR 2023) is missing.** [3] (Nature 2024) is cited as the
  source of semantic entropy; the formal definition is in the ICLR paper. Citing only the
  Nature version reads as having entered the field through the headline.
- **m5 · RQ wordings differ across the three documents.** Proposal RQ1 adds *"and across
  parameter scales within a family"*; `CLAUDE.md` and the writing guide do not. Writing
  guide RQ3 still says *"on both models"* (two-model vintage) against the proposal's
  *"across all evaluated models."* `CLAUDE.md`'s assertion that *"RQ1–RQ3 already matched
  verbatim"* is incorrect. Pick the proposal's wording and propagate.
- **m6 · §4.1 and §5 disagree on calibration-set size.** §4.1 says the gate calibrates
  *"from a small calibration set"*; §5 justifies HaluEval specifically on *"sufficient
  volume to support a held-out calibration split per checkpoint."* Resolve to the actual
  split size and state it.
- **m7 · §6's hardware claim and the setup doc disagree.** §6 claims the study fits *"a
  single 16 GB accelerator"*; `docs/gpu-environment-setup.md` recommends an A100 40GB and
  calls a 24GB 4090 *"tighter,"* and still references bf16 and a two-model lineup from the
  retired scope. The doc needs re-pinning to 4-bit and five checkpoints, and §6's 16GB
  figure needs to be either verified against a real load or softened.
- **m8 · Table I states a hypothesis as a property.** Qwen3 1.7B's role reads *"the point
  at which threshold transfer is most likely to fail."* That is a prediction the study is
  designed to test; mark it as such so a reader cannot read it as an established result.
- **m9 · The AI Use Statement is an open item and the log has not been started.** §8.2 asks
  whether the thesis requirement applies to the Starred Paper. The writing guide's answer
  is to *"keep a running log **now**."* Whichever way the requirement resolves, the log is
  far cheaper to append to weekly than to reconstruct at deposit time, and this project's
  implementation has been substantially AI-assisted.

---

## Reproducibility and Verification

**Verified present:**

| Requirement | Status |
|---|---|
| Model revisions pinned by SHA | ✓ recorded in result files |
| Quantization held constant | ✓ `assert_pinned_gpu_quantization`, called by every GPU harness |
| Three-metric reporting enforced | ✓ `assert_factuality_metrics_reported_together` |
| Split discipline, committed index lists | ✓ `data/splits/*.json`, leakage guard present |
| One-command reproduction | ✓ `experiments/run_gpu_experiments.sh`, with failing preflight |
| Per-run seed | ✗ see M3 — one split seed only |
| Variance / repeated runs | ✗ see M3 |

**`Verification: BLOCKED`** — distinct from the weaknesses above; these are checks this
review could not complete, not findings:

- The test suite was not executed. *"All four required correctness checks pass"* is the
  repo's own claim and is unverified here.
- `services/symbolic` source was not read beyond grep; the Z3 round-trip and the
  decomposition front-end are unverified.
- No HaluEval or five-checkpoint results exist, so every empirical claim about the
  reconciled scope is unverifiable by construction. The 10 files in `results/` are CPU
  smoke runs and retired-scope TruthfulQA pilots.
- Reference [1]'s publisher record was not fetched.
- Proposal v4 was not compared against v5; the description of v5's changes is taken from
  `CLAUDE.md`.

---

## Inline Annotations

| Location | Claim | Annotation |
|---|---|---|
| Abstract | *"Because entropy distributions differ across model families and scales, a fixed gating threshold is unlikely to generalize"* | Premise. The one pilot run contradicts it, and C3 explains why the design may have removed the effect before measuring it. Do not let this sentence reach the final paper unqualified by the RQ1 result. |
| §2.1 | Semantic entropy [3] and probes [4] introduced as the project's signal | Per C2, reframe as the expensive alternative that motivates token entropy. Add Kuhn et al. ICLR 2023 (m4). |
| §2.3, final sentence | The novelty claim | M1. Contradicted by AdaDec, Varshney et al., UnCert-CoT. Narrow the claim and cite them. |
| §2 (absent) | No §2.5 | M1. The writing guide designates §2.5 as where novelty is defended. Restore it. |
| §3, RQ1 | *"and across parameter scales within a family"* | m5. Not present in the other two documents. Propagate this wording. |
| §3, premise | *"Entropy is not directly comparable across models"* | C2. True for token entropy, largely false for semantic entropy. The premise and §4.1's signal are inconsistent. |
| §4.1 | *"semantic entropy, or a probe-based approximation"* | C2. The repo builds normalized token entropy. Rewrite. |
| §4.1 | *"from a small calibration set"* | m6. Contradicts §5's volume rationale. |
| §4.3 | *"checked against a domain constraint set"* | M4. Disclose the KB's coverage rate and the abstain-on-non-resolution behavior. |
| Table I, Qwen3 1.7B | *"the point at which threshold transfer is most likely to fail"* | m8. Mark as hypothesis. |
| §5 | *"the difference in routing quality ... constitutes the primary evidence for RQ1 and RQ2"* | **C1.** Undefined term carrying the paper's central claim. Define as gate detection performance against ungated correctness. |
| §5 | *"benchmarked against ... post-hoc verification [12], [13]"* | M2. Not implemented. |
| §5 | No statistical treatment | M3. |
| §5, final sentence | *"since the real-time claim depends entirely on those measurements"* | M5. Claim retired 2026-08-10. |
| §6 | *"The principal cost driver is the uncertainty estimate itself"* | C2. False for token entropy. The driver is symbolic round-trip × routing rate. |
| §6 | *"a single 16 GB accelerator"* | m7. Disagrees with the repo's own setup doc. |
| §7, Phase 1 | *"choosing between semantic entropy and a probe-based approximation"* | C2. Decided on 2026-07-26, and neither option was chosen. |
| §8.2 | AI Use Statement open | m9. Start the log regardless of how the requirement resolves. |
| References, notes | [15] and [16] discrepancies disclosed | Correct and commendable. No action. |

---

## Recommendation

**Revise and proceed — but resolve C1 and C3 before the GPU run, not before the next draft.**

The proposal is approved and the implementation is sound engineering. The risk is not that
the work fails; it is that the five-checkpoint run completes, passes every mechanical
check, and produces numbers that do not bear on RQ1 or RQ2. C1 and C3 are both cheap to
fix now and expensive to fix after the compute is spent.

### Revision plan, in dependency order

**Before the GPU run — design changes (est. 1–2 days):**

1. **Define routing quality operationally and implement it** (C1). Gate detection
   performance against the ungated model's own correctness: precision and recall of
   routed-vs-hallucinated, plus AUROC over the entropy score, with routing rate reported
   alongside. HaluEval's `right_answer`/`hallucinated_answer` pairs and the existing NLI
   judge supply the labels; no new annotation is required. Keep `transfer_agreement_rate`
   and `calibration_fidelity_gap` as secondary diagnostics.
2. **Retain raw entropy** (C3). `entropy.py` records both raw and normalized; results files
   carry both distributions.
3. **Add an absolute-threshold transfer condition to RQ1** (C3), so "fixed threshold" means
   a raw-scale number transferred unchanged — the naive practitioner behavior the RQ is
   about. Keep the normalized-threshold condition as the second arm.
4. **Run the entropy-characterization diagnostic on two real checkpoints** before the full
   suite: raw and normalized distributions side by side, roughly an hour of GPU time. If
   normalization is what makes thresholds portable, that reframes the paper's contribution
   and you want to know before Phase 4, not during it.

**Before the next document revision (est. 1 day):**

5. **Rewrite §4.1, §6, and §7 Phase 1 for token entropy** (C2), and reframe §2.1
   accordingly. Replace §6's cost argument with symbolic round-trip × routing rate.
6. **Restore §2.5 and rewrite the novelty claim** (M1), citing AdaDec, Varshney et al., and
   UnCert-CoT as [24]–[26] and stating the narrowed residual contribution.
7. **Decide M2 explicitly** — build SelfCheckGPT against the existing NLI judge, or amend
   §5 and the contributions list to base + RAG with a stated reason.
8. **Add the statistical-treatment paragraph to §5** (M3) and record a per-run seed in
   `_common.py`.
9. **Disclose the symbolic backend's coverage limits in §4.3 and RQ3's probe design**
   (M4).
10. **Sweep the four stale "real-time" references** (M5) across the proposal and writing
    guide.
11. **Minor pass:** m1–m9, including the reference renumbering the notes already flag.

**Independent of the above:** start the AI Use Statement log this week (m9), and keep the
`gpu-environment-setup.md` update (m7) in the same commit as the §6 hardware claim so the
two cannot drift again.

### What not to change

The TruthfulQA exclusion argument, the autoformalization-error reasoning in §4.3, the
abstention reporting rule, the exclusion of reasoning-distilled and MoE models, and the
scope-containment posture throughout. These are the document's strongest material and all
four would be weakened by hedging.

---

## Sources

**Artifact and project documents**
- `SENSE_starred_paper_proposal_v5.docx` — claude.ai Project docs store, read 2026-10-04
- `SENSE_starred_paper_writing_guide.md` — same, read 2026-10-04

**Implementation repo** (`/Users/udaysah/projects/sense` @ `14ecd46`)
- `CLAUDE.md`; `experiments/transfer_threshold_halueval.py`;
  `experiments/adaptive_threshold_halueval.py`; `experiments/_common.py`;
  `services/neural/src/sense_neural/entropy.py`; `configs/gate.yaml`, `configs/rq2.yaml`,
  `configs/rq3.yaml`, `configs/dataset.yaml`;
  `results/rq1_transfer_truthful_qa_llama3_to_mistral.json`; `results/` listing;
  `experiments/` listing; `docs/gpu-environment-setup.md`;
  `experiments/run_gpu_experiments.sh`; `git log`

**Prior art cited in M1 and m4** — identified during this project's literature work;
metadata verified to the level noted, and each requires a direct check before entering a
submitted draft
- AdaDec — K. He et al., "Towards better code generation: Adaptive decoding with
  uncertainty guidance," arXiv:2506.08980, 2025. Full text retrieved and read.
- N. Varshney, W. Yao, H. Zhang, J. Chen, D. Yu, "A stitch in time saves nine: Detecting
  and mitigating hallucinations of LLMs by validating low-confidence generation,"
  arXiv:2307.03987, 2023. `Verification: PARTIAL` — identified via secondary citation.
- Y. Zhu et al., "Uncertainty-guided chain-of-thought for code generation with LLMs,"
  arXiv:2503.15341, 2025. `Verification: PARTIAL` — identified via AdaDec's reference list.
- L. Kuhn, Y. Gal, S. Farquhar, "Semantic uncertainty: Linguistic invariances for
  uncertainty estimation in natural language generation," ICLR 2023.
  `Verification: PARTIAL` — identified via secondary citations.
