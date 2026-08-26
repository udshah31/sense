# Provenance: FActScore human-annotated labeled set

Used by `sense_data.factscore_labeled` to calibrate
`configs/nli_judge.yaml`'s FActScore-shaped thresholds
(`experiments/calibrate_factscore_thresholds.py`). See that module's and that
script's docstrings for how and why.

## `InstructGPT.jsonl`, `ChatGPT.jsonl`, `PerplexityAI.jsonl`

- **Source**: [shmsw25/FActScore](https://github.com/shmsw25/FActScore) (Min
  et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in
  Long Form Text Generation," EMNLP 2023, arXiv:2305.14251). MIT-licensed.
- **Fetched**: 2026-08-24, from the paper's released Google Drive folder
  (linked from the repo's README), file `data.zip`.
- **Integrity**: `data.zip` sha256
  `404482c9a7d588d581a2f7a78d609fe6bfec4ddc0f729c8560a61b1743dbffab`. Not a
  pinned HF Hub revision (unlike this project's other data loaders) — no such
  pin exists for this release, so the extracted files are committed directly
  instead, matching the reproducibility principle already applied to this
  repo's split index lists (`data/splits/*.json`: committed, not re-fetched at
  run time).
- **Content**: 183 Wikipedia entities x 3 real commercial-LLM biography
  generations (InstructGPT, ChatGPT, PerplexityAI, all circa 2023), each
  sentence-decomposed into atomic facts and hand-labeled by human annotators
  as `S` (Supported), `NS` (Not Supported), or `IR` (Irrelevant) against
  Wikipedia. 549 rows total (183 x 3); 44 rows have `annotations: null`
  (empty generation — the model declined or failed to answer for that
  topic).
- **Disjoint from this project's own FActScore data**: zero entity overlap
  with `sense_data.factscore`'s 500-entity `dskar/FActScore` set (confirmed
  at fixture-generation time) — the original paper split entities into this
  183-entity human-labeled set (its Section 3) and a separate 500-entity
  unlabeled set scored only by automatic verifiers (its Section 4.3, the set
  this project uses for RQ1-3/RAG/symbolic-verification).

## `wikipedia_reference_text.json`, `wikipedia_fetch_failures.json`

- **Source**: public Wikipedia MediaWiki API
  (`en.wikipedia.org/w/api.php?action=query&prop=extracts`), plain-text
  article extracts.
- **Fetched**: 2026-08-24, via
  `data/scripts/fetch_factscore_labeled_reference_text.py`.
- **Honest limitation**: this is a 2026 snapshot, not the 2023-vintage
  Wikipedia text (`enwiki-20230401.db`, ~20GB, not fetched or committed here)
  the original human annotators actually verified against. An entity's
  article can have drifted between those dates. A handful of topics also
  failed to resolve entirely (ambiguous/renamed page titles) and are listed
  in `wikipedia_fetch_failures.json` — `factscore_labeled.py`'s loader
  returns `reference_text=None` for those, and the calibration script
  excludes them.
