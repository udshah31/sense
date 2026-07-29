# Graph Report - .  (2026-07-28)

## Corpus Check
- 81 files · ~126,326 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 514 nodes · 848 edges · 37 communities (33 shown, 4 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 89 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Entropy Gate Policy|Entropy Gate Policy]]
- [[_COMMUNITY_RAG Baseline Implementation|RAG Baseline Implementation]]
- [[_COMMUNITY_Experiment Results & Split Artifacts|Experiment Results & Split Artifacts]]
- [[_COMMUNITY_Token Entropy Monitor|Token Entropy Monitor]]
- [[_COMMUNITY_Symbolic Verification HTTP API|Symbolic Verification HTTP API]]
- [[_COMMUNITY_RQ3 Accuracy-Latency Harness|RQ3 Accuracy-Latency Harness]]
- [[_COMMUNITY_Truncated Entropy Bounds Check|Truncated Entropy Bounds Check]]
- [[_COMMUNITY_RAG Baseline Result Schema|RAG Baseline Result Schema]]
- [[_COMMUNITY_RQ2 Result Schema|RQ2 Result Schema]]
- [[_COMMUNITY_TruthfulQA Data Loading|TruthfulQA Data Loading]]
- [[_COMMUNITY_RQ1 Result Schema|RQ1 Result Schema]]
- [[_COMMUNITY_Wikipedia Corpus Build Script|Wikipedia Corpus Build Script]]
- [[_COMMUNITY_Gate Calibration Result Schema|Gate Calibration Result Schema]]
- [[_COMMUNITY_RQ1 Threshold Transfer Harness|RQ1 Threshold Transfer Harness]]
- [[_COMMUNITY_Factuality Eval Package|Factuality Eval Package]]
- [[_COMMUNITY_Gate Calibration Harness|Gate Calibration Harness]]
- [[_COMMUNITY_Shared Experiment-Harness Helpers (refactor)|Shared Experiment-Harness Helpers (refactor)]]
- [[_COMMUNITY_RQ2 Adaptive Threshold Harness|RQ2 Adaptive Threshold Harness]]
- [[_COMMUNITY_Experiment Config Files|Experiment Config Files]]
- [[_COMMUNITY_RQ2 Harness Test Suite|RQ2 Harness Test Suite]]
- [[_COMMUNITY_Gate Calibration Test Suite|Gate Calibration Test Suite]]
- [[_COMMUNITY_GPU Model Configs (Unpinned)|GPU Model Configs (Unpinned)]]
- [[_COMMUNITY_TruthfulQA Benchmark Decision|TruthfulQA Benchmark Decision]]
- [[_COMMUNITY_Entropy Terminology|Entropy Terminology]]
- [[_COMMUNITY_Symbolic Backend Design Decision|Symbolic Backend Design Decision]]
- [[_COMMUNITY_Docker Compose Services|Docker Compose Services]]
- [[_COMMUNITY_Uncalibrated-Gate Correctness Check|Uncalibrated-Gate Correctness Check]]
- [[_COMMUNITY_Split Leakage Correctness Check|Split Leakage Correctness Check]]
- [[_COMMUNITY_Entropy Bounds Correctness Check|Entropy Bounds Correctness Check]]
- [[_COMMUNITY_Configs README|Configs README]]

## God Nodes (most connected - your core abstractions)
1. `load_truthful_qa` - 29 edges
2. `generate_splits()` - 23 edges
3. `GatePolicy` - 23 edges
4. `TokenEntropyMonitor` - 20 edges
5. `PassageIndex` - 18 edges
6. `route_and_annotate()` - 17 edges
7. `load_splits` - 17 edges
8. `run_experiment()` - 16 edges
9. `run_experiment()` - 16 edges
10. `run()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `int` --uses--> `SplitIndices`  [INFERRED]
  services/orchestrator/src/sense_orchestrator/gate.py → data/src/sense_data/splits.py
- `float` --uses--> `GatePolicy`  [INFERRED]
  experiments/adaptive_threshold_truthful_qa.py → services/orchestrator/src/sense_orchestrator/gate.py
- `Why run_experiment() bridges 7 communities (saved query answer)` --references--> `run_experiment()`  [EXTRACTED]
  graphify-out/memory/query_20260728_210933_why_does_run_experiment___connect_the_rag_baseline.md → experiments/rag_baseline_truthful_qa.py
- `test_monitor_is_noop_on_generation()` --implements--> `Correctness Check: Entropy Monitor No-Op on Generation`  [EXTRACTED]
  services/neural/tests/test_entropy_monitor.py → CLAUDE.md
- `test_monitor_attached_during_latency_measurement_is_still_noop()` --implements--> `Correctness Check: Entropy Monitor No-Op on Generation`  [INFERRED]
  services/neural/tests/test_latency.py → CLAUDE.md

## Import Cycles
- 2-file cycle: `services/orchestrator/src/sense_orchestrator/router.py -> services/orchestrator/tests/test_router.py -> services/orchestrator/src/sense_orchestrator/router.py`

## Hyperedges (group relationships)
- **Five experiment harnesses share load_examples_and_splits/write_results boilerplate extracted into _common.py** — experiments__common_load_examples_and_splits, experiments__common_write_results, experiments_adaptive_threshold_truthful_qa_run, experiments_calibrate_gate_truthful_qa_run, experiments_rag_baseline_truthful_qa_run_experiment, experiments_rq3_accuracy_latency_truthful_qa_run_experiment, experiments_transfer_threshold_truthful_qa_run [EXTRACTED 0.95]

## Communities (37 total, 4 thin omitted)

### Community 0 - "Entropy Gate Policy"
Cohesion: 0.05
Nodes (43): Merge-back policy: annotate-only, Correctness check: gate refuses to run uncalibrated, Correctness check: split leakage test, Exception, GatePolicy, _quantile(), Entropy-gating policy: routes generation to the symbolic stream when token entro, Linear-interpolation quantile, no numpy dependency. (+35 more)

### Community 1 - "RAG Baseline Implementation"
Cohesion: 0.05
Nodes (44): bool, Design Decision: Annotate-Only Merge-Back Policy, Design Decision: FAISS + all-MiniLM-L6-v2 RAG Baseline, RQ1: Fixed Threshold Transfer, RQ2: Self-Adaptive Threshold, RQ3: Accuracy-Latency Trade-off, RAG comparison baseline (capped effort), build_prompt() (+36 more)

### Community 2 - "Experiment Results & Split Artifacts"
Cohesion: 0.08
Nodes (39): data/README.md, load_splits, float, int, Path, Gate calibration result, cpu_test, threshold=0.99996, 327 calib examples, generate_truthful_qa_splits.py: main, RAG baseline result, cpu_test, factuality_accuracy_proxy=0.0, 327 test examples (+31 more)

### Community 3 - "Token Entropy Monitor"
Cohesion: 0.06
Nodes (31): Correctness Check: Entropy Monitor No-Op on Generation, Latency instrumentation: TTFT and mean inter-token latency, FloatTensor, LogitsProcessor, LongTensor, tiny_model fixture, tiny_tokenizer fixture, exact_token_entropy() (+23 more)

### Community 4 - "Symbolic Verification HTTP API"
Cohesion: 0.10
Nodes (30): BaseModel, Response, EntityExistsRequest, EntityExistsResponse, Symbolic verification backend HTTP contract.  Two endpoints, both wrapping publi, VerifyTripleRequest, VerifyTripleResponse, Thin client for the two public Wikidata HTTP endpoints this backend uses: wbsear (+22 more)

### Community 5 - "RQ3 Accuracy-Latency Harness"
Cohesion: 0.09
Nodes (21): RQ3 harness: what accuracy-latency trade-off does routing introduce?  Starts the, dataset, decoding, do_sample, max_new_tokens, eval_split, factuality_accuracy_proxy, factuality_metric (+13 more)

### Community 6 - "Truncated Entropy Bounds Check"
Cohesion: 0.15
Nodes (18): Entropy normalization by ln(vocab_size) for cross-tokenizer comparability, Correctness check: truncated vs exact entropy bounds, Bounds on true token entropy derivable from a truncated top-k logprob response (, Return (lower, upper) bounds on the true Shannon entropy (nats), given only, truncated_entropy_bounds(), float, int, str (+10 more)

### Community 7 - "RAG Baseline Result Schema"
Cohesion: 0.10
Nodes (20): dataset, decoding, do_sample, max_new_tokens, eval_split, factuality_accuracy_proxy, factuality_metric, hf_repo (+12 more)

### Community 8 - "RQ2 Result Schema"
Cohesion: 0.11
Nodes (20): adaptive_calibration_fidelity_gap, adaptive_dev_routing_rate, adaptive_threshold, dataset, decoding, do_sample, max_new_tokens, eval_split (+12 more)

### Community 9 - "TruthfulQA Data Loading"
Cohesion: 0.16
Nodes (15): load_truthful_qa, TruthfulQA loader (generation config).  TruthfulQA ships only a single 817-examp, Load TruthfulQA in the project's fixed index order.      Index `i` here is the s, TruthfulQAExample, development, test, Integration test for the RQ3 harness on a tiny subset — the real script runs aga, The core RQ3 invariant: annotate-only merge-back must never change output. (+7 more)

### Community 10 - "RQ1 Result Schema"
Cohesion: 0.13
Nodes (18): dataset, decoding, do_sample, max_new_tokens, eval_split, n_eval_examples, native_routing_rate, quantile (+10 more)

### Community 11 - "Wikipedia Corpus Build Script"
Cohesion: 0.23
Nodes (13): Client, fetch_passage, main, Committed RAG passages.json corpus, One-time, offline corpus build: queries the Wikipedia REST API for each Truthful, Search Wikipedia for `question`, return the top hit's {title, text}     intro ex, Resolve each question to a passage, skipping (and logging) any that     don't re, str (+5 more)

### Community 12 - "Gate Calibration Result Schema"
Cohesion: 0.13
Nodes (14): dataset, decoding, do_sample, max_new_tokens, hf_repo, max_calibration_entropy, mean_calibration_entropy, model_name (+6 more)

### Community 13 - "RQ1 Threshold Transfer Harness"
Cohesion: 0.16
Nodes (10): RQ1: fixed threshold transfer, RQ2: self-adaptive threshold calibration fidelity, RQ1 harness doc entry, load_config(), RQ1 harness: does a fixed entropy-gating threshold, calibrated on one model, tra, run(), Integration test for the RQ1 transfer harness, on a small subset for speed — tra, small_splits() (+2 more)

### Community 14 - "Factuality Eval Package"
Cohesion: 0.21
Nodes (12): Eval Package README, str, Factuality proxy metric for TruthfulQA-style examples.  This is a simplified lex, correct" if the generated text contains the best/any correct answer,     "incorr, FactualityVerdict, lexical_containment_verdict, Factuality Metric Test Suite, test_case_insensitive() (+4 more)

### Community 15 - "Gate Calibration Harness"
Cohesion: 0.23
Nodes (11): load_config(), Wire the entropy-gating policy against real TruthfulQA calibration entropies.  F, run(), calibration_entropies(), load_model(), load_yaml_config(), mean_calibration_entropy(), float (+3 more)

### Community 16 - "Shared Experiment-Harness Helpers (refactor)"
Cohesion: 0.21
Nodes (11): Shared experiment-harness boilerplate extraction, load_examples_and_splits(), Write `result` as the full JSON at RESULTS_DIR/filename, then print a summary, write_results(), load_config(), main(), Why run_experiment() bridges 7 communities (saved query answer), test_load_examples_and_splits_returns_real_data() (+3 more)

### Community 17 - "RQ2 Adaptive Threshold Harness"
Cohesion: 0.25
Nodes (10): RQ3: accuracy-latency trade-off, annotate-only no-op invariant, Shared experiment-harness pattern (config→model→per-example loop→results JSON), load_config(), float, RQ2 harness: does a self-adaptive threshold — recalibrated on the target model's, routing_rate(), run(), RQ2 harness doc entry (+2 more)

### Community 18 - "Experiment Config Files"
Cohesion: 0.25
Nodes (9): Decoding Configuration Held Constant, Entropy Not Comparable Across Models Raw, Entropy-Gating Quantile Policy Config, Model Config: cpu_test (sshleifer/tiny-gpt2), Model Config: cpu_test_transfer_target (tiny-random-GPTNeoX), RAG Baseline Config, RQ1 Harness Config, RQ2 Harness Config (+1 more)

### Community 19 - "RQ2 Harness Test Suite"
Cohesion: 0.29
Nodes (4): Integration test for the RQ2 adaptive-threshold harness, on a small subset for s, small_splits(), test_adaptive_gate_calibrates_independently_of_fixed_gate(), test_routing_rate_helper_matches_manual_count()

### Community 20 - "Gate Calibration Test Suite"
Cohesion: 0.40
Nodes (3): Integration test for the real calibration wiring, on a small subset for speed —, test_calibrating_on_real_test_split_indices_raises(), test_gate_calibrates_on_a_real_truthful_qa_subset()

### Community 21 - "GPU Model Configs (Unpinned)"
Cohesion: 0.50
Nodes (4): Model Revisions Are Pinned, Quantization Held Constant, Model Config: llama3 (GPU, unpinned), Model Config: mistral (GPU, unpinned)

### Community 22 - "TruthfulQA Benchmark Decision"
Cohesion: 0.67
Nodes (3): Design Decision: TruthfulQA Benchmark, Data Split Discipline (calibration/development/test), TruthfulQA Dataset Config

### Community 23 - "Entropy Terminology"
Cohesion: 0.67
Nodes (3): Semantic Entropy, Token Entropy, Design Decision: Token Entropy as Sole Gating Signal

### Community 24 - "Symbolic Backend Design Decision"
Cohesion: 0.67
Nodes (3): Design Decision: SPARQL/Wikidata Symbolic Backend, RQ3 Fixed Symbolic Probe Triple (Einstein/P106/physicist), Symbolic Backend HTTP Endpoint Config

### Community 25 - "Docker Compose Services"
Cohesion: 0.67
Nodes (3): docker-compose: neural service, docker-compose: orchestrator service, docker-compose: symbolic service

## Knowledge Gaps
- **118 isolated node(s):** `development`, `test`, `int`, `float`, `str` (+113 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_experiment()` connect `RQ2 Adaptive Threshold Harness` to `Entropy Gate Policy`, `RAG Baseline Implementation`, `Experiment Results & Split Artifacts`, `Token Entropy Monitor`, `RQ3 Accuracy-Latency Harness`, `TruthfulQA Data Loading`, `Factuality Eval Package`, `Gate Calibration Harness`, `Shared Experiment-Harness Helpers (refactor)`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `run_experiment()` connect `RAG Baseline Implementation` to `Experiment Results & Split Artifacts`, `TruthfulQA Data Loading`, `Factuality Eval Package`, `Gate Calibration Harness`, `Shared Experiment-Harness Helpers (refactor)`, `RQ2 Adaptive Threshold Harness`?**
  _High betweenness centrality (0.102) - this node is a cross-community bridge._
- **Why does `TokenEntropyMonitor` connect `Token Entropy Monitor` to `Entropy Gate Policy`, `Truncated Entropy Bounds Check`, `TruthfulQA Data Loading`, `Gate Calibration Harness`, `RQ2 Adaptive Threshold Harness`?**
  _High betweenness centrality (0.093) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `GatePolicy` (e.g. with `float` and `GatePolicy`) actually correct?**
  _`GatePolicy` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `TokenEntropyMonitor` (e.g. with `float` and `str`) actually correct?**
  _`TokenEntropyMonitor` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `PassageIndex` (e.g. with `bool` and `str`) actually correct?**
  _`PassageIndex` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `One-time generation of the committed calibration/development/test split for Trut`, `development`, `test` to the rest of the system?**
  _184 weakly-connected nodes found - possible documentation gaps or missing edges._