# results/

Saved run outputs, flat by design (the harnesses write here and the derive scripts glob these
names). Pattern: `<rq>_<...>_<dataset>_<...>.json`; `*.console_summary.json` are the condensed
headline numbers, the same-named file without it holds the full result with per-example rows.

| Files | What |
|---|---|
| `rq1_transfer_halueval_*.console_summary.json` | RQ1, five pairs, **development** split (1,000 examples) |
| `rq1_transfer_test_halueval_*.json` and `.console_summary.json` | RQ1, five pairs, **test** split (first 1,000 committed test indices), one completed run |
| `rq2_from_rq1.json`, `rq2_from_rq1_test.json` | RQ2 derived from the RQ1 files above (`experiments/derive_rq2_from_rq1.py [test]`) |
| `rq3_accuracy_latency_halueval.console_summary.json` | RQ3, five checkpoints, 300 development examples |
| `selfcheck_baseline_halueval_{llama3,qwen3_1_7b}.json`, `selfcheck_vs_entropy_gate.json` | SelfCheckGPT-NLI baseline (300 development examples) and its comparison with the entropy gate |
| `retrieval_gated_halueval.console_summary.json` | Retrieval-gated smoke test, Qwen3-1.7B |
| `nli_short_answer_calibration.json`, `factscore_threshold_calibration.json` | Judge threshold calibrations |
| `run_metadata_<timestamp>.json` | Hardware and environment record for the test-split run |
| `*truthful_qa*` | **Legacy**: pilots from the retired TruthfulQA scope (and CPU tiny-model smoke tests); kept as the record, not results |

Reading guards for every number here are in `CLAUDE.md` ("2026-10-13 — experiments complete").
Development-split and test-split files must never be mixed in one table without saying so.
