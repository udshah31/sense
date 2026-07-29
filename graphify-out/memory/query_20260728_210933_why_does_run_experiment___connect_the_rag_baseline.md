---
type: "query"
date: "2026-07-28T21:09:33.186225+00:00"
question: "Why does run_experiment() connect the RAG Baseline Experiment Script to Experiment Results, TruthfulQA Data Loading, RQ1/RQ2 Threshold Harnesses, RQ3, RAG Prompt Budget Truncation, FAISS Passage Index, and Project Design Decisions?"
contributor: "graphify"
source_nodes: ["run_experiment()", "load_truthful_qa()", "load_splits()", "build_prompt_within_budget()", "retrieve()", "Shared experiment-harness pattern (config→model→per-example loop→results JSON)", "RAG Baseline Implementation Plan"]
---

# Q: Why does run_experiment() connect the RAG Baseline Experiment Script to Experiment Results, TruthfulQA Data Loading, RQ1/RQ2 Threshold Harnesses, RQ3, RAG Prompt Budget Truncation, FAISS Passage Index, and Project Design Decisions?

## Answer

Traced why run_experiment() (experiments/rag_baseline_truthful_qa.py) bridges 7 communities: it calls load_truthful_qa()/load_splits() (shared TruthfulQA split loading, same as RQ1-RQ3), build_prompt_within_budget() and retrieve() (RAG-specific prompt truncation and FAISS retrieval), is semantically_similar_to the shared experiment-harness pattern (config->model->loop->results JSON, INFERRED 0.8-0.9, no direct call - all 4 harnesses independently converged on this shape), and is referenced (EXTRACTED) by the RAG Baseline Implementation Plan and RAG baseline doc entry design docs. Two regression tests (overlong-prompt crash, end-to-end fixture) call into it directly - born from the final-review truncation bug fix.

## Source Nodes

- run_experiment()
- load_truthful_qa()
- load_splits()
- build_prompt_within_budget()
- retrieve()
- Shared experiment-harness pattern (config→model→per-example loop→results JSON)
- RAG Baseline Implementation Plan