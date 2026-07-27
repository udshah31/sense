# RAG Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a FAISS + `all-MiniLM-L6-v2` retrieval-augmented-generation baseline on the TruthfulQA test split (327 examples), scored with the project's existing lexical-containment factuality proxy, matching the RQ1–RQ3 CPU-validation setup (tiny-gpt2, same decoding config).

**Architecture:** A new `services/rag/` Python package (mirroring `services/symbolic/`'s layout) provides an in-memory FAISS retriever over a committed, pre-built passage corpus. A one-time offline script builds that corpus from the Wikipedia REST API, keyed to the test split's question text. A new `experiments/rag_baseline_truthful_qa.py` script (mirroring `rq3_accuracy_latency_truthful_qa.py`'s shape) retrieves top-k passages per question, prepends them as context, generates with tiny-gpt2, scores with `lexical_containment_verdict`, and writes a results JSON.

**Tech Stack:** Python 3.13, `sentence-transformers` (for `all-MiniLM-L6-v2` embeddings), `faiss-cpu`, `httpx` (Wikipedia REST calls), `pytest` + `pytest-asyncio`, `uv` for dependency management — same stack conventions as `services/symbolic/`.

## Global Constraints

- No latency instrumentation anywhere in this baseline (spec non-goal — RAG is accuracy-only).
- No interaction with the entropy gate, symbolic backend, or orchestrator.
- No live network calls during the scored experiment run — Wikipedia is called only by the one-time, offline `build_corpus.py` script; `retrieve()` and the experiment script never touch the network.
- Runs on the test split only (327 examples, seed 42, fixed index order from `data/splits/truthful_qa.json`), touched once for this baseline's reported number.
- Same decoding config as RQ1–RQ3 (`configs/gate.yaml`'s `decoding` block: `do_sample: false`, `max_new_tokens: 20`) and same model (`configs/model.yaml`'s `cpu_test` entry, `sshleifer/tiny-gpt2`).
- Top-k = 3 passages per query, prompt format is a plain `Context: ...\n\nQuestion: ...` block (no few-shot template).
- Config-driven — no parameters hard-coded into scripts (`configs/rag.yaml` holds model key, top_k, corpus_path, eval_split).
- All new code must be testable on CPU without live network access in the test suite (project-wide dev-environment rule).

---

### Task 1: `services/rag` package scaffold

**Files:**
- Create: `services/rag/pyproject.toml`
- Create: `services/rag/src/sense_rag/__init__.py`
- Create: `services/rag/.python-version`
- Create: `services/rag/README.md`

**Interfaces:**
- Produces: an installable `sense-rag` package other tasks build modules into.

- [ ] **Step 1: Create the package files**

`services/rag/.python-version`:
```
3.13.5
```

`services/rag/pyproject.toml`:
```toml
[project]
name = "sense-rag"
version = "0.1.0"
description = "RAG comparison baseline: FAISS + all-MiniLM-L6-v2 over a curated Wikipedia passage subset."
requires-python = ">=3.13.5"
dependencies = [
    "sentence-transformers==3.3.1",
    "faiss-cpu==1.9.0.post1",
    "httpx==0.28.1",
]

[dependency-groups]
dev = [
    "pytest==8.3.5",
    "pytest-asyncio==0.25.3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sense_rag"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

`services/rag/src/sense_rag/__init__.py`:
```python
```//(empty file)

`services/rag/README.md`:
```markdown
# RAG baseline

FAISS + `all-MiniLM-L6-v2` retrieval-augmented-generation comparison baseline
(see `../../CLAUDE.md`'s Phase-0 design decisions). This is a comparison
baseline, not the project's contribution — accuracy-only, no latency
instrumentation, no gate interaction.

## Modules

- `sense_rag.index` — builds an in-memory FAISS index from a committed
  passages file.
- `sense_rag.retrieve` — embeds a query and returns the top-k passage texts.

## Corpus

`data/rag_corpus/passages.json` is built once, offline, via
`scripts/build_corpus.py` (hits the Wikipedia REST API) and committed to the
repo. Neither `sense_rag.index` nor `sense_rag.retrieve` ever calls the
network — they only read the committed file.

## Setup

```
uv sync
```

## Tests

```
uv run pytest
```
```

- [ ] **Step 2: Run `uv sync` to generate the lockfile and venv**

Run: `cd services/rag && uv sync`
Expected: completes without error, creates `uv.lock` and `.venv/`.

- [ ] **Step 3: Commit**

```bash
git add services/rag/pyproject.toml services/rag/src/sense_rag/__init__.py \
    services/rag/.python-version services/rag/README.md services/rag/uv.lock
git commit -m "Scaffold sense-rag package"
```

---

### Task 2: Passage corpus build script

**Files:**
- Create: `services/rag/scripts/build_corpus.py`
- Create: `services/rag/tests/test_build_corpus.py`

**Interfaces:**
- Consumes: `sense_data.truthful_qa.load_truthful_qa() -> list[TruthfulQAExample]` (has `.question` field), `sense_data.splits.load_splits(path) -> SplitIndices` (has `.test: list[int]`).
- Produces: `fetch_passage(question: str, client: httpx.Client) -> dict | None` — returns `{"title": str, "text": str}` or `None` if no search hit. `build_corpus(questions: list[str], client: httpx.Client) -> list[dict]` — calls `fetch_passage` per question, skips (and logs) `None` results, returns the list of passage dicts. A `data/rag_corpus/passages.json` file (list of `{"title", "text"}` dicts) is the artifact later tasks depend on.

This script is run once, by hand, not part of any automated test suite's execution path (its output is committed). Its unit tests mock `httpx.Client` so no live network calls happen in CI.

- [ ] **Step 1: Write the failing tests**

`services/rag/tests/test_build_corpus.py`:
```python
import httpx
import pytest

from sense_rag_scripts.build_corpus import build_corpus, fetch_passage

WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"


def _mock_transport(response_json):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response_json)

    return httpx.MockTransport(handler)


def test_fetch_passage_returns_title_and_extract():
    response_json = {
        "query": {
            "pages": {
                "736": {
                    "title": "Albert Einstein",
                    "extract": "Albert Einstein was a theoretical physicist.",
                }
            }
        }
    }
    client = httpx.Client(transport=_mock_transport(response_json))

    passage = fetch_passage("Who was Albert Einstein?", client)

    assert passage == {
        "title": "Albert Einstein",
        "text": "Albert Einstein was a theoretical physicist.",
    }


def test_fetch_passage_returns_none_when_no_pages():
    response_json = {"query": {"pages": {}}}
    client = httpx.Client(transport=_mock_transport(response_json))

    passage = fetch_passage("asdlkfjasldkfj nonsense query", client)

    assert passage is None


def test_build_corpus_skips_unresolved_questions():
    response_json = {"query": {"pages": {}}}
    client = httpx.Client(transport=_mock_transport(response_json))

    passages = build_corpus(["a question with no hit"], client)

    assert passages == []


def test_build_corpus_collects_resolved_passages():
    response_json = {
        "query": {
            "pages": {
                "1": {"title": "Paris", "text_stub": "unused"},
            }
        }
    }
    # fetch_passage keys off "extract", not "text_stub" — this response has no
    # extract, so it should resolve to None and be skipped, proving build_corpus
    # tolerates a mix of resolved and unresolved questions in one run.
    client = httpx.Client(transport=_mock_transport(response_json))

    passages = build_corpus(["question one", "question two"], client)

    assert passages == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/rag && uv run pytest tests/test_build_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sense_rag_scripts'`

- [ ] **Step 3: Write the implementation**

`services/rag/scripts/build_corpus.py`:
```python
"""One-time, offline corpus build: queries the Wikipedia REST API for each
TruthfulQA test-split question and writes the resolved passages to
data/rag_corpus/passages.json, committed to the repo.

Not part of any automated test-suite execution path or experiment run path —
sense_rag.index and sense_rag.retrieve only ever read the committed output file.
Run by hand: `uv run python scripts/build_corpus.py`.
"""

import json
import sys
from pathlib import Path

import httpx

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_HEADERS = {"User-Agent": "sense-rag/0.1 (CSCI699 research project; no contact URL yet)"}

REPO_ROOT = Path(__file__).parent.parent.parent.parent
CORPUS_PATH = REPO_ROOT / "data" / "rag_corpus" / "passages.json"
SPLIT_PATH = REPO_ROOT / "data" / "splits" / "truthful_qa.json"


def fetch_passage(question: str, client: httpx.Client) -> dict | None:
    """Search Wikipedia for `question`, return the top hit's {title, text}
    intro extract, or None if nothing resolved."""
    response = client.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": question,
            "gsrlimit": 1,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
        },
        headers=REQUEST_HEADERS,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    if not pages:
        return None

    page = next(iter(pages.values()))
    extract = page.get("extract")
    if not extract:
        return None

    return {"title": page["title"], "text": extract}


def build_corpus(questions: list[str], client: httpx.Client) -> list[dict]:
    """Resolve each question to a passage, skipping (and logging) any that
    don't resolve. Returns the list of resolved {title, text} passages."""
    passages = []
    for question in questions:
        passage = fetch_passage(question, client)
        if passage is None:
            print(f"skipping question with no Wikipedia hit: {question!r}", file=sys.stderr)
            continue
        passages.append(passage)
    return passages


def main() -> None:
    sys.path.insert(0, str(REPO_ROOT / "data" / "src"))
    from sense_data.splits import load_splits
    from sense_data.truthful_qa import load_truthful_qa

    examples = load_truthful_qa()
    splits = load_splits(SPLIT_PATH)
    questions = [examples[i].question for i in splits.test]

    with httpx.Client(timeout=10.0) as client:
        passages = build_corpus(questions, client)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(json.dumps(passages, indent=2))
    print(f"wrote {len(passages)} passages (of {len(questions)} questions) to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
```

Since the test file imports `sense_rag_scripts.build_corpus`, add a package marker so `scripts/` is importable as `sense_rag_scripts` during tests:

`services/rag/scripts/__init__.py`: empty file.

Update `services/rag/pyproject.toml`'s wheel packages line to also include the scripts package for test discovery:
```toml
[tool.hatch.build.targets.wheel]
packages = ["src/sense_rag", "scripts"]
```
and rename the `scripts` directory's importable name — simplest fix: add `services/rag/pyproject.toml`'s test config to put `scripts` on the path. Use a `conftest.py` instead, which is simpler than repackaging:

`services/rag/tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
```

And change the test imports from `sense_rag_scripts.build_corpus` to `build_corpus` directly:

Edit `services/rag/tests/test_build_corpus.py`'s import line to:
```python
from build_corpus import build_corpus, fetch_passage
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/rag && uv run pytest tests/test_build_corpus.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add services/rag/scripts/build_corpus.py services/rag/tests/test_build_corpus.py \
    services/rag/tests/conftest.py
git commit -m "Add offline Wikipedia corpus build script for RAG baseline"
```

---

### Task 3: FAISS index module

**Files:**
- Create: `services/rag/src/sense_rag/index.py`
- Create: `services/rag/tests/test_index.py`

**Interfaces:**
- Consumes: a passages file on disk, list of `{"title": str, "text": str}` dicts (same shape Task 2 produces).
- Produces: `class PassageIndex` with:
  - `PassageIndex.from_passages(passages: list[dict]) -> "PassageIndex"` — builds the FAISS index in memory from a list of passage dicts.
  - `PassageIndex.from_file(path: Path) -> "PassageIndex"` — reads a JSON file of passages, then calls `from_passages`.
  - `.search(query: str, k: int) -> list[str]` — returns the top-k passage `text` strings, ranked most-similar first.
  - `.passages: list[dict]` — the underlying passage list, same order as embedded.

- [ ] **Step 1: Write the failing tests**

`services/rag/tests/test_index.py`:
```python
import json

from sense_rag.index import PassageIndex

PASSAGES = [
    {"title": "Albert Einstein", "text": "Albert Einstein was a theoretical physicist known for relativity."},
    {"title": "Paris", "text": "Paris is the capital city of France."},
    {"title": "Photosynthesis", "text": "Photosynthesis converts light energy into chemical energy in plants."},
]


def test_from_passages_search_returns_most_similar_text_first():
    index = PassageIndex.from_passages(PASSAGES)

    results = index.search("Who discovered the theory of relativity?", k=1)

    assert results == ["Albert Einstein was a theoretical physicist known for relativity."]


def test_search_respects_k():
    index = PassageIndex.from_passages(PASSAGES)

    results = index.search("science topics", k=2)

    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_from_file_loads_committed_passages_json(tmp_path):
    passages_path = tmp_path / "passages.json"
    passages_path.write_text(json.dumps(PASSAGES))

    index = PassageIndex.from_file(passages_path)

    assert index.passages == PASSAGES
    results = index.search("capital of France", k=1)
    assert results == ["Paris is the capital city of France."]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/rag && uv run pytest tests/test_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sense_rag.index'`

- [ ] **Step 3: Write the implementation**

`services/rag/src/sense_rag/index.py`:
```python
"""In-memory FAISS index over a committed passages file, embedded with
all-MiniLM-L6-v2. No server process, no network calls at query time — matches
CLAUDE.md's "in-memory vector store" decision for the RAG comparison baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class PassageIndex:
    def __init__(self, passages: list[dict], embeddings: np.ndarray, model: SentenceTransformer):
        self.passages = passages
        self._model = model
        self._faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
        self._faiss_index.add(embeddings)

    @classmethod
    def from_passages(cls, passages: list[dict]) -> "PassageIndex":
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        texts = [p["text"] for p in passages]
        embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return cls(passages, embeddings.astype(np.float32), model)

    @classmethod
    def from_file(cls, path: Path) -> "PassageIndex":
        passages = json.loads(Path(path).read_text())
        return cls.from_passages(passages)

    def search(self, query: str, k: int) -> list[str]:
        query_embedding = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        _, indices = self._faiss_index.search(query_embedding.astype(np.float32), k)
        return [self.passages[i]["text"] for i in indices[0] if i != -1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/rag && uv run pytest tests/test_index.py -v`
Expected: 3 passed
(Note: first run downloads the `all-MiniLM-L6-v2` model weights from Hugging Face — requires network access once, then cached locally. If this is a concern in a fully offline CI environment, flag it, but per this project's existing pattern (`load_model` in `experiments/_common.py`), model downloads on first use are the accepted norm.)

- [ ] **Step 5: Commit**

```bash
git add services/rag/src/sense_rag/index.py services/rag/tests/test_index.py
git commit -m "Add in-memory FAISS passage index for RAG baseline"
```

---

### Task 4: Retriever module

**Files:**
- Create: `services/rag/src/sense_rag/retrieve.py`
- Create: `services/rag/tests/test_retrieve.py`

**Interfaces:**
- Consumes: `PassageIndex` from Task 3 (`.search(query, k) -> list[str]`).
- Produces: `retrieve(index: PassageIndex, query: str, k: int = 3) -> list[str]` — thin wrapper, the public entry point the experiment script (Task 6) calls.

- [ ] **Step 1: Write the failing test**

`services/rag/tests/test_retrieve.py`:
```python
from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve

PASSAGES = [
    {"title": "Albert Einstein", "text": "Albert Einstein was a theoretical physicist known for relativity."},
    {"title": "Paris", "text": "Paris is the capital city of France."},
    {"title": "Photosynthesis", "text": "Photosynthesis converts light energy into chemical energy in plants."},
]


def test_retrieve_returns_default_top_3():
    index = PassageIndex.from_passages(PASSAGES)

    results = retrieve(index, "Tell me about science")

    assert len(results) == 3


def test_retrieve_respects_explicit_k():
    index = PassageIndex.from_passages(PASSAGES)

    results = retrieve(index, "Who was Einstein?", k=1)

    assert results == ["Albert Einstein was a theoretical physicist known for relativity."]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/rag && uv run pytest tests/test_retrieve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sense_rag.retrieve'`

- [ ] **Step 3: Write the implementation**

`services/rag/src/sense_rag/retrieve.py`:
```python
"""Public retrieval entry point used by the RAG baseline experiment script."""

from sense_rag.index import PassageIndex


def retrieve(index: PassageIndex, query: str, k: int = 3) -> list[str]:
    return index.search(query, k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/rag && uv run pytest tests/test_retrieve.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add services/rag/src/sense_rag/retrieve.py services/rag/tests/test_retrieve.py
git commit -m "Add retrieve() entry point for RAG baseline"
```

---

### Task 5: `configs/rag.yaml`

**Files:**
- Create: `configs/rag.yaml`

**Interfaces:**
- Produces: a config dict the experiment script (Task 6) loads via `load_yaml_config("rag.yaml")`, with keys `model` (string key into `configs/model.yaml`'s `models` map), `top_k` (int), `corpus_path` (string, repo-root-relative), `eval_split` (string, attribute name on `SplitIndices`).

- [ ] **Step 1: Create the config file**

`configs/rag.yaml`:
```yaml
# RAG comparison baseline (CLAUDE.md's Phase-0 design decisions). Accuracy-only,
# no latency instrumentation, no gate interaction — see docs/superpowers/specs/
# 2026-07-27-rag-baseline-design.md.
model: cpu_test        # key into configs/model.yaml, same model as RQ1-RQ3
top_k: 3
corpus_path: data/rag_corpus/passages.json
eval_split: test
```

- [ ] **Step 2: Commit**

```bash
git add configs/rag.yaml
git commit -m "Add configs/rag.yaml for the RAG baseline experiment"
```

---

### Task 6: RAG baseline experiment script

**Files:**
- Create: `experiments/rag_baseline_truthful_qa.py`
- Create: `experiments/tests/test_rag_baseline_truthful_qa.py` (create `experiments/tests/__init__.py` if no test dir exists yet — check first)

**Interfaces:**
- Consumes:
  - `sense_rag.index.PassageIndex.from_file(path) -> PassageIndex`, `.search`/`retrieve(index, query, k) -> list[str]` (Tasks 3-4)
  - `experiments._common.load_yaml_config(name) -> dict`, `load_model(model_cfg) -> (model, tokenizer)` (existing)
  - `sense_data.splits.load_splits(path) -> SplitIndices` (existing, `.test` attribute)
  - `sense_data.truthful_qa.load_truthful_qa() -> list[TruthfulQAExample]` (existing)
  - `sense_eval.factuality.lexical_containment_verdict(generated_text, best_answer, correct_answers, incorrect_answers) -> FactualityVerdict` (existing, confirmed signature)
- Produces: `build_prompt(passages: list[str], question: str) -> str`, `run_experiment(config: dict) -> dict` (importable for integration testing), a `results/rag_baseline_truthful_qa_{model}.json` file when run as `__main__`.

First, check whether `experiments/` already has a `tests/` directory and how RQ3 test file(s) are structured, to match conventions exactly.

- [ ] **Step 1: Inspect existing experiment test conventions**

Run: `find experiments -iname "test_*" -o -iname "conftest.py" | grep -v __pycache__`

Read whatever test file(s) that finds (likely testing `rq3_accuracy_latency_truthful_qa.py`) before writing Task 6's test, to reuse the same server-fixture-free, generation-based test pattern (RAG has no server to boot, unlike RQ3 — simpler).

- [ ] **Step 2: Write the failing tests**

`experiments/tests/test_rag_baseline_truthful_qa.py`:
```python
import json

import pytest

from rag_baseline_truthful_qa import build_prompt, run_experiment


def test_build_prompt_formats_context_and_question():
    prompt = build_prompt(["Paris is the capital of France.", "France is in Europe."], "What is the capital of France?")

    assert "Paris is the capital of France." in prompt
    assert "France is in Europe." in prompt
    assert "What is the capital of France?" in prompt
    assert prompt.index("Paris is the capital of France.") < prompt.index("What is the capital of France?")


@pytest.mark.asyncio
async def test_run_experiment_end_to_end_on_fixture_corpus(tmp_path, monkeypatch):
    fixture_passages = [
        {"title": "Fixture", "text": "This is a fixture passage about a fixture topic."},
    ]
    corpus_path = tmp_path / "passages.json"
    corpus_path.write_text(json.dumps(fixture_passages))

    config = {
        "models": {"cpu_test": {"hf_repo": "sshleifer/tiny-gpt2", "revision": "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"}},
        "gate": {"decoding": {"do_sample": False, "max_new_tokens": 5}},
        "rag": {"model": "cpu_test", "top_k": 1, "corpus_path": str(corpus_path), "eval_split": "test", "n_eval_examples": 2},
    }

    result = await run_experiment(config)

    assert result["n_eval_examples"] == 2
    assert 0.0 <= result["factuality_accuracy_proxy"] <= 1.0
    assert len(result["per_example"]) == 2
    for record in result["per_example"]:
        assert "retrieved_titles" in record
        assert "factuality" in record
```

Note: `run_experiment` takes an `n_eval_examples` override in `config["rag"]` for CPU-tractable testing, following the same subsampling pattern `rq3.yaml`/RQ3's harness already uses (`n_requested`).

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd experiments && uv run pytest tests/test_rag_baseline_truthful_qa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_baseline_truthful_qa'`

- [ ] **Step 4: Write the implementation**

`experiments/rag_baseline_truthful_qa.py`:
```python
"""RAG comparison baseline: retrieve-then-generate accuracy on the TruthfulQA test
split, using FAISS + all-MiniLM-L6-v2 over a committed, pre-built Wikipedia passage
corpus (services/rag/scripts/build_corpus.py). Accuracy-only — no latency
instrumentation, no gate interaction (CLAUDE.md: "a comparison baseline, not the
contribution — capped effort on purpose").

Retrieval and generation never touch the network at run time: the corpus is
pre-built and committed, and the embedding model is loaded from the local HF
cache after its first download. Same decoding config and model (tiny-gpt2) as
RQ1-RQ3, so this baseline's number sits in the same honest "pipeline-mechanics
validation, not a scientific finding" category until Llama-3/Mistral GPU runs.
"""

import json

from _common import RESULTS_DIR, REPO_ROOT, load_model, load_yaml_config
from sense_data.splits import load_splits
from sense_data.truthful_qa import load_truthful_qa
from sense_eval.factuality import lexical_containment_verdict
from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve

SPLIT_PATH = REPO_ROOT / "data" / "splits" / "truthful_qa.json"


def load_config() -> dict:
    return {
        "models": load_yaml_config("model.yaml")["models"],
        "gate": load_yaml_config("gate.yaml"),
        "rag": load_yaml_config("rag.yaml"),
    }


def build_prompt(passages: list[str], question: str) -> str:
    context_block = "\n".join(passages)
    return f"Context: {context_block}\n\nQuestion: {question}"


async def run_experiment(config: dict) -> dict:
    model_name = config["rag"]["model"]
    model_cfg = config["models"][model_name]
    decoding_cfg = config["gate"]["decoding"]
    top_k = config["rag"]["top_k"]

    model, tokenizer = load_model(model_cfg)
    index = PassageIndex.from_file(REPO_ROOT / config["rag"]["corpus_path"])

    examples = load_truthful_qa()
    splits = load_splits(SPLIT_PATH)
    all_eval_indices = getattr(splits, config["rag"]["eval_split"])

    n_requested = config["rag"].get("n_eval_examples", len(all_eval_indices))
    eval_indices = all_eval_indices[:n_requested]
    if n_requested < len(all_eval_indices):
        print(
            f"subsampling {config['rag']['eval_split']} split: using {len(eval_indices)} of "
            f"{len(all_eval_indices)} available examples for CPU tractability"
        )

    per_example = []
    for index_i in eval_indices:
        example = examples[index_i]
        passages = retrieve(index, example.question, k=top_k)
        prompt = build_prompt(passages, example.question)

        inputs = tokenizer(prompt, return_tensors="pt")
        output_ids = model.generate(
            **inputs,
            max_new_tokens=decoding_cfg["max_new_tokens"],
            do_sample=decoding_cfg["do_sample"],
        )
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        verdict = lexical_containment_verdict(
            generated_text, example.best_answer, example.correct_answers, example.incorrect_answers
        )

        per_example.append(
            {
                "index": index_i,
                "retrieved_titles": [p[:60] for p in passages],
                "generated_text": generated_text,
                "factuality": verdict.label,
            }
        )

    n = len(per_example)
    correct = [r for r in per_example if r["factuality"] == "correct"]

    return {
        "research_question": "RAG baseline",
        "dataset": "truthful_qa",
        "model_name": model_name,
        "hf_repo": model_cfg["hf_repo"],
        "revision": model_cfg["revision"],
        "eval_split": config["rag"]["eval_split"],
        "n_eval_examples": n,
        "n_available_in_split": len(all_eval_indices),
        "top_k": top_k,
        "decoding": decoding_cfg,
        "factuality_accuracy_proxy": len(correct) / n,
        "factuality_metric": "lexical_containment (placeholder, see eval/README.md)",
        "per_example": per_example,
    }


async def main() -> dict:
    config = load_config()
    result = await run_experiment(config)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"rag_baseline_truthful_qa_{config['rag']['model']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    summary = {k: v for k, v in result.items() if k != "per_example"}
    print(json.dumps(summary, indent=2))
    return result


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

Note: `retrieved_titles` in the results is actually storing truncated passage text, not titles (`PassageIndex.search`/`retrieve` return text, not titles) — fix the field name to `retrieved_passages_preview` for honesty, and adjust the test assertion key to match:

Edit the `per_example.append` block's key from `"retrieved_titles"` to `"retrieved_passages_preview"`, and edit the test's assertion `"retrieved_titles" in record` to `"retrieved_passages_preview" in record`.

Add `sense-rag` as an editable dependency of `experiments/pyproject.toml` (same pattern as the existing `sense-symbolic`/`sense-eval` entries) — inspect `experiments/pyproject.toml` first to match its exact `tool.uv.sources` style before editing.

- [ ] **Step 5: Wire up `experiments/pyproject.toml` dependency**

Run: `cat experiments/pyproject.toml` to see the existing `sense-symbolic`/`sense-eval` dependency + `[tool.uv.sources]` entries, then add matching entries for `sense-rag` pointing at `../services/rag`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd experiments && uv sync && uv run pytest tests/test_rag_baseline_truthful_qa.py -v`
Expected: 2 passed
(This test still needs the real tiny-gpt2 model, downloaded/cached like every other experiment test in this repo — consistent with RQ3's integration test pattern.)

- [ ] **Step 7: Commit**

```bash
git add experiments/rag_baseline_truthful_qa.py experiments/tests/test_rag_baseline_truthful_qa.py \
    experiments/pyproject.toml experiments/uv.lock
git commit -m "Add RAG baseline experiment script"
```

---

### Task 7: Build the real corpus and run the full baseline

**Files:**
- Create: `data/rag_corpus/passages.json` (generated artifact, committed)
- Create: `results/rag_baseline_truthful_qa_cpu_test.json` (generated artifact, committed — matches existing convention of committing RQ result JSONs)

**Interfaces:**
- Consumes: Task 2's `build_corpus.py`, Task 6's `rag_baseline_truthful_qa.py`.
- Produces: the actual data artifacts other documentation (Task 8) references.

- [ ] **Step 1: Run the corpus build script**

Run: `cd services/rag && uv run python scripts/build_corpus.py`
Expected: prints `wrote N passages (of 327 questions) to .../data/rag_corpus/passages.json`. Some skipped-question lines on stderr are expected and fine (per the spec's error-handling section).

- [ ] **Step 2: Run the full baseline experiment (test split, all resolved examples)**

Set `configs/rag.yaml` to not subsample (no `n_eval_examples` key means the full split runs — confirm `run_experiment`'s `config["rag"].get("n_eval_examples", ...)` default handles this).

Run: `cd experiments && uv run python rag_baseline_truthful_qa.py`
Expected: completes, writes `results/rag_baseline_truthful_qa_cpu_test.json`, prints a summary with `factuality_accuracy_proxy`.

- [ ] **Step 3: Commit the generated artifacts**

```bash
git add data/rag_corpus/passages.json results/rag_baseline_truthful_qa_cpu_test.json
git commit -m "Run RAG baseline on TruthfulQA test split, commit corpus and results"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md` (root)
- Modify: `CLAUDE.md` (Current status section)
- Modify: `experiments/README.md`

**Interfaces:**
- No code interfaces — this task updates prose to reflect Tasks 1-7's artifacts.

- [ ] **Step 1: Update `experiments/README.md`**

Read the existing file first, then add a section describing `rag_baseline_truthful_qa.py` in the same style as its existing `rq3` section: what it does, what config it reads, what result file it produces, and the explicit caveat that `factuality_accuracy_proxy` uses the lexical-containment placeholder metric.

- [ ] **Step 2: Update root `README.md`'s status section**

Read the existing status section, then add one sentence noting the RAG baseline now exists and ran end-to-end (with its own honest tiny-gpt2 caveat, consistent with the RQ1-RQ3 sentences already there).

- [ ] **Step 3: Update `CLAUDE.md`'s "Current status" section**

Add the RAG baseline to the list of what's built, matching the existing bullet style. Remove "RAG baseline" from the "Not yet built" list at the end of that section (it currently reads "Not yet built: RAG baseline, Llama-3/Mistral runs on GPU...").

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md experiments/README.md
git commit -m "Document the RAG baseline in project READMEs and CLAUDE.md"
```

---

## Self-Review Notes

- **Spec coverage:** every spec section has a task — package scaffold (Task 1), corpus build (Task 2), index/retriever (Tasks 3-4), config (Task 5), experiment script (Task 6), the actual run + committed artifacts (Task 7), docs (Task 8). Testing and error-handling sections are folded into Tasks 2/3/4/6's test steps and Task 2's skip-and-log behavior.
- **Type consistency:** `PassageIndex.from_passages`/`from_file`/`.search` (Task 3) are the exact names `retrieve()` (Task 4) and the experiment script (Task 6) call. `build_corpus`/`fetch_passage` (Task 2) signatures match their test file. `lexical_containment_verdict`'s signature was confirmed against the real `eval/src/sense_eval/factuality.py` source, not assumed.
- **Fixed during self-review:** Task 6's result field was originally named `retrieved_titles` while actually holding truncated passage text (since `retrieve()` returns text, not titles) — renamed to `retrieved_passages_preview` in both the implementation and its test, inline, before finalizing.
