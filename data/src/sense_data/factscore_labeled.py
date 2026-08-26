"""Loader for the FActScore paper's own human-annotated labeled set (Min et al.,
2023, EMNLP) — 183 Wikipedia entities x 3 real commercial-LLM generations
(InstructGPT, ChatGPT, PerplexityAI), each biography broken into atomic facts
and hand-labeled Supported/Not-Supported/Irrelevant against Wikipedia by human
annotators.

This is disjoint from `sense_data.factscore` (the 500-entity `dskar/FActScore`
prompt set this project's RQ/RAG/symbolic-verification harnesses use, and its
committed split file `data/splits/factscore.json`) — zero entity overlap
confirmed at fixture-generation time. Its only use in this project is
calibrating `configs/nli_judge.yaml`'s FActScore-shaped thresholds
(`experiments/calibrate_factscore_thresholds.py`): it supplies real human
factuality labels on real (if not this project's own five-checkpoint) model
generations, the "reused human annotations for different models' outputs"
path named as the calibration route in
`docs/superpowers/specs/2026-08-16-nli-judge-design.md` and `eval/README.md`.

Provenance (see `data/fixtures/factscore_labeled/PROVENANCE.md`): fetched
2026-08-24 from the paper's own Google Drive release
(https://github.com/shmsw25/FActScore, MIT-licensed), file `data.zip`,
sha256 `404482c9a7d588d581a2f7a78d609fe6bfec4ddc0f729c8560a61b1743dbffab`.
Not available as a pinned HF Hub dataset (unlike this project's other
loaders), so the small (~3.3MB) label files are committed directly into the
repo instead — the same reproducibility principle CLAUDE.md applies to split
index lists ("committed to the repo, not a runtime fetch"), extended here
because no HF revision SHA exists to pin against.

Reference text: this fixture set also carries a `wikipedia_reference_text.json`
mapping topic -> current Wikipedia plain-text extract, fetched 2026-08-24 via
the public MediaWiki API (see `data/scripts/fetch_factscore_labeled_reference_text.py`).
Honest limitation: this is a 2026 snapshot, not the 2023-vintage Wikipedia
text the original human annotators actually verified against — an entity's
article can drift between those dates (more so for entities in the news
after 2023). Stated wherever this fixture's calibration result is cited, same
as the short-answer calibration's own honest-limitation note.
"""

import json
from dataclasses import dataclass
from pathlib import Path

MODELS = ("InstructGPT", "ChatGPT", "PerplexityAI")
FACT_LABELS = frozenset({"S", "NS", "IR"})


@dataclass(frozen=True)
class LabeledAtomicFact:
    text: str
    label: str  # "S" (supported) | "NS" (not supported) | "IR" (irrelevant)


@dataclass(frozen=True)
class FActScoreLabeledExample:
    model: str
    topic: str
    generated_text: str
    human_atomic_facts: tuple[LabeledAtomicFact, ...]
    reference_text: str | None  # None if the Wikipedia fetch failed for this topic


def _load_reference_text(fixtures_dir: Path) -> dict[str, str]:
    raw = json.loads((fixtures_dir / "wikipedia_reference_text.json").read_text())
    return {topic: entry["extract"] for topic, entry in raw.items()}


def load_factscore_labeled(fixtures_dir: Path) -> list[FActScoreLabeledExample]:
    """Loads all 3 models' labeled examples (183 topics each) from the committed
    fixture files at `fixtures_dir` (see `data/fixtures/factscore_labeled/`).

    Each source line nests per-sentence `annotations`, each carrying its own
    `human-atomic-facts` list — this flattens every annotation's atomic facts
    into one `human_atomic_facts` tuple per example, matching the whole-
    generation granularity `factscore_style_verdict` scores against.

    Two null cases in the raw data, both handled by producing an empty
    `human_atomic_facts` tuple rather than raising: a row's `annotations` is
    null when the model's `output` was empty (it declined/failed to
    generate — 44 of 549 rows); a sentence `annotation`'s `human-atomic-facts`
    is null when annotators marked that sentence `is-relevant: false` (an
    irrelevant sentence gets no atomic-fact breakdown at all).
    """
    reference_text = _load_reference_text(fixtures_dir)
    examples = []
    for model in MODELS:
        path = fixtures_dir / f"{model}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            facts = tuple(
                LabeledAtomicFact(text=fact["text"], label=fact["label"])
                for annotation in (row["annotations"] or [])
                for fact in (annotation.get("human-atomic-facts") or [])
            )
            examples.append(
                FActScoreLabeledExample(
                    model=model,
                    topic=row["topic"],
                    generated_text=row["output"],
                    human_atomic_facts=facts,
                    reference_text=reference_text.get(row["topic"]),
                )
            )
    return examples
