"""RAG comparison baseline: retrieve-then-generate accuracy on the HaluEval test
split, using FAISS + all-MiniLM-L6-v2 over a committed, pre-built Wikipedia passage
corpus (services/rag/scripts/build_corpus.py). Accuracy-only — no latency
instrumentation, no gate interaction (CLAUDE.md: "a comparison baseline, not the
contribution — capped effort on purpose").

Re-pointed from TruthfulQA to HaluEval (2026-08-13 scope reconciliation) —
TruthfulQA stays in the repo for the excluded-benchmark discussion (CLAUDE.md),
but is no longer what this baseline is scored against. HaluEval gives each
example a `right_answer` and a model-authored `hallucinated_answer`; the
lexical-containment verdict below treats those the same way the TruthfulQA
version treated best/correct vs. incorrect answers.

Retrieval and generation never touch the network at run time: the corpus is
pre-built and committed, and the embedding model is loaded from the local HF
cache after its first download. Same decoding config and model (tiny-gpt2) as
RQ1-RQ3, so this baseline's number sits in the same honest "pipeline-mechanics
validation, not a scientific finding" category until Llama-3/Mistral GPU runs.
Factuality is scored by the NLI judge (`sense_eval.nli_judge`), not the
retired lexical-containment placeholder — see `eval/README.md`.

Reports task_accuracy/hallucination_rate/abstention_rate together (never a
subset — write_results enforces this); abstention_rate is always 0 here, since
this baseline has no routing/merge-back policy to abstain with at all.
"""

from _common import REPO_ROOT, load_model, load_model_registry, load_yaml_config, write_results
from _common import load_halueval_examples_and_splits as load_examples_and_splits
from sense_eval.factuality import FactualityVerdict, summarize_factuality
from sense_eval.nli_judge import NLI_METRIC_LABEL_TEMPLATE, load_nli_model, nli_verdict_short_answer
from sense_rag.index import PassageIndex
from sense_rag.retrieve import retrieve


def load_config() -> dict:
    return {
        "models": load_model_registry(),
        "gate": load_yaml_config("gate.yaml"),
        "rag": load_yaml_config("rag.yaml"),
        "nli_judge": load_yaml_config("nli_judge.yaml"),
    }


def build_prompt(passages: list[str], question: str) -> str:
    context_block = "\n".join(passages)
    return f"Context: {context_block}\n\nQuestion: {question}"


def build_prompt_within_budget(
    passages: list[str], question: str, tokenizer, max_tokens: int
) -> tuple[str, int, bool]:
    """Build a prompt by adding passages in rank order (most relevant first),
    stopping once the next passage would exceed the token budget.

    This keeps the question intact and preserves the highest-ranked passages
    that fit, rather than blindly truncating a fixed-order block from an
    arbitrary point (which would discard the most relevant passage first when
    it's truncated from the left).

    Returns (prompt, n_passages_used, any_passage_dropped).
    """
    used: list[str] = []
    for passage in passages:
        candidate = build_prompt(used + [passage], question)
        n_tokens = len(tokenizer(candidate, truncation=False)["input_ids"])
        if n_tokens > max_tokens and used:
            # Adding this passage would exceed budget and we already have at
            # least one passage kept; stop here.
            return build_prompt(used, question), len(used), True
        if n_tokens > max_tokens and not used:
            # Even the first passage alone doesn't fit; keep it anyway and
            # rely on the tokenizer's truncation as a final safety net.
            return candidate, 1, True
        used.append(passage)

    return build_prompt(used, question), len(used), False


async def run_experiment(config: dict) -> dict:
    model_name = config["rag"]["model"]
    model_cfg = config["models"][model_name]
    decoding_cfg = config["gate"]["decoding"]
    top_k = config["rag"]["top_k"]

    model, tokenizer = load_model(model_cfg)
    nli_model, nli_tokenizer = load_nli_model(config["nli_judge"]["hf_repo"], config["nli_judge"]["revision"])
    index = PassageIndex.from_file(REPO_ROOT / config["rag"]["corpus_path"])

    # tokenizer.model_max_length can report a placeholder (e.g. 1e30) for models
    # without an explicit config value; fall back to the model's real position
    # embedding limit in that case. Llama/Mistral expose max_position_embeddings;
    # GPT-2 exposes n_positions — never hard-code a model family (CLAUDE.md).
    model_max_length = tokenizer.model_max_length
    if model_max_length is None or model_max_length > 1_000_000:
        model_max_length = getattr(model.config, "max_position_embeddings", None) or model.config.n_positions

    # Final safety net only: build_prompt_within_budget should make this rarely
    # if ever fire, but keep it in case even question+one-passage is too long.
    tokenizer.truncation_side = "left"

    examples, splits = load_examples_and_splits()
    all_eval_indices = getattr(splits, config["rag"]["eval_split"])

    n_requested = config["rag"].get("n_eval_examples", len(all_eval_indices))
    eval_indices = all_eval_indices[:n_requested]
    if n_requested < len(all_eval_indices):
        print(
            f"subsampling {config['rag']['eval_split']} split: using {len(eval_indices)} of "
            f"{len(all_eval_indices)} available examples for CPU tractability"
        )

    budget = model_max_length - decoding_cfg["max_new_tokens"]

    per_example = []
    for index_i in eval_indices:
        example = examples[index_i]
        passages = retrieve(index, example.question, k=top_k)
        prompt, n_passages_used, passage_dropped = build_prompt_within_budget(
            passages, example.question, tokenizer, budget
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=budget,
        )
        safety_net_truncated = inputs["input_ids"].shape[1] >= budget
        prompt_truncated = passage_dropped or safety_net_truncated

        output_ids = model.generate(
            **inputs,
            max_new_tokens=decoding_cfg["max_new_tokens"],
            do_sample=decoding_cfg["do_sample"],
        )
        generated_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

        verdict = nli_verdict_short_answer(
            nli_model,
            nli_tokenizer,
            generated_text,
            example.right_answer,
            example.hallucinated_answer,
            config["nli_judge"]["short_answer_entailment_threshold"],
        )

        per_example.append(
            {
                "index": index_i,
                "retrieved_passages_preview": [p[:60] for p in passages],
                "generated_text": generated_text,
                "factuality": verdict.label,
                "prompt_truncated": prompt_truncated,
                "n_passages_used": n_passages_used,
            }
        )

    n = len(per_example)
    # No routing/merge-back policy exists in this baseline at all — n_abstained is
    # always 0, but stated explicitly (not defaulted) so that stays a deliberate
    # fact about this harness rather than an assumption baked into the reporting
    # layer.
    n_abstained = 0
    verdicts = [FactualityVerdict(label=r["factuality"]) for r in per_example]
    factuality_report = summarize_factuality(verdicts, n_abstained=n_abstained)
    n_examples_with_truncated_prompt = sum(1 for r in per_example if r["prompt_truncated"])

    return {
        "research_question": "RAG baseline",
        "dataset": "halueval",
        "model_name": model_name,
        "hf_repo": model_cfg["hf_repo"],
        "revision": model_cfg["revision"],
        "eval_split": config["rag"]["eval_split"],
        "n_eval_examples": n,
        "n_available_in_split": len(all_eval_indices),
        "top_k": top_k,
        "decoding": decoding_cfg,
        "task_accuracy": factuality_report["task_accuracy"],
        "hallucination_rate": factuality_report["hallucination_rate"],
        "abstention_rate": factuality_report["abstention_rate"],
        "factuality_report": factuality_report,
        "factuality_metric": NLI_METRIC_LABEL_TEMPLATE.format(
            hf_repo=config["nli_judge"]["hf_repo"], revision=config["nli_judge"]["revision"]
        ),
        "n_examples_with_truncated_prompt": n_examples_with_truncated_prompt,
        "per_example": per_example,
    }


async def main() -> dict:
    config = load_config()
    result = await run_experiment(config)

    write_results(
        f"rag_baseline_halueval_{config['rag']['model']}.json",
        result,
        print_exclude_keys=frozenset({"per_example"}),
    )
    return result


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
