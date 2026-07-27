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
                "retrieved_passages_preview": [p[:60] for p in passages],
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
