"""
Computes RAGAS generation metrics (Faithfulness, Answer Relevancy) on a
stratified subsample of the golden dataset, using Groq as the judge LLM.

Usage:
    python -m eval.compute_generation_metrics
"""
import json
import random
from pathlib import Path
from collections import defaultdict
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
import math

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

RESULTS_PATH = Path(__file__).resolve().parent / "pipeline_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "generation_metrics.json"

SAMPLE_SIZE = 20
RANDOM_SEED = 42


def stratified_sample(results: list, sample_size: int) -> list:
    """Sample proportionally across categories so small categories aren't left out."""
    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    random.seed(RANDOM_SEED)
    total = len(results)
    sample = []

    for cat, items in by_category.items():
        # proportional share of this category, at least 1 if the category has any items
        share = max(1, round(len(items) / total * sample_size))
        random.shuffle(items)
        sample.extend(items[:share])

    random.shuffle(sample)
    return sample[:sample_size]


def build_ragas_dataset(sample: list) -> Dataset:
    return Dataset.from_dict({
        "question": [r["question"] for r in sample],
        "answer": [r["generated_answer"] for r in sample],
        "contexts": [r["retrieved_texts"] for r in sample],
    })


def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    sample = stratified_sample(results, SAMPLE_SIZE)

    print(f"Sampled {len(sample)} questions out of {len(results)} for generation metrics.")
    cat_counts = defaultdict(int)
    for r in sample:
        cat_counts[r["category"]] += 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    judge_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    # Reuse the same local embedding model from Stage 1 (no OpenAI key needed)
    local_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )

    dataset = build_ragas_dataset(sample)

    print("\nRunning RAGAS evaluation (this calls the LLM multiple times per question)...")
    run_config = RunConfig(max_workers=1, timeout=120)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=local_embeddings,
        run_config=run_config,
    )

    df = result.to_pandas()

    output_rows = []
    for i in range(len(sample)):
        faith = df["faithfulness"].iloc[i] if "faithfulness" in df.columns and i < len(df) else None
        rel = df["answer_relevancy"].iloc[i] if "answer_relevancy" in df.columns and i < len(df) else None
        output_rows.append({
            "id": sample[i]["id"],
            "category": sample[i]["category"],
            "question": sample[i]["question"],
            "faithfulness": float(faith) if faith is not None and faith == faith else None,
            "answer_relevancy": float(rel) if rel is not None and rel == rel else None,
        })

    valid_faith = [r["faithfulness"] for r in output_rows if r["faithfulness"] is not None]
    valid_rel = [r["answer_relevancy"] for r in output_rows if r["answer_relevancy"] is not None]

    summary = {
        "sample_size": len(sample),
        "avg_faithfulness": sum(valid_faith) / len(valid_faith) if valid_faith else None,
        "avg_answer_relevancy": sum(valid_rel) / len(valid_rel) if valid_rel else None,
    }

    print("\n" + "=" * 60)
    print("GENERATION METRICS (subsample)")
    print("=" * 60)
    print(f"  Sample size:        {summary['sample_size']}")
    print(f"  Avg Faithfulness:   {summary['avg_faithfulness']:.3f}")
    print(f"  Avg Answer Relevancy: {summary['avg_answer_relevancy']:.3f}")

    output = {"summary": summary, "per_question": output_rows}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()