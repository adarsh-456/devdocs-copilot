"""
Logs the Stage 1 baseline pipeline's evaluation results to MLflow —
one run, combining retrieval metrics (full dataset) and generation
metrics (sampled subset) with the config that produced them.

Usage:
    python -m eval.log_baseline_to_mlflow
"""
import json
import mlflow
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

RETRIEVAL_METRICS_PATH = EVAL_DIR / "retrieval_metrics.json"
GENERATION_METRICS_PATH = EVAL_DIR / "generation_metrics.json"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden_questions.json"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("devdocs-copilot-rag-eval")


def main():
    retrieval = json.loads(RETRIEVAL_METRICS_PATH.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION_METRICS_PATH.read_text(encoding="utf-8"))

    with mlflow.start_run(run_name="baseline_stage1_pipeline"):
        # --- Config that produced these results ---
        mlflow.log_params({
            "chunk_size_words": 220,
            "chunk_overlap_words": 40,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "vector_db": "chromadb (cosine similarity)",
            "retrieval_top_k": 5,
            "llm_model": "openai/gpt-oss-20b",
            "llm_temperature": 0.1,
            "llm_max_tokens": 500,
            "llm_reasoning_effort": "low",
            "hybrid_search": False,
            "reranker": False,
            "golden_dataset_size": 114,
        })

        # --- Retrieval metrics (full 114 questions) ---
        overall = retrieval["overall"]
        mlflow.log_metrics({
            "retrieval_hit_rate_at_5": overall["hit_rate"],
            "retrieval_mrr": overall["mrr"],
            "retrieval_precision_at_5": overall["precision"],
        })

        # Per-category retrieval metrics too, for later comparison
        for cat, stats in retrieval["by_category"].items():
            if stats["hit_rate"] is not None:
                mlflow.log_metric(f"hit_rate_{cat}", stats["hit_rate"])
                mlflow.log_metric(f"mrr_{cat}", stats["mrr"])

        # --- Generation metrics (sampled subset, with honest sample sizes logged) ---
        gen_summary = generation["summary"]
        mlflow.log_metrics({
            "generation_faithfulness": gen_summary["avg_faithfulness"],
            "generation_faithfulness_sample_size": 8,  # documented limitation, see eval/NOTES.md
            "generation_answer_relevancy": gen_summary["avg_answer_relevancy"],
            "generation_answer_relevancy_sample_size": gen_summary["sample_size"],
        })

        # --- Attach the actual result files as artifacts ---
        mlflow.log_artifact(str(RETRIEVAL_METRICS_PATH))
        mlflow.log_artifact(str(GENERATION_METRICS_PATH))
        mlflow.log_artifact(str(GOLDEN_DATASET_PATH))
        mlflow.log_artifact(str(EVAL_DIR / "NOTES.md"))

        run = mlflow.active_run()
        print(f"Logged baseline run: {run.info.run_id}")
        print(f"\nSummary:")
        print(f"  Hit Rate@5:      {overall['hit_rate']:.3f}")
        print(f"  MRR:             {overall['mrr']:.3f}")
        print(f"  Precision@5:     {overall['precision']:.3f}")
        print(f"  Faithfulness:    {gen_summary['avg_faithfulness']:.3f} (n=8, see NOTES.md)")
        print(f"  Answer Relevancy: {gen_summary['avg_answer_relevancy']:.3f} (n=20)")


if __name__ == "__main__":
    main()