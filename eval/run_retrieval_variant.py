"""
Runs retrieval-only evaluation (no LLM calls) for a given configuration,
computes metrics, and optionally logs to MLflow. Reusable across variants.

Usage:
    python -m eval.run_retrieval_variant --top_k 8 --run_name "top_k_8"
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()
import mlflow
from retrieval.search import search
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank

DATASET_PATH = Path(__file__).resolve().parent / "golden_questions.json"


def load_dataset():
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return raw["records"] if "records" in raw else raw


def source_matches_expected(retrieved_source: str, expected: str) -> bool:
    if expected.startswith("issue_"):
        issue_num = expected.replace("issue_", "")
        return f"#{issue_num} " in retrieved_source or retrieved_source.endswith(f"#{issue_num}")
    return expected in retrieved_source


def evaluate_question(question: str, expected: list, top_k: int, collection_name: str, use_hybrid: bool = False, use_reranker: bool = False) -> dict:
    if not expected:
        return {"scored": False}
    
    if use_reranker:
        candidates = search(question, top_k=15, collection_name=collection_name)
        chunks = rerank(question, candidates, top_k=top_k)
    elif use_hybrid:
        chunks = hybrid_search(question, top_k=top_k, collection_name=collection_name)
    else:
        chunks = search(question, top_k=top_k, collection_name=collection_name)
    retrieved = [c["source"] for c in chunks]

    match_ranks = []
    for rank, src in enumerate(retrieved, 1):
        if any(source_matches_expected(src, exp) for exp in expected):
            match_ranks.append(rank)

    hit = len(match_ranks) > 0
    mrr = (1.0 / match_ranks[0]) if match_ranks else 0.0
    precision = len(match_ranks) / len(retrieved) if retrieved else 0.0

    return {"scored": True, "hit": hit, "mrr": mrr, "precision": precision}


def main(top_k: int, run_name: str, collection_name: str, extra_params: dict = None, use_hybrid: bool = False, use_reranker: bool = False):
    records = load_dataset()
    scored = []
    by_category = defaultdict(list)

    for i, r in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {r['id']}", end="\r")
        result = evaluate_question(r["question"], r.get("expected_source_contains", []), top_k, collection_name, use_hybrid, use_reranker)
        if result["scored"]:
            scored.append(result)
            by_category[r["category"]].append(result)

    def summarize(items):
        n = len(items)
        if n == 0:
            return {"n": 0, "hit_rate": None, "mrr": None, "precision": None}
        return {
            "n": n,
            "hit_rate": sum(1 for i in items if i["hit"]) / n,
            "mrr": sum(i["mrr"] for i in items) / n,
            "precision": sum(i["precision"] for i in items) / n,
        }

    overall = summarize(scored)

    print(f"\n\n{'='*60}")
    print(f"VARIANT: {run_name} (top_k={top_k})")
    print(f"{'='*60}")
    print(f"  Hit Rate: {overall['hit_rate']:.3f}")
    print(f"  MRR:      {overall['mrr']:.3f}")
    print(f"  Precision: {overall['precision']:.3f}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("devdocs-copilot-rag-eval")

    with mlflow.start_run(run_name=run_name):
        params = {
            "retrieval_top_k": top_k,
            "collection": collection_name,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "hybrid_search": use_hybrid,
            "reranker": use_reranker,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        mlflow.log_metrics({
            "retrieval_hit_rate": overall["hit_rate"],
            "retrieval_mrr": overall["mrr"],
            "retrieval_precision": overall["precision"],
        })
        print(f"\nLogged to MLflow as run: {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="variant")
    parser.add_argument("--collection", type=str, default="devdocs_copilot")
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    args = parser.parse_args()

    extra = {}
    if args.chunk_size:
        extra["chunk_size_words"] = args.chunk_size
    if args.overlap:
        extra["chunk_overlap_words"] = args.overlap

    main(args.top_k, args.run_name, args.collection, extra, args.hybrid, args.reranker)
