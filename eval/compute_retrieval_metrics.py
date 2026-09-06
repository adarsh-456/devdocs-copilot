"""
Computes retrieval-quality metrics (Hit Rate@k, MRR, Precision@k) by
comparing retrieved sources against the golden dataset's expected sources.

No LLM calls needed — purely string matching against pipeline_results.json.

Usage:
    python -m eval.compute_retrieval_metrics
"""
import json
import re
from pathlib import Path
from collections import defaultdict

RESULTS_PATH = Path(__file__).resolve().parent / "pipeline_results.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "retrieval_metrics.json"


def source_matches_expected(retrieved_source: str, expected: str) -> bool:
    """Check if one retrieved source string matches one expected reference."""
    if expected.startswith("issue_"):
        issue_num = expected.replace("issue_", "")
        return f"#{issue_num} " in retrieved_source or retrieved_source.endswith(f"#{issue_num}")
    else:
        return expected in retrieved_source


def evaluate_question(result: dict) -> dict:
    expected = result.get("expected_source_contains", [])
    retrieved = result.get("retrieved_sources", [])

    if not expected:
        # negative/out-of-scope questions: nothing to score for retrieval
        return {"scored": False}

    match_ranks = []  # 1-indexed rank of each retrieved chunk that matches ANY expected source
    for rank, src in enumerate(retrieved, 1):
        if any(source_matches_expected(src, exp) for exp in expected):
            match_ranks.append(rank)

    hit = len(match_ranks) > 0
    mrr = (1.0 / match_ranks[0]) if match_ranks else 0.0
    precision = len(match_ranks) / len(retrieved) if retrieved else 0.0

    return {
        "scored": True,
        "hit": hit,
        "mrr": mrr,
        "precision": precision,
        "num_matches": len(match_ranks),
        "num_retrieved": len(retrieved),
    }


def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    per_question = []
    by_category = defaultdict(list)

    for r in results:
        score = evaluate_question(r)
        score["id"] = r["id"]
        score["category"] = r["category"]
        per_question.append(score)
        if score["scored"]:
            by_category[r["category"]].append(score)

    scored_all = [s for s in per_question if s["scored"]]

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

    overall = summarize(scored_all)

    print("=" * 60)
    print("RETRIEVAL METRICS — OVERALL")
    print("=" * 60)
    print(f"  Scored questions: {overall['n']} (out of {len(results)} total; "
          f"{len(results) - overall['n']} are negative/out-of-scope, not scored)")
    print(f"  Hit Rate@5:  {overall['hit_rate']:.3f}")
    print(f"  MRR:         {overall['mrr']:.3f}")
    print(f"  Precision@5: {overall['precision']:.3f}")

    print("\n" + "=" * 60)
    print("BY CATEGORY")
    print("=" * 60)
    category_summaries = {}
    for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
        s = summarize(items)
        category_summaries[cat] = s
        print(f"  {cat:20s} n={s['n']:3d}  hit_rate={s['hit_rate']:.3f}  mrr={s['mrr']:.3f}  precision={s['precision']:.3f}")

    output = {
        "overall": overall,
        "by_category": category_summaries,
        "per_question": per_question,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()