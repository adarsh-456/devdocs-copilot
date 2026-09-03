"""
Validates the golden question dataset against the actual files in data/docs/
and data/issues/, before running any evaluation metrics.

Checks:
1. Every expected_source_contains file stem actually exists in data/docs/
   (for issue references, checks data/issues/issue_<number>.json exists)
2. Flags questions with empty expected_source_contains that aren't
   negative/out-of-scope questions (likely a mistake)
3. Prints a per-category summary

Usage:
    python -m eval.validate_dataset
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = DATA_DIR / "docs"
ISSUES_DIR = DATA_DIR / "issues"
DATASET_PATH = Path(__file__).resolve().parent / "golden_questions.json"

NEGATIVE_LIKE_CATEGORIES = {"negative", "out_of_scope"}


def load_dataset():
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return raw["records"] if "records" in raw else raw


def get_existing_doc_stems():
    return {p.stem for p in DOCS_DIR.glob("*.md")}


def get_existing_issue_numbers():
    numbers = set()
    for p in ISSUES_DIR.glob("issue_*.json"):
        numbers.add(p.stem.replace("issue_", ""))
    return numbers


def validate():
    records = load_dataset()
    doc_stems = get_existing_doc_stems()
    issue_numbers = get_existing_issue_numbers()

    print(f"Found {len(doc_stems)} doc files and {len(issue_numbers)} issue files locally.\n")

    problems = []
    category_counts = {}

    for r in records:
        cat = r.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

        sources = r.get("expected_source_contains", [])

        if not sources:
            if cat not in NEGATIVE_LIKE_CATEGORIES:
                problems.append((r["id"], cat, "empty expected_source_contains but category isn't negative/out-of-scope"))
            continue

        for s in sources:
            # could be a doc stem or a raw issue number
            if s in doc_stems:
                continue
            if s.isdigit() and s in issue_numbers:
                continue
            problems.append((r["id"], cat, f"'{s}' not found in data/docs/ or data/issues/"))

    print("=" * 60)
    print("CATEGORY BREAKDOWN")
    print("=" * 60)
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        flag = "  <- low sample size" if count < 5 else ""
        print(f"  {cat:20s} {count:3d}{flag}")

    print("\n" + "=" * 60)
    print(f"VALIDATION ISSUES: {len(problems)}")
    print("=" * 60)
    for qid, cat, msg in problems:
        print(f"  [{qid}] ({cat}) {msg}")

    if not problems:
        print("  None — all references check out.")

    print(f"\nTotal questions: {len(records)}")
    return problems


if __name__ == "__main__":
    validate()