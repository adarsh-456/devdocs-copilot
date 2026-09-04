"""
Runs every question in the golden dataset through the actual RAG pipeline
(retrieval + generation) and saves raw results to JSON, incrementally.

Resumable: if this crashes or is interrupted, just re-run it — it will
skip questions that already have saved results and continue from there.

Usage:
    python -m eval.run_pipeline
"""
import json
import time
from pathlib import Path

from retrieval.search import search
from generation.generate import generate_answer

DATASET_PATH = Path(__file__).resolve().parent / "golden_questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "pipeline_results.json"

TOP_K = 5
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 30


def load_dataset():
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return raw["records"] if "records" in raw else raw


def load_existing_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return []


def save_results(results):
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def generate_with_retry(question: str, chunks: list, max_retries: int = MAX_RETRIES) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            return generate_answer(question, chunks)
        except Exception as e:
            error_msg = str(e).lower()
            is_rate_limit = "rate" in error_msg or "429" in error_msg or "quota" in error_msg
            if is_rate_limit and attempt < max_retries:
                print(f"  Rate limit hit (attempt {attempt}/{max_retries}). "
                      f"Waiting {RETRY_WAIT_SECONDS}s before retrying...")
                time.sleep(RETRY_WAIT_SECONDS)
            elif attempt < max_retries:
                print(f"  Error (attempt {attempt}/{max_retries}): {e}. Retrying...")
                time.sleep(5)
            else:
                return f"[ERROR after {max_retries} attempts: {e}]"


def run_all(top_k: int = TOP_K):
    records = load_dataset()
    results = load_existing_results()
    done_ids = {r["id"] for r in results}

    remaining = [r for r in records if r["id"] not in done_ids]
    print(f"Total questions: {len(records)} | Already done: {len(done_ids)} | Remaining: {len(remaining)}\n")

    for i, record in enumerate(remaining, 1):
        question = record["question"]
        print(f"[{i}/{len(remaining)}] {record['id']}: {question}")

        try:
            chunks = search(question, top_k=top_k)
            answer = generate_with_retry(question, chunks)

            results.append({
                "id": record["id"],
                "question": question,
                "category": record["category"],
                "expected_source_contains": record.get("expected_source_contains", []),
                "expected_answer_keypoints": record.get("expected_answer_keypoints", []),
                "retrieved_sources": [c["source"] for c in chunks],
                "retrieved_texts": [c["text"] for c in chunks],
                "retrieved_scores": [c["score"] for c in chunks],
                "generated_answer": answer,
            })

            # save after every question, so a crash doesn't lose progress
            save_results(results)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Progress saved — re-run this script to resume.")
            return
        except Exception as e:
            print(f"  Unexpected error on {record['id']}: {e}. Skipping for now.")

        time.sleep(0.5)  # be gentle on the free API rate limit

    print(f"\nDone. {len(results)}/{len(records)} results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_all()


# """
# Runs every question in the golden dataset through the actual RAG pipeline
# (retrieval + generation) and saves raw results to JSON, incrementally.

# Resumable: if this crashes, hits a rate limit, or is interrupted, just
# re-run it — it will skip questions that already have a SUCCESSFUL saved
# result and retry anything that previously errored out.

# Usage:
#     python -m eval.run_pipeline
# """
# import json
# import time
# from pathlib import Path

# from retrieval.search import search
# from generation.generate import generate_answer

# DATASET_PATH = Path(__file__).resolve().parent / "golden_questions.json"
# RESULTS_PATH = Path(__file__).resolve().parent / "pipeline_results.json"

# TOP_K = 5
# MAX_RETRIES = 3
# RETRY_WAIT_SECONDS = 30


# def load_dataset():
#     raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
#     return raw["records"] if "records" in raw else raw


# def load_existing_results():
#     if RESULTS_PATH.exists():
#         return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
#     return []


# def save_results(results):
#     RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


# def is_error_result(answer: str) -> bool:
#     return isinstance(answer, str) and answer.startswith("[ERROR")


# class DailyQuotaExceeded(Exception):
#     pass


# def generate_with_retry(question: str, chunks: list, max_retries: int = MAX_RETRIES) -> str:
#     for attempt in range(1, max_retries + 1):
#         try:
#             return generate_answer(question, chunks)
#         except Exception as e:
#             error_msg = str(e).lower()

#             # Daily quota (TPD = tokens per day) can't be fixed by a short retry
#             if "tokens per day" in error_msg or "tpd" in error_msg:
#                 raise DailyQuotaExceeded(str(e))

#             is_rate_limit = "rate" in error_msg or "429" in error_msg or "quota" in error_msg
#             if is_rate_limit and attempt < max_retries:
#                 print(f"  Rate limit hit (attempt {attempt}/{max_retries}). "
#                       f"Waiting {RETRY_WAIT_SECONDS}s before retrying...")
#                 time.sleep(RETRY_WAIT_SECONDS)
#             elif attempt < max_retries:
#                 print(f"  Error (attempt {attempt}/{max_retries}): {e}. Retrying...")
#                 time.sleep(5)
#             else:
#                 return f"[ERROR after {max_retries} attempts: {e}]"


# def run_all(top_k: int = TOP_K):
#     records = load_dataset()
#     existing = load_existing_results()

#     # Only count a question as "done" if it succeeded — errors get retried
#     existing_by_id = {r["id"]: r for r in existing}
#     done_ids = {qid for qid, r in existing_by_id.items() if not is_error_result(r["generated_answer"])}

#     results = [r for r in existing if r["id"] in done_ids]
#     remaining = [r for r in records if r["id"] not in done_ids]

#     print(f"Total questions: {len(records)} | Already done: {len(done_ids)} | Remaining: {len(remaining)}\n")

#     for i, record in enumerate(remaining, 1):
#         question = record["question"]
#         print(f"[{i}/{len(remaining)}] {record['id']}: {question}")

#         try:
#             chunks = search(question, top_k=top_k)
#             answer = generate_with_retry(question, chunks)

#             results.append({
#                 "id": record["id"],
#                 "question": question,
#                 "category": record["category"],
#                 "expected_source_contains": record.get("expected_source_contains", []),
#                 "expected_answer_keypoints": record.get("expected_answer_keypoints", []),
#                 "retrieved_sources": [c["source"] for c in chunks],
#                 "retrieved_texts": [c["text"] for c in chunks],
#                 "retrieved_scores": [c["score"] for c in chunks],
#                 "generated_answer": answer,
#             })

#             save_results(results)  # save after every question

#         except DailyQuotaExceeded as e:
#             save_results(results)
#             print(f"\nDaily token quota exceeded on Groq's free tier.")
#             print(f"Progress saved ({len(results)}/{len(records)} done).")
#             print(f"Wait for the quota to reset (rolling 24h window), then re-run this script to continue.")
#             return

#         except KeyboardInterrupt:
#             save_results(results)
#             print("\nInterrupted by user. Progress saved — re-run this script to resume.")
#             return

#         except Exception as e:
#             print(f"  Unexpected error on {record['id']}: {e}. Skipping for now.")

#         time.sleep(0.5)

#     print(f"\nDone. {len(results)}/{len(records)} results saved to {RESULTS_PATH}")


# if __name__ == "__main__":
#     run_all()