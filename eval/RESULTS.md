# Stage 2 Results: RAG Evaluation Summary

This document summarizes the evaluation work performed on DevDocs Copilot's
RAG pipeline: the baseline measurement, four pipeline-variant experiments,
and the one change that was actually adopted into the live system as a
result.

All experiments are logged in MLflow (`mlflow.db` / `mlruns/`, tracked in
this repo) under the experiment `devdocs-copilot-rag-eval`.

---

## 1. Golden Dataset

- **114 questions**, validated against the actual indexed corpus (155 docs,
  20 GitHub issues) — every `expected_source_contains` reference confirmed
  to exist before any metric was trusted (see `eval/validate_dataset.py`).
- 15 categories, with `factual` (42), `edge_case` (17), and `issue_based`
  (14, added specifically to cover GitHub issue retrieval) as the
  statistically meaningful ones. Several categories have n=1-4 and should
  be read as directional only, not conclusive.

## 2. Baseline (Stage 1 pipeline, as originally built)

**Configuration:** chunk_size=220 words, overlap=40, `bge-small-en-v1.5`
embeddings, Chroma (cosine similarity), top_k=5, `openai/gpt-oss-20b` via Groq.

| Metric | Score | Sample size |
|---|---|---|
| Hit Rate@5 | 0.904 | 114 |
| MRR | 0.786 | 114 |
| Precision@5 | 0.421 | 114 |
| Faithfulness (RAGAS) | 0.794 | 8 (see limitation below) |
| Answer Relevancy (RAGAS) | 0.903 | 20 |

**Known limitation — Faithfulness sample size:** RAGAS's faithfulness metric
issues multiple sequential LLM calls per question. On Groq's free tier
(`openai/gpt-oss-20b`, 200,000 tokens/day), most attempts hit timeouts or
the daily quota before completing. Full details and root-cause analysis in
`eval/NOTES.md`. Answer Relevancy, a lighter metric, was reliably computed
on the full sample.

## 3. Pipeline Variant Experiments

Four dimensions were tested against the golden dataset. All retrieval-only
metrics (Hit Rate, MRR, Precision) cost zero API tokens — computed entirely
with the local embedding model — so every variant below was tested on all
114 questions.

### 3a. `top_k` sweep (2, 3, 4, 5, 6, 8)

| top_k | Hit Rate | MRR | Precision |
|---|---|---|---|
| 2 | 0.816 | 0.759 | 0.588 |
| 3 | 0.877 | 0.779 | 0.518 |
| **4** | **0.904** | **0.786** | **0.461** |
| 5 (baseline) | 0.904 | 0.786 | 0.421 |
| 6 | 0.904 | 0.786 | 0.382 |
| 8 | 0.912 | 0.787 | 0.322 |

**Finding:** Hit Rate/MRR plateau from top_k=4 onward; Precision decreases
monotonically as top_k increases. **top_k=4 is the lowest value that
achieves peak Hit Rate/MRR**, making it strictly better than the baseline
(same accuracy, better precision) and better than any higher value (same
or worse accuracy, worse precision).

**✅ Adopted.**

### 3b. Chunk size (120, 220, 350 words)

| Chunk size | Hit Rate | MRR | Precision |
|---|---|---|---|
| 120 | 0.877 | 0.729 | 0.493 |
| **220 (baseline)** | **0.904** | **0.786** | 0.461 |
| 350 | 0.904 | 0.762 | 0.414 |

**Finding:** 220 words (the original Stage 1 choice) outperforms both
smaller and larger alternatives on MRR, and ties for best Hit Rate. Smaller
chunks lose surrounding context (hurts ranking); larger chunks dilute
embedding specificity (also hurts ranking).

**❌ Not changed — original choice confirmed as near-optimal.**

### 3c. Hybrid search (BM25 + embeddings, Reciprocal Rank Fusion)

| Config | Hit Rate | MRR | Precision |
|---|---|---|---|
| Embedding-only (top_k=4) | **0.904** | 0.786 | 0.461 |
| Hybrid (top_k=4) | 0.895 | 0.786 | **0.496** |

**Finding:** Small precision gain, small hit-rate loss, MRR unchanged —
not a clear win. Likely explanation: this corpus (clean technical docs) is
already well-served by semantic search; BM25's keyword-matching strength
doesn't add much here and occasionally pulls in superficially-matching but
less relevant chunks.

**❌ Not adopted.**

### 3d. Reranker (`bge-reranker-base` cross-encoder, top-15 → top-4)

| Config | Hit Rate | MRR | Precision |
|---|---|---|---|
| Embedding-only (top_k=4) | **0.904** | 0.786 | 0.461 |
| Reranker (top_k=4) | 0.868 | 0.786 | 0.467 |

**Finding:** Hit Rate decreased noticeably; MRR unchanged; Precision barely
moved. Reranking can only work with what the first-stage retrieval already
found — if the correct chunk isn't in the initial top-15 candidates, no
amount of reranking recovers it. Added meaningful latency for no net gain.

**❌ Not adopted.**

## 4. Final Adopted Configuration

| Setting | Baseline | Final |
|---|---|---|
| Chunk size | 220 words | 220 words (unchanged) |
| top_k | 5 | **4** |
| Hybrid search | No | No |
| Reranker | No | No |

**This change was applied to the live system**, not just recorded here:
`retrieval/search.py` and `generation/pipeline.py`'s defaults were updated
from `top_k=5` to `top_k=4`, and the Streamlit UI's slider default/range
was updated accordingly (`min=3, max=8, value=4`).

## 5. Reflections / What This Process Demonstrated

- Most experiments (3 of 4) did **not** improve on the baseline — this is
  normal and expected in real evaluation work, not a failure of the process.
- The one adopted change (`top_k=4`) was a genuine, evidence-backed
  improvement: same retrieval accuracy, meaningfully less irrelevant context
  per question.
- Retrieval-only metrics (free, local) were used to screen all variants
  cheaply; expensive LLM-judged metrics (RAGAS) were reserved for the
  baseline only, avoiding repeated free-tier quota exhaustion.
- Context Recall (as distinct from Hit Rate) was not computed — the golden
  dataset's ground truth format (1-2 expected sources per question) makes
  Hit Rate@k functionally equivalent to Recall for this dataset's structure.
  A dataset with exhaustively labeled multi-chunk relevance would be needed
  for true Recall.