# Evaluation Notes — Known Limitations

## Faithfulness metric (RAGAS) — small sample size

**Status:** Faithfulness was successfully computed for only **8 out of 20** sampled
questions (n=8), not the full sample.

**Why:** RAGAS's faithfulness metric issues multiple sequential LLM calls per
question (it breaks the generated answer into individual claims, then verifies
each claim against the retrieved context). On Groq's free tier — using
`openai/gpt-oss-20b`, a reasoning model with limited daily token quota (200,000
tokens/day) and inconsistent response times under this call pattern — most
attempts hit either `TimeoutError` or the daily rate limit before completing.

Two separate attempts were made (see git history), including reducing
concurrency to `max_workers=1` and shrinking the sample size, but the free
tier's reliability ceiling for this specific metric's workload was consistently
hit.

**Result on the successful subset (n=8):** Average Faithfulness = 0.794

**Answer Relevancy**, a lighter metric requiring fewer internal calls, was
computed reliably on the full sample (n=20): Average = 0.903

## Planned follow-up
Faithfulness evaluation should be re-run with a more capable or higher-quota
LLM API (e.g. a paid tier, or a different provider) once the project moves
past free-tier constraints, to get a statistically reliable measurement across
the full golden dataset (or at least matching the n=20 used for Answer
Relevancy).

## Files
- `eval/generation_metrics.json` — raw results from the successful run (n=20
  sampled, n=8 valid faithfulness scores, n=20 valid answer relevancy scores)
- `eval/compute_generation_metrics.py` — the script, configured conservatively
  (`max_workers=1`, `SAMPLE_SIZE=15`) after these findings