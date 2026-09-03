# 📘 DevDocs Copilot

A RAG (Retrieval-Augmented Generation) system that answers developer questions
about **FastAPI** using its official documentation and real GitHub issues —
with citations, and grounded answers instead of hallucinated ones.

See [`PROJECT.md`](PROJECT.md) for the full project plan (problem statement,
scope, stages, success criteria).

**Current status: Stage 1 complete** — a full working pipeline: ingestion →
chunking → embedding → retrieval → generation → UI. Evaluation, observability,
and guardrails are planned next (see Roadmap below).

---

## Demo

Ask a question like *"How do I add a background task in FastAPI?"* and get
a grounded, cited answer:

> To add a background task in FastAPI you:
> 1. Import `BackgroundTasks` from `fastapi`
> 2. Define the task function (normal `def` or `async def`)
> 3. Call `background_tasks.add_task(...)` inside your path operation
>
> Sources: [Source 1] docs/tutorial__background-tasks.md, ...

---

## 1. Setup

```bash
git clone https://github.com/adarsh-456/devdocs-copilot.git
cd devdocs-copilot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### API keys
1. Copy `.env.example` to `.env`
2. Get a free Groq API key: https://console.groq.com/keys → add as `GROQ_API_KEY`
3. (Recommended) Get a free GitHub token: https://github.com/settings/tokens
   (no special scopes needed) → add as `GITHUB_TOKEN`, to avoid GitHub API
   rate limits (60/hour without a token vs 5,000/hour with one)

---

## 2. Build the data + index (one-time setup)

```bash
python ingestion/fetch_docs.py       # downloads FastAPI's official docs via git clone
python ingestion/fetch_issues.py     # downloads closed GitHub issues via GitHub API
python -m retrieval.embed_store      # chunks everything + builds the local Chroma index
```

This creates:
- `data/docs/` — 155 markdown documentation files
- `data/issues/` — closed GitHub issues as JSON (title, body, top comments)
- `chroma_db/` — local vector database (not committed to git — regenerate anytime)

---

## 3. Run it

**Web UI:**
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

**Command line (quick testing):**
```bash
python -m generation.generate
```

---

## 4. How it works

Question
│
▼
Embed question (bge-small-en-v1.5)
│
▼
Search Chroma vector DB (cosine similarity) → top-k relevant chunks
│
▼
Build grounded prompt (chunks + question + instructions to cite sources
and admit uncertainty if unsure)
│
▼
Groq API (openai/gpt-oss-20b) generates answer
│
▼
Answer + cited sources shown in UI


**Chunking strategy:** fixed-size word windows (~220 words) with 40-word
overlap, applied separately to docs and issues. Docs are chunked as-is;
issues are chunked as title + body + top 3 comments combined.

---

## 5. Project structure

devdocs-copilot/
├── data/ # downloaded docs + issues (generated, gitignored)
├── ingestion/
│ ├── fetch_docs.py # git-clones FastAPI repo, extracts docs/en/docs
│ ├── fetch_issues.py # pulls closed issues via GitHub API
│ └── chunks.py # splits docs/issues into overlapping chunks
├── retrieval/
│ ├── embed_store.py # embeds chunks (bge-small), builds Chroma index
│ └── search.py # embeds a query, retrieves top-k chunks
├── generation/
│ └── generate.py # builds grounded prompt, calls Groq API
├── eval/ # (Stage 2 — not yet implemented)
├── observability/ # (Stage 3 — not yet implemented)
├── guardrails/ # (Stage 4 — not yet implemented)
├── app.py # Streamlit UI
├── requirements.txt
├── .env.example
└── PROJECT.md # full project plan


---

## 6. Tech stack (100% free)

| Purpose | Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (local, via sentence-transformers) |
| Vector store | Chroma (local, persistent, cosine similarity) |
| LLM | Groq API (`openai/gpt-oss-20b`, free tier) |
| UI | Streamlit |
| Data source | FastAPI's public GitHub repo (docs + issues) |

---

## 7. Known limitations (honest, as of Stage 1)

- Chunking is simple fixed-size splitting — no smart splitting by
  section/heading yet, which sometimes hurts retrieval precision on
  narrow topics (e.g. startup/shutdown events vs `lifespan`).
- No formal accuracy measurement yet — retrieval and generation quality
  have only been spot-checked manually, not evaluated systematically.
  This is exactly what Stage 2 (evaluation) will add.
- No guardrails yet — the system doesn't yet explicitly detect
  off-topic/adversarial questions or refuse when confidence is low.

---

## 8. Roadmap

- [x] **Stage 1** — Ingestion → chunking → embedding → retrieval → generation → UI
- [ ] **Stage 1b** — Hybrid search (BM25 + embeddings) and re-ranking
- [ ] **Stage 2** — Evaluation harness (RAGAS) with a golden question set
- [ ] **Stage 3** — Observability with Langfuse tracing
- [ ] **Stage 4** — Guardrails (input filtering + grounding checks)