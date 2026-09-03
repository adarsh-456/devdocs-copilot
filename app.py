"""
DevDocs Copilot — Streamlit UI (runs locally, not deployed).

Run with:
    streamlit run app.py
"""
import streamlit as st
from pathlib import Path

from retrieval.search import search
from generation.generate import generate_answer

DOCS_DIR = Path(__file__).resolve().parent / "data" / "docs"
ISSUES_DIR = Path(__file__).resolve().parent / "data" / "issues"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"

st.set_page_config(page_title="DevDocs Copilot", page_icon="📘", layout="centered")

st.title("📘 DevDocs Copilot")
st.caption("Ask questions about FastAPI. Answers are grounded in official docs + GitHub issues, with citations.")

# --- Sidebar: index status ---
with st.sidebar:
    st.header("Index status")
    num_docs = len(list(DOCS_DIR.glob("*.md"))) if DOCS_DIR.exists() else 0
    num_issues = len(list(ISSUES_DIR.glob("*.json"))) if ISSUES_DIR.exists() else 0
    index_built = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())

    st.write(f"📄 Doc files: **{num_docs}**")
    st.write(f"💬 Issue files: **{num_issues}**")
    st.write(f"🔎 Vector index built: **{'Yes' if index_built else 'No'}**")

    if not index_built:
        st.warning("Index not found. Run `python -m retrieval.embed_store` in a terminal first.")
    else:
        st.success("Ready to answer questions.")

    top_k = st.slider("Number of sources to retrieve", min_value=2, max_value=10, value=5)

# --- Main: question input ---
question = st.text_input("Ask a question about FastAPI:", placeholder="How do I add a background task?")

if st.button("Ask", type="primary") and question.strip():
    if not index_built:
        st.error("Index isn't built yet. See the sidebar for setup steps.")
    else:
        with st.spinner("Retrieving relevant sources..."):
            chunks = search(question, top_k=top_k)

        if not chunks:
            st.warning("No relevant sources found for this question.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, chunks)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources retrieved")
            for i, c in enumerate(chunks, 1):
                with st.expander(f"[Source {i}] {c['source']}  (score: {c['score']:.3f})"):
                    st.write(c["text"])

st.divider()
st.caption("Local, non-deployed learning project — RAG pipeline over FastAPI docs + GitHub issues.")