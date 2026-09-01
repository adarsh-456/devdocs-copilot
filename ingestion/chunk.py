"""
Reads everything in data/docs/ and data/issues/, splits it into overlapping
word-based chunks, and returns a list of chunk dicts ready for embedding.

This is intentionally simple (word-count based, not token-exact) so it has
zero extra dependencies. We'll compare smarter chunking strategies later.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = DATA_DIR / "docs"
ISSUES_DIR = DATA_DIR / "issues"

CHUNK_SIZE_WORDS = 220     # roughly ~300 tokens
CHUNK_OVERLAP_WORDS = 40


def split_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def load_doc_chunks():
    """Chunk every markdown file in data/docs/."""
    chunks = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, piece in enumerate(split_text(text)):
            chunks.append({
                "id": f"doc::{path.stem}::{i}",
                "text": piece,
                "source": f"docs/{path.name}",
                "type": "doc",
            })
    return chunks


def load_issue_chunks():
    """Chunk every issue JSON file in data/issues/ (title + body + comments together)."""
    chunks = []
    for path in sorted(ISSUES_DIR.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        full_text = (
            f"Issue title: {record['title']}\n\n"
            f"{record['body']}\n\n"
            + "\n\n".join(f"Comment: {c}" for c in record.get("comments", []))
        )
        for i, piece in enumerate(split_text(full_text)):
            chunks.append({
                "id": f"issue::{record['number']}::{i}",
                "text": piece,
                "source": f"GitHub Issue #{record['number']} - {record['title']} ({record['url']})",
                "type": "issue",
            })
    return chunks


def build_all_chunks():
    doc_chunks = load_doc_chunks()
    issue_chunks = load_issue_chunks()
    print(f"Doc chunks: {len(doc_chunks)}  |  Issue chunks: {len(issue_chunks)}")
    return doc_chunks + issue_chunks


if __name__ == "__main__":
    all_chunks = build_all_chunks()
    print(f"Total chunks: {len(all_chunks)}")
    if all_chunks:
        print("\nExample chunk:")
        print(all_chunks[0])