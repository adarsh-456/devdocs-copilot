"""
Embeds all chunks using a local sentence-transformers model and stores them
in a local, persistent Chroma database.

Usage:
    python -m retrieval.embed_store
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from ingestion.chunk import build_all_chunks

PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "devdocs_copilot"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (first run downloads it, ~130MB)")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index(batch_size: int = 64):
    chunks = build_all_chunks()
    if not chunks:
        print("No chunks found. Run ingestion/fetch_docs.py and ingestion/fetch_issues.py first.")
        return

    model = get_model()
    collection = get_collection()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [{"source": c["source"], "type": c["type"]} for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\nIndex built. Stored at: {PERSIST_DIR}")


if __name__ == "__main__":
    build_index()