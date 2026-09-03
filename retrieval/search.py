"""
Given a question, embeds it and retrieves the top-k most relevant chunks
from the Chroma index.

Usage:
    python -m retrieval.search
"""
from retrieval.embed_store import get_model, get_collection


def search(query: str, top_k: int = 5):
    model = get_model()
    collection = get_collection()

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(docs, metas, distances):
        hits.append({
            "text": text,
            "source": meta.get("source", "unknown"),
            "type": meta.get("type", "unknown"),
            "score": 1 - dist,  # convert distance to a rough similarity score
        })
    return hits


if __name__ == "__main__":
    q = "How do I add a background task in FastAPI?"
    for h in search(q):
        print(f"[{h['score']:.3f}] ({h['type']}) {h['source']}")
        print(f"   {h['text'][:150]}...\n")