"""
Builds and queries a BM25 keyword-search index over the same chunks used
for embedding search, so it can be combined for hybrid retrieval.
"""
import re
from rank_bm25 import BM25Okapi
from ingestion.chunk import build_all_chunks

_cache = {}


def _tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def ensure_bm25_built(chunk_size: int = None, overlap: int = None):
    key = (chunk_size, overlap)
    if key not in _cache:
        kwargs = {}
        if chunk_size is not None:
            kwargs["chunk_size"] = chunk_size
        if overlap is not None:
            kwargs["overlap"] = overlap
        chunks = build_all_chunks(**kwargs)
        tokenized = [_tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        _cache[key] = (bm25, chunks)
    return _cache[key]


def bm25_search(query: str, top_k: int = 10, chunk_size: int = None, overlap: int = None):
    bm25, chunks = ensure_bm25_built(chunk_size, overlap)
    scores = bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in ranked_idx:
        c = chunks[idx]
        results.append({
            "text": c["text"],
            "source": c["source"],
            "type": c["type"],
            "score": float(scores[idx]),
        })
    return results