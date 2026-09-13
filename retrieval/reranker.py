"""
Reranks a list of retrieved chunks using a local cross-encoder model
(bge-reranker-base), which scores question+chunk pairs directly for
more accurate relevance than pure embedding similarity.
"""
from FlagEmbedding import FlagReranker

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading reranker model: BAAI/bge-reranker-base (first run downloads it, ~280MB)")
        _reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
    return _reranker


def rerank(query: str, chunks: list, top_k: int = 4):
    if not chunks:
        return chunks

    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.compute_score(pairs)

    # compute_score returns a single float if only one pair, else a list
    if isinstance(scores, float):
        scores = [scores]

    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]