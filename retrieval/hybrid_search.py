"""
Hybrid retrieval: combines embedding search (Chroma) with BM25 keyword
search, merged via Reciprocal Rank Fusion (RRF).
"""
from retrieval.search import search as embedding_search
from retrieval.bm25_index import bm25_search

RRF_K = 60  # standard constant used in RRF, dampens the impact of very top ranks


def hybrid_search(query, top_k=4, collection_name="devdocs_copilot", fetch_k=20, chunk_size=None, overlap=None):
    embed_results = embedding_search(query, top_k=fetch_k, collection_name=collection_name)
    bm_results = bm25_search(query, top_k=fetch_k, chunk_size=chunk_size, overlap=overlap)

    scores = {}
    meta = {}
    for rank_list in (embed_results, bm_results):
        for rank, item in enumerate(rank_list, start=1):
            key = item["text"]  # exact chunk text as the dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            meta[key] = item

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    fused = []
    for key in ranked_keys[:top_k]:
        item = dict(meta[key])
        item["score"] = scores[key]
        fused.append(item)
    return fused