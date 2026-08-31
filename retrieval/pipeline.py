from __future__ import annotations

from typing import Optional


def retrieve_and_rerank(
    store,
    reranker,
    query: str,
    *,
    filters: Optional[dict] = None,
    retrieve_top_k: int = 20,
    rerank_top_k: int = 8,

) -> list[dict]:
    """Hybrid retrieve (top_k) then cross-encoder rerank (top_k) a text query."""
    results = store.search(query, filters=filters, top_k=retrieve_top_k)
    return reranker.rerank(query, results, top_k=rerank_top_k)
