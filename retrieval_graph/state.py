"""
RetrievalState — per-question working memory for the multi-hop retriever.

Single scope: everything resets per new question. Evidence accumulates across
hops (deduped by chunk_id in `evidence_pool`); `final_evidence` is the
reranked output consumed by callers (LLM synthesize / Agent adapter).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


class RetrievalState(TypedDict):
    # ---- question ----
    original_query: str
    current_query: str               # may be refined per hop (planner)
    filters: Optional[Dict[str, Any]]  # Qdrant filter spec for the current hop

    # ---- evidence ----
    evidence_pool: List[Dict[str, Any]]   # deduped across all hops
    hop_evidence: List[Dict[str, Any]]    # latest hop's retrieved+reranked

    # ---- multi-hop control ----
    current_hop: int
    max_hops: int
    last_score: float
    low_confidence: bool
    expansion_done: bool                  # has expand run at least once?
    last_expansion_added: bool            # last expansion added new evidence?
    decision: Literal["retrieve", "expand", "sufficient", ""]

    # ---- planner (optional LLM) ----
    next_query: str
    next_filters: Optional[Dict[str, Any]]
    planner_note: str

    # ---- output ----
    final_evidence: List[Dict[str, Any]]


def fresh_state(
    original_query: str,
    max_hops: int = 3,
    filters: Optional[Dict[str, Any]] = None,
) -> RetrievalState:
    return RetrievalState(
        original_query=original_query,
        current_query=original_query,
        filters=filters,
        evidence_pool=[],
        hop_evidence=[],
        current_hop=0,
        max_hops=max_hops,
        last_score=0.0,
        low_confidence=False,
        expansion_done=False,
        last_expansion_added=False,
        decision="",
        next_query="",
        next_filters=None,
        planner_note="",
        final_evidence=[],
    )
