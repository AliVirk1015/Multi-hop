"""
R4 nodes — node + edge functions for the multi-hop retriever graph.

Multi-hop strategy (LLM-optional):
  1. retrieve  : hybrid search (dense+sparse RRF) on current_query + filters, top_k.
  2. rerank     : cross-encoder on the hop's candidates.
  3. quality    : good -> finalize | poor -> fallback (broaden) | expand -> next hop.
  4. expand     : optional LLM `planner` refines query+filters, OR deterministic
                  parent-child (small-to-big) expansion via payload parent_id.
  5. finalize   : dedupe across hops and one last rerank on the combined pool.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .state import RetrievalState

# planner(query, evidence, hop) -> {"next_query": str, "filters": dict|None, "note": str}
Planner = Callable[[str, List[Dict[str, Any]], int], Dict[str, Any]]


def make_retrieval_nodes(
    store,
    reranker,
    planner: Optional[Planner] = None,
    max_hops: int = 3,
    top_k: int = 20,
    rerank_top_k: int = 10,
    confidence_threshold: float = 0.4,
):
    def _record(state: RetrievalState, items: List[dict]) -> None:
        seen = {e["chunk_id"] for e in state["evidence_pool"] if e.get("chunk_id")}
        for r in items:
            cid = r.get("chunk_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            state["evidence_pool"].append(r)

    # ------------------------------------------------------------------
    def initialize(state: RetrievalState) -> RetrievalState:
        state["current_query"] = state["original_query"]
        state["evidence_pool"] = []
        state["hop_evidence"] = []
        state["current_hop"] = 0
        state["last_score"] = 0.0
        state["low_confidence"] = False
        state["expansion_done"] = False
        state["last_expansion_added"] = False
        state["decision"] = ""
        state["next_query"] = ""
        state["next_filters"] = None
        state["planner_note"] = ""
        state["final_evidence"] = []
        return state

    def router(state: RetrievalState) -> RetrievalState:
        # Optional LLM intent router goes here later; default is broad retrieval.
        state["decision"] = "retrieve"
        return state

    def route_decision(state: RetrievalState) -> str:
        if state["current_hop"] >= state["max_hops"]:
            return "sufficient"
        return state["decision"] if state["decision"] in ("retrieve", "expand") else "retrieve"

    def retrieve(state: RetrievalState) -> RetrievalState:
        state["current_hop"] += 1
        results = store.search(state["current_query"], filters=state["filters"], top_k=top_k)
        state["hop_evidence"] = results
        _record(state, results)
        state["last_score"] = float(results[0]["score"]) if results else 0.0
        state["low_confidence"] = len(results) == 0
        return state

    def rerank(state: RetrievalState) -> RetrievalState:
        if not state["hop_evidence"]:
            state["low_confidence"] = True
            return state
        ranked = reranker.rerank(state["current_query"], state["hop_evidence"], top_k=rerank_top_k)
        state["hop_evidence"] = ranked
        state["last_score"] = float(ranked[0].get("rerank_score", 0.0))
        state["low_confidence"] = len(ranked) == 0 or state["last_score"] < confidence_threshold
        return state

    def check_quality(state: RetrievalState) -> str:
        if state["current_hop"] >= state["max_hops"]:
            return "hop_limit"
        if state["low_confidence"]:
            return "poor"
        if planner is not None:
            return "expand"                      # LLM-driven next hop
        if not state.get("expansion_done"):
            return "expand"                      # run deterministic expansion at least once
        if state.get("last_expansion_added"):
            return "expand"                      # parent-child still adding new evidence
        return "good"

    def expand(state: RetrievalState) -> RetrievalState:
        state["decision"] = "retrieve"
        state["expansion_done"] = True

        # Optional LLM planner path.
        if planner is not None:
            try:
                plan = planner(state["original_query"], state["evidence_pool"], state["current_hop"])
                state["next_query"] = str(plan.get("next_query") or state["original_query"])
                state["next_filters"] = plan.get("filters")
                state["planner_note"] = str(plan.get("note", ""))
                state["current_query"] = state["next_query"]
                if state["next_filters"] is not None:
                    state["filters"] = state["next_filters"]
                state["last_expansion_added"] = True
                return state
            except Exception as exc:
                state["planner_note"] = f"planner error: {exc}"

        # Deterministic parent-child (small-to-big) expansion.
        added: List[dict] = []
        for e in state["evidence_pool"]:
            pid = e.get("parent_id")
            if pid:
                parent = store.get_by_chunk_id(str(pid))
                if parent:
                    parent["_hop"] = state["current_hop"]
                    added.append(parent)
            if e.get("is_parent"):
                for child in store.get_children(str(e["chunk_id"])):
                    child["_hop"] = state["current_hop"]
                    added.append(child)
        _record(state, added)
        state["last_expansion_added"] = len(added) > 0

        # Broaden for the next hybrid pass.
        state["current_query"] = state["original_query"]
        state["filters"] = None
        return state

    def fallback_retrieve(state: RetrievalState) -> RetrievalState:
        # Drop filters, re-run on the original query -> broader recall.
        state["current_query"] = state["original_query"]
        state["filters"] = None
        state["decision"] = "retrieve"
        return state

    def finalize(state: RetrievalState) -> RetrievalState:
        pool = state["evidence_pool"]
        if pool:
            pool = reranker.rerank(state["original_query"], pool, top_k=20)
        state["final_evidence"] = pool
        return state

    return {
        "initialize": initialize,
        "router": router,
        "retrieve": retrieve,
        "rerank": rerank,
        "expand": expand,
        "fallback_retrieve": fallback_retrieve,
        "finalize": finalize,
        "route_decision": route_decision,
        "check_quality": check_quality,
    }
