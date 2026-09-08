from __future__ import annotations

from langgraph.graph import StateGraph, END

from .state import RetrievalState
from .nodes import make_retrieval_nodes


def build_retrieval_graph(
    store,
    encoder=None,
    reranker=None,
    planner=None,
    max_hops: int = 3,
    top_k: int = 20,
    rerank_top_k: int = 10,
    kg=None,                 # KnowledgeGraphClient (Neo4j) — optional graph expansion
    kg_hops: int = 1,
    kg_max_nodes: int = 12,
):
    """Compile the multi-hop retriever.

    store    : QdrantHybridStore (must accept string queries -> needs
               query_encoder; `encoder` is wired in if not already set)
    reranker : CrossEncoderReranker
    planner  : optional LLM planner(query, evidence, hop) -> dict
    """
    if getattr(store, "query_encoder", None) is None and encoder is not None:
        store.query_encoder = encoder

    nodes = make_retrieval_nodes(
        store,
        reranker,
        planner=planner,
        max_hops=max_hops,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        kg=kg,
        kg_hops=kg_hops,
        kg_max_nodes=kg_max_nodes,
    )

    graph = StateGraph(RetrievalState)

    graph.add_node("initialize", nodes["initialize"])
    graph.add_node("router", nodes["router"])
    graph.add_node("retrieve", nodes["retrieve"])
    graph.add_node("rerank", nodes["rerank"])
    graph.add_node("expand", nodes["expand"])
    graph.add_node("kg_expand", nodes["kg_expand"])
    graph.add_node("fallback_retrieve", nodes["fallback_retrieve"])
    graph.add_node("finalize", nodes["finalize"])

    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "router")

    graph.add_conditional_edges(
        "router",
        nodes["route_decision"],
        {"retrieve": "retrieve", "expand": "expand", "sufficient": "finalize"},
    )

    graph.add_edge("retrieve", "rerank")

    # When a KG client is available, good/hop_limit results pass through kg_expand
    # so graph-discovered provisions join the evidence pool before finalize.
    end_target = "kg_expand" if kg is not None else "finalize"
    graph.add_conditional_edges(
        "rerank",
        nodes["check_quality"],
        {
            "good": end_target,
            "poor": "fallback_retrieve",
            "expand": "expand",
            "hop_limit": end_target,
        },
    )

    graph.add_edge("fallback_retrieve", "retrieve")
    graph.add_edge("expand", "retrieve")
    if kg is not None:
        graph.add_edge("kg_expand", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
