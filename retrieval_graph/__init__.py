"""
R4 — LangGraph multi-hop retriever for the Qdrant hybrid pipeline.

Mirrors the Agent folder's structure (state.py / nodes.py / graph.py) but is
self-contained and runs without an LLM: multi-hop is driven by parent-child
(small-to-big) expansion plus optional LLM planner injection.
"""
from .state import RetrievalState, fresh_state
from .graph import build_retrieval_graph

__all__ = ["RetrievalState", "fresh_state", "build_retrieval_graph"]
