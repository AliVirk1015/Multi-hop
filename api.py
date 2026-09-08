from __future__ import annotations
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import llm_client  # noqa: E402
from retrieval.query_encoder import BgeM3QueryEncoder  # noqa: E402
from retrieval.qdrant_store import QdrantHybridStore  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
from retrieval_graph.graph import build_retrieval_graph  # noqa: E402
from retrieval_graph.state import fresh_state  # noqa: E402
from Graph.query import KnowledgeGraphClient  # noqa: E402
from main_query import make_llm_planner, DEFAULT_MAX_HOPS, DEFAULT_TOP_K, DEFAULT_RERANK_K  # noqa: E402

# --------------------------------------------------------------------------- #
# Lazy heavy singletons (built once at startup)
# --------------------------------------------------------------------------- #
_encoder: Any = None
_store: Any = None
_reranker: Any = None
_kg: Any = None
_graph: Any = None
_status = {"models": "down", "store": "down", "kg": "down"}


def _ensure_resources() -> None:
    global _encoder, _store, _reranker, _kg, _graph
    if _encoder is None:
        _encoder = BgeM3QueryEncoder()
        _status["models"] = "up"
    if _store is None:
        _store = QdrantHybridStore(query_encoder=_encoder)
        _status["store"] = "up"
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    if _kg is None:
        try:
            _kg = KnowledgeGraphClient()
            _status["kg"] = "up"
        except Exception as exc:
            print(f"[api] Neo4j KG unavailable ({exc}); graph expansion disabled")
            _status["kg"] = "down"
    if _graph is None:
        planner = make_llm_planner() if llm_client.llm_reachable() else None
        _graph = build_retrieval_graph(
            _store, _encoder, _reranker, planner=planner,
            max_hops=DEFAULT_MAX_HOPS, kg=_kg, kg_hops=1, kg_max_nodes=12,
        )


def ensure_resources() -> dict:
    """Public: lazily load all heavy resources once; return a status dict.

    Safe to call repeatedly (idempotent) — used by the FastAPI lifespan and
    the Streamlit frontend warm-up screen.
    """
    _ensure_resources()
    return {
        "models": _status.get("models"),
        "store": _status.get("store"),
        "kg": _status.get("kg"),
        "llm_reachable": llm_client.llm_reachable(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] warming up resources (model load can take a while)...")
    _ensure_resources()
    print(f"[api] ready  resources={_status}  llm={llm_client.llm_reachable()}")
    yield


app = FastAPI(title="Legal Multi-Hop Graph-RAG API", version="1.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str = Field(..., description="The legal question")
    use_llm: bool = True
    multi_hop: bool = True
    history: list = Field(
        default_factory=list,
        description="Prior turns: [{'role': 'user'|'assistant', 'content': str}]",
    )


# --------------------------------------------------------------------------- #
# Answer synthesis (evidence + graph case-law -> cited answer)
# --------------------------------------------------------------------------- #
def _synthesize(evidence: list, kg_evidence: list, query: str,
                history: Optional[list] = None) -> str:
    """Turn evidence (+ optional conversation history) into a cited answer."""
    docs = "\n\n".join(
        f"[{i}] {e.get('doc')} — Section {e.get('section_no')} "
        f"({e.get('section_title') or 'n/a'}) [{e.get('doc_type')}]\n"
        f"{(e.get('text') or '')[:1200]}"
        for i, e in enumerate(evidence[:8], 1)
    )
    case_law = ""
    if kg_evidence:
        lines = []
        for i, j in enumerate([x for x in kg_evidence if x.get("_kg_source") == "CITES"][:5], 1):
            lines.append(
                f"[C{i}] {j.get('case_no')} — {j.get('court') or 'court n/a'} "
                f"(outcome: {j.get('outcome') or 'n/a'})"
            )
        if lines:
            case_law = "\n\nRelated case law (from the knowledge graph):\n" + "\n".join(lines)

    # Prior turns give the model conversational context ("this article", "that case").
    conversation = ""
    turns = _format_history(history)
    if turns:
        conversation = (
            "\n\nConversation so far (answer the LATEST question using this context; "
            "resolve references such as 'this article' or 'that case'):\n" + turns
        )

    return llm_client.llm_chat(
        system=(
            "You are a Pakistani legal research assistant specialising in cyber-crime and "
            "criminal law. Answer the user's question using ONLY the provided evidence and "
            "cite the exact act + section number for every claim.\n"
            "Structure your answer with these markdown headings:\n"
            "## Direct Answer\n## Legal Basis\n## Analysis\n## Practical Guidance\n## Caveats\n"
            "Be direct and practical. Do NOT invent statutes, sections, or facts that are not "
            "in the evidence. If the evidence is insufficient, say so clearly and advise "
            "consulting a licensed lawyer. Respond in the same language as the question."
        ),
        user=(
            f"Question: {query}\n\nRelevant legal provisions:\n{docs}{case_law}"
            f"{conversation}"
        ),
        max_tokens=1200,
    )


def _format_history(history: Optional[list], limit: int = 6) -> str:
    """Collapse prior turns into a compact transcript (most recent first-ish)."""
    if not history:
        return ""
    lines = [
        f"{'User' if (h.get('role') or '').lower() == 'user' else 'Assistant'}: "
        f"{(h.get('content') or '').strip()}"
        for h in history[-limit:]
        if (h.get('content') or '').strip()
    ]
    return "\n".join(lines)


def _contextualize_query(query: str, history: Optional[list]) -> str:
    """Rewrite a follow-up question into a standalone searchable question.

    Resolves conversational references ("this article", "that case", "the above")
    so the retriever searches for the right provisions. Falls back to the raw
    question when there is no history or the LLM is unavailable.
    """
    if not history or not llm_client.llm_reachable():
        return query
    turns = _format_history(history)
    if not turns:
        return query
    try:
        rewritten = llm_client.llm_chat(
            system=(
                "You rewrite conversational follow-up questions into standalone "
                "questions for a legal statute search engine.\n"
                "Rules:\n"
                "- Detect references (e.g. 'this article', 'that section', 'the "
                "above', 'that act', 'it', 'the same', 'its', 'in that context') "
                "and replace them with the concrete act / section / case they refer "
                "to, taken from the history.\n"
                "- Preserve the original meaning. Output ONE question only.\n"
                "- If the latest question is already standalone, return it unchanged.\n"
                "- Reply with ONLY the rewritten question text — nothing else.\n"
                "\n"
                "Example:\n"
                "History:\n"
                "User: What is the punishment for qatl-i-amd under the Pakistan "
                "Penal Code?\n"
                "Assistant: Under the Pakistan Penal Code, qatl-i-amd is punishable "
                "under Section 302.\n"
                "Latest question: Does the same apply to its attempt?\n"
                "Rewritten: What is the punishment for attempting qatl-i-amd under "
                "the Pakistan Penal Code (Section 302)?\n"
                "\n"
                "Example 2:\n"
                "History:\n"
                "User: How is post-arrest bail handled in a PECA cyber crime case?\n"
                "Latest question: What about pre-arrest bail in that context?\n"
                "Rewritten: How is pre-arrest bail handled in a PECA cyber crime case?"
            ),
            user=f"History:\n{turns}\n\nLatest question: {query}",
            max_tokens=160,
            temperature=0.0,
        )
        rewritten = (rewritten or "").replace("\u00a0", " ").strip().strip('"').strip()
        return rewritten or query
    except Exception:
        return query


# --------------------------------------------------------------------------- #
# Evidence shaping for JSON responses
# --------------------------------------------------------------------------- #
def _ev_out(e: dict, trunc: int = 300) -> dict:
    return {
        "chunk_id": e.get("chunk_id"),
        "doc": e.get("doc"),
        "section_no": e.get("section_no"),
        "section_title": e.get("section_title"),
        "doc_type": e.get("doc_type"),
        "rerank_score": round(float(e.get("rerank_score", 0.0)), 4),
        "from_graph": bool(e.get("_kg")),
        "text": ((e.get("text") or "")[:trunc]),
    }


def _kg_out(j: dict) -> dict:
    return {
        "source": j.get("_kg_source", "kg"),
        "act": j.get("act"),
        "section_no": j.get("section_no"),
        "case_no": j.get("case_no"),
        "court": j.get("court"),
        "outcome": j.get("outcome"),
        "date": j.get("date"),
    }


# --------------------------------------------------------------------------- #
# Core answer logic (returns structured data, no printing)
# --------------------------------------------------------------------------- #
def answer(question: str, use_llm: bool = True, multi_hop: bool = True,
           history: Optional[list] = None) -> dict:
    if not question.strip():
        raise HTTPException(status_code=422, detail="query must be non-empty")
    _ensure_resources()

    # Resolve conversational references so retrieval searches the right provisions.
    search_q = _contextualize_query(question, history or [])

    hops = 1
    if multi_hop:
        out = _graph.invoke(fresh_state(search_q, max_hops=DEFAULT_MAX_HOPS))
        evidence = out.get("final_evidence", [])
        hops = int(out.get("current_hop", 1))
        kg_evidence = out.get("kg_evidence", [])
    else:
        results = _store.search(search_q, top_k=DEFAULT_TOP_K)
        evidence = _reranker.rerank(search_q, results, top_k=DEFAULT_RERANK_K)
        kg_evidence = []

    llm_on = use_llm and llm_client.llm_reachable()
    answer_text: Optional[str] = None
    if llm_on:
        try:
            answer_text = _synthesize(evidence, kg_evidence, question, history)
        except Exception as exc:
            answer_text = f"(LLM synthesis failed: {exc})"

    return {
        "query": question,
        "mode": "multi_hop" if multi_hop else "fast",
        "hops": hops,
        "llm_used": llm_on,
        "answer": answer_text,
        "evidence_count": len(evidence),
        "evidence": [_ev_out(e) for e in evidence[:12]],
        "kg_evidence_count": len(kg_evidence),
        "kg_evidence": [_kg_out(j) for j in kg_evidence[:12]],
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> dict:
    return {
        "service": "Legal Multi-Hop Graph-RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET  /health": "resource status (models / store / kg / llm)",
            "POST /query": "ask a legal question -> multi-hop retriever + Neo4j KG + LLM answer",
        },
        "docs": "/docs  (Swagger UI)",
        "example_post": {"query": "punishment for cyber stalking", "use_llm": True, "multi_hop": True},
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "resources": _status,
        "llm_reachable": llm_client.llm_reachable(),
    }


@app.post("/query")
def query(req: QueryRequest) -> dict:
    t0 = time.perf_counter()
    try:
        result = answer(
            req.query,
            use_llm=req.use_llm,
            multi_hop=req.multi_hop,
            history=req.history or None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"query failed: {exc}")
    result["latency_s"] = round(time.perf_counter() - t0, 3)
    return result
