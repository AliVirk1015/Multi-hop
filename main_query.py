from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm_client  # noqa: E402
from retrieval.query_encoder import BgeM3QueryEncoder  # noqa: E402
from retrieval.qdrant_store import QdrantHybridStore  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
from retrieval_graph.graph import build_retrieval_graph  # noqa: E402
from retrieval_graph.state import fresh_state  # noqa: E402
from Graph.query import KnowledgeGraphClient  # noqa: E402

DEFAULT_TOP_K = 20          # hybrid candidates fetched in the fast path
DEFAULT_RERANK_K = 6        # reranked evidence fed to the LLM
# Lean default = 1 hop: with a single hop the graph ends at "hop_limit" and the
# optional LLM hop-planner is never called (biggest latency saver). Raise to 2
# (or more) to re-enable full multi-hop/planner behaviour for deep demos.
DEFAULT_MAX_HOPS = 1        # multi-hop graph depth
EVIDENCE_PRINT = 8

# ----------------------------------------------------------------------
# Heavy singletons — the encoder + reranker are GB-scale model loads.
# Build them ONCE and reuse across queries (interactive mode was reloading
# both models on every query, which is most of the latency).
# ----------------------------------------------------------------------
_encoder: Any = None
_store: Any = None
_reranker: Any = None
_graph: Any = None
_planner: Any = None
_graph_max_hops = 0
_kg: Any = None      # lazy KnowledgeGraphClient (None if Neo4j unavailable)


def _get_resources(multi_hop: bool, max_hops: int = DEFAULT_MAX_HOPS):
    """Lazily build (and cache) encoder / store / reranker / graph / kg."""
    global _encoder, _store, _reranker, _graph, _planner, _graph_max_hops, _kg
    if _encoder is None:
        _encoder = BgeM3QueryEncoder()
    if _store is None:
        _store = QdrantHybridStore(query_encoder=_encoder)
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    if _kg is None:
        try:
            _kg = KnowledgeGraphClient()
        except Exception as exc:
            print(f"[main_query] Neo4j KG unavailable ({exc}); continuing without graph expansion")
            _kg = None
    if multi_hop and (_graph is None or _graph_max_hops != max_hops):
        if _planner is None:
            _planner = make_llm_planner() if llm_client.llm_reachable() else None
        _graph = build_retrieval_graph(
            _store, _encoder, _reranker, planner=_planner, max_hops=max_hops,
            kg=_kg, kg_hops=1, kg_max_nodes=12,
        )
        _graph_max_hops = max_hops
    return _encoder, _store, _reranker, _graph


def make_llm_planner():
    """LLM-driven next-hop planner (injected into the graph)."""
    def planner(query: str, evidence: list, hop: int) -> dict:
        try:
            sample = "\n".join(
                f"- {e['chunk_id']} (§{e.get('section_no')}): {(e.get('text') or '')[:150]}"
                for e in evidence[-8:]
            )
            result = llm_client.llm_chat_json(
                system=(
                    "You are a Pakistani Cyber Crime legal retrieval planner. Decide the NEXT retrieval hop.\n"
                    'Reply ONLY with JSON: {"next_query": "refined query string", '
                    '"filters": {"doc_type": "..." | null, "doc": "..." | null} | null, '
                    '"note": "why"}'
                ),
                user=f"Original question: {query}\n\nEvidence gathered so far:\n{sample}",
            )
            return {
                "next_query": result.get("next_query") or query,
                "filters": result.get("filters") or None,
                "note": result.get("note", ""),
            }
        except Exception as exc:
            return {"next_query": query, "filters": None, "note": f"planner fallback: {exc}"}
    return planner


def synthesize_answer(evidence: list, query: str) -> str:
    """Ask the LLM to resolve the question into a structured, cited answer."""
    docs = "\n\n".join(
        f"[{i}] {e.get('doc')} — Section {e.get('section_no')} "
        f"({e.get('section_title') or 'n/a'}) [{e.get('doc_type')}]\n"
        f"{(e.get('text') or '')[:1200]}"
        for i, e in enumerate(evidence[:8], 1)
    )
    return llm_client.llm_chat(
        system=(
            "You are a Pakistani legal research assistant specialising in cyber-crime and "
            "criminal law. Answer the user's question using ONLY the provided statute "
            "evidence and cite the exact act + section number for every claim.\n"
            "Structure your answer with these markdown headings:\n"
            "## Direct Answer\n## Legal Basis\n## Analysis\n## Practical Guidance\n## Caveats\n"
            "Be direct and practical. Do NOT invent statutes, sections, or facts that are not "
            "in the evidence. If the evidence is insufficient, say so clearly and advise "
            "consulting a licensed lawyer. Respond in the same language as the question."
        ),
        user=f"Question: {query}\n\nRelevant legal provisions:\n{docs}",
        max_tokens=1200,
    )


def print_evidence(evidence: list, limit: int = EVIDENCE_PRINT) -> None:
    """Compact evidence appendix (for --no-llm / failure fallback)."""
    print("\n-- retrieved evidence --")
    for i, e in enumerate(evidence[:limit], 1):
        txt = (e.get("text") or "").strip().replace("\n", " ")
        print(
            f"  [{i}] {e.get('doc')} §{e.get('section_no')} ({e.get('doc_type')}) "
            f"rerank={e.get('rerank_score', 0.0):.3f} | {txt[:100]}"
        )


def run(
    query: str,
    use_llm: bool = True,
    multi_hop: bool = False,
    top_k: int = DEFAULT_TOP_K,
    rerank_k: int = DEFAULT_RERANK_K,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> None:
    enc, store, reranker, graph = _get_resources(multi_hop, max_hops)
    llm_on = use_llm and llm_client.llm_reachable()
    t0 = time.perf_counter()

    if multi_hop:
        out = graph.invoke(fresh_state(query, max_hops=max_hops))
        evidence = out["final_evidence"]
        hops = out["current_hop"]
    else:
        from retrieval.pipeline import retrieve_and_rerank

        evidence = retrieve_and_rerank(
            store, reranker, query, retrieve_top_k=top_k, rerank_top_k=rerank_k
        )
        hops = 1

    latency = time.perf_counter() - t0

    print(f"\nQ: {query}")
    print(
        f"[{'multi-hop' if multi_hop else 'fast'} path | {hops} hop(s) | "
        f"{len(evidence)} evidence | retrieval {latency:.1f}s]"
    )

    if not llm_on:
        if not use_llm:
            print("\n(LLM synthesis disabled via --no-llm — showing evidence only.)")
        else:
            print(
                "\n(No LLM configured — the answer cannot be generated. Enable it with\n"
                "  LLM_BASE_URL + LLM_MODEL in .env, or set OPENAI_API_KEY.)"
            )
        print_evidence(evidence, limit=EVIDENCE_PRINT)
        return

    try:
        print("\n" + "=" * 78)
        print(synthesize_answer(evidence, query))
        print("=" * 78)
    except Exception as exc:
        print(f"\n(LLM synthesis failed: {exc})")
        print_evidence(evidence, limit=EVIDENCE_PRINT)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ask a question and get an LLM-shaped, cited legal answer."
    )
    ap.add_argument("--query", default=None, help="single query (else interactive loop)")
    ap.add_argument("--no-llm", action="store_true", help="skip LLM synthesis (evidence only)")
    ap.add_argument(
        "--multi-hop",
        action="store_true",
        help="use the LangGraph multi-hop retriever (slower, opt-in)",
    )
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="hybrid candidates")
    ap.add_argument("--rerank-k", type=int, default=DEFAULT_RERANK_K, help="evidence sent to LLM")
    ap.add_argument("--max-hops", type=int, default=DEFAULT_MAX_HOPS, help="multi-hop depth")
    args = ap.parse_args()

    if args.query:
        run(
            args.query,
            use_llm=not args.no_llm,
            multi_hop=args.multi_hop,
            top_k=args.top_k,
            rerank_k=args.rerank_k,
            max_hops=args.max_hops,
        )
        return

    print("Interactive mode — type a question (blank line to quit).")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        try:
            run(
                q,
                use_llm=not args.no_llm,
                multi_hop=args.multi_hop,
                top_k=args.top_k,
                rerank_k=args.rerank_k,
                max_hops=args.max_hops,
            )
        except Exception as exc:
            print(f"\n[!] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
