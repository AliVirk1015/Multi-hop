"""
Recall@5 for the CURRENT retrieval pipeline (fast path as used by main_query.py).

Pipeline under test:
    BGE-M3 encode (dense + sparse) -> QdrantHybridStore hybrid RRF search (top_k=20)
        -> cross-encoder rerank (top 6) -> take top 5

Ground truth: hand-built gold set from VERIFIED corpus facts (there is no labeled
eval dataset in this workspace). Statute queries are anchored to exact
(doc, section_no) pairs confirmed to exist in chunks.json; the judgment query is
anchored to judgment docs confirmed to discuss PECA bail.

Metric: Success@5 / Recall@5 (single-target queries) = fraction of queries where
at least one gold-relevant chunk appears in the top-5 reranked evidence.
Also reports retrieval-level (pre-rerank) top-5 for diagnosis.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.query_encoder import BgeM3QueryEncoder  # noqa: E402
from retrieval.qdrant_store import QdrantHybridStore  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402

# (query, [ (doc, section_no|None) ... ])  — None section = any chunk of that doc
GOLD = [
    ("punishment for qatl-i-amd intentional killing",
     [("Pakistan Penal Code.pdf", "302")]),
    ("punishment for theft under the penal code",
     [("Pakistan Penal Code.pdf", "379")]),
    ("procedure for registration of FIR information in cognizable cases",
     [("THE CODE OF CRIMINAL PROCEDURE, 1898.pdf", "154")]),
    ("unauthorized access to information system or data",
     [("PECA 2016.pdf", "3")]),
    ("offences against dignity of a natural person",
     [("PECA 2016.pdf", "20")]),
    ("punishment for cyber stalking",
     [("PECA 2016.pdf", "24")]),
    ("punishment for criminal breach of trust",
     [("Pakistan Penal Code.pdf", "406")]),
    ("post arrest bail in a PECA cyber crime case",
     [("j27.pdf", None), ("j69.pdf", None), ("j40.pdf", None), ("j9.pdf", None)]),
]

RETRIEVE_TOP_K = 20
RERANK_TOP_K = 6
TAKE = 5


def _hit(ev: dict, gold: list) -> bool:
    edoc = ev.get("doc")
    esec = str(ev.get("section_no")) if ev.get("section_no") is not None else None
    for gdoc, gsec in gold:
        if edoc == gdoc and (gsec is None or esec == gsec):
            return True
    return False


def _fmt(ev: dict) -> str:
    return f"{ev.get('doc')} §{ev.get('section_no')} ({ev.get('doc_type')}) s={ev.get('rerank_score', 0.0):.3f}"


def main() -> None:
    print("loading encoder / store / reranker (first run is slow)...", flush=True)
    enc = BgeM3QueryEncoder()
    store = QdrantHybridStore(query_encoder=enc)
    reranker = CrossEncoderReranker()

    ret_hits, rr_hits = 0, 0
    print(f"\n{'QUERY':<58} {'retrv@5':>8} {'rerank@5':>9}")
    print("-" * 78)
    for query, gold in GOLD:
        results = store.search(query, top_k=RETRIEVE_TOP_K)          # pre-rerank
        ret_top5 = results[:TAKE]
        reranked = reranker.rerank(query, results, top_k=RERANK_TOP_K)
        rr_top5 = reranked[:TAKE]

        r_hit = any(_hit(e, gold) for e in ret_top5)
        p_hit = any(_hit(e, gold) for e in rr_top5)
        ret_hits += r_hit
        rr_hits += p_hit

        mark = "HIT " if p_hit else "MISS"
        print(f"{query[:58]:<58} {str(r_hit):>8} {str(p_hit):>9}   {mark}")
        if not p_hit:
            print("    gold      : " + "; ".join(f"{d}§{s}" for d, s in gold))
            print("    rerank@5  : " + " | ".join(_fmt(e) for e in rr_top5))
        else:
            hits = [e for e in rr_top5 if _hit(e, gold)]
            print("    matched   : " + " | ".join(_fmt(e) for e in hits))

    n = len(GOLD)
    print("-" * 78)
    print(f"Retrieval-level Recall@5 : {ret_hits}/{n} = {ret_hits / n:.2%}")
    print(f"Pipeline  (rerank) Recall@5: {rr_hits}/{n} = {rr_hits / n:.2%}")


if __name__ == "__main__":
    main()
