"""
R4 smoke test — run the compiled multi-hop retrieval graph end-to-end (offline).

Run: .venv\Scripts\python.exe retrieval_graph\smoke_test_retrieval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.query_encoder import BgeM3QueryEncoder  # noqa: E402
from retrieval.qdrant_store import QdrantHybridStore  # noqa: E402
from retrieval.reranker import CrossEncoderReranker  # noqa: E402
from retrieval_graph.graph import build_retrieval_graph  # noqa: E402
from retrieval_graph.state import fresh_state  # noqa: E402

QUERY = "punishment for qatl-i-amd intentional killing"
PARENT_CHUNK = "Anti Money Laundring 2010:section:2"


def run_case(store, graph, query, label):
    state = fresh_state(query, max_hops=3)
    out = graph.invoke(state)
    print(f"\n=== {label} ===")
    print(f"  query      : {query}")
    print(f"  hops       : {out['current_hop']} / {out['max_hops']} | low_conf: {out['low_confidence']}")
    print(f"  pool size  : {len(out['evidence_pool'])} | final: {len(out['final_evidence'])}")
    for i, r in enumerate(out["final_evidence"][:5], 1):
        print(
            f"  [{i}] {r['chunk_id']}  §{r.get('section_no')}  level={r.get('level')}  "
            f"rerank={r.get('rerank_score', 0.0):.4f}  hop={r.get('_hop', '-')}"
        )
    return out


def main():
    enc = BgeM3QueryEncoder()
    store = QdrantHybridStore(query_encoder=enc)
    reranker = CrossEncoderReranker()
    graph = build_retrieval_graph(store, enc, reranker, planner=None, max_hops=3)

    out1 = run_case(store, graph, QUERY, "SCENARIO 1 — standard query")
    print(f"  [PASS] real PPC §302 present: "
          f"{any(r.get('section_no') == '302' for r in out1['final_evidence'])}")

    # Scenario 2: parent section text as query -> expand should add children.
    parent = store.get_by_chunk_id(PARENT_CHUNK)
    parent_text = (parent or {}).get("text") or "customer due diligence"
    out2 = run_case(store, graph, parent_text, "SCENARIO 2 — parent-section query (multi-hop)")

    child_ids = [r["chunk_id"] for r in out2["evidence_pool"]
                 if r.get("level") in ("paragraph", "clause")]
    print(f"  children in pool: {len(child_ids)} | hops: {out2['current_hop']}")
    assert out2["current_hop"] >= 2, "parent query should trigger >=2 hops"
    print("  [PASS] multi-hop expansion added child evidence")

    store.close()
    print("\nR4 SMOKE TEST COMPLETE")


if __name__ == "__main__":
    main()
