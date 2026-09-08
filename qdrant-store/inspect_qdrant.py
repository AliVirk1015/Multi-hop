"""
inspect_qdrant.py — look at what is inside Qdrant and verify it was stored safely.

Same data is reachable via the raw REST API (good to learn both):

    # list collections
    curl.exe -s http://localhost:6333/collections

    # one collection's config (points, vectors, payload indexes)
    curl.exe -s http://localhost:6333/collections/<name>

    # exact point count
    curl.exe -s -X POST http://localhost:6333/collections/<name>/points/count ^
        -H "Content-Type: application/json" -d "{\"exact\": true}"

    # scroll sample points
    curl.exe -s -X POST http://localhost:6333/collections/<name>/points/scroll ^
        -H "Content-Type: application/json" -d "{\"limit\": 3, \"with_payload\": true, \"with_vector\": false}"

Usage (PowerShell):
    .venv\Scripts\python.exe qdrant-store/inspect_qdrant.py                          # list all collections + info
    .venv\Scripts\python.exe qdrant-store/inspect_qdrant.py --collection judgements  # info on one
    .venv\Scripts\python.exe qdrant-store/inspect_qdrant.py --collection judgements --scroll 5
    .venv\Scripts\python.exe qdrant-store/inspect_qdrant.py --collection judgements --chunk-id "2021LHC3627:preamble:1"
    .venv\Scripts\python.exe qdrant-store/inspect_qdrant.py --collection judgements --verify
"""
from __future__ import annotations

import argparse
import math
import uuid

from qdrant_client import QdrantClient


def point_id(chunk_id: str) -> str:
    """uuid5 point id — must match the ingest scripts or lookups return nothing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def vector_desc(params) -> str:
    v = getattr(params, "vectors", None)
    out: list[str] = []
    if v is None:
        out.append("(none)")
    elif isinstance(v, dict):
        for name, vp in v.items():
            out.append(f"{name or '<default>'}: {vp.size}-d {vp.distance}")
    else:  # a single VectorParams == the default (unnamed) vector
        out.append(f"<default>: {v.size}-d {v.distance}")
    sv = getattr(params, "sparse_vectors", None)
    if sv:
        for name in sv:
            out.append(f"sparse[{name}]")
    return " | ".join(out)


def show_info(client: QdrantClient, name: str) -> None:
    info = client.get_collection(name)
    print(f"  points_count   : {info.points_count}")
    print(f"  vectors        : {vector_desc(info.config.params)}")
    try:
        ps = getattr(info, "payload_schema", None) or {}
        if ps:
            print(f"  payload indexes: {', '.join(sorted(ps))}")
    except Exception:
        pass


def verify_collection(client: QdrantClient, name: str, sample: int) -> None:
    """Deep safety checks: ids resolve, dims correct, vectors normalized, no empties."""
    info = client.get_collection(name)
    print(f"[verify:{name}] points_count={info.points_count}")
    if info.points_count == 0:
        print("  !! collection is empty")
        return

    pts, _ = client.scroll(collection_name=name, limit=sample, with_payload=True, with_vectors=True)
    checked = 0
    for p in pts:
        vecs = p.vector if isinstance(p.vector, dict) else {"<default>": p.vector}
        dense = vecs.get("", vecs.get("<default>"))
        if dense is not None and len(dense):
            if len(dense) != 1024:
                print(f"  !! bad dense dim {len(dense)} on {p.id}")
            n = math.sqrt(sum(float(x) ** 2 for x in dense))
            if abs(n - 1.0) > 1e-2:
                print(f"  !! L2 norm {n:.4f} (not normalized) on {p.id}")
            checked += 1
        sparse = vecs.get("sparse")
        if sparse is not None and (not sparse.indices or not sparse.values):
            print(f"  !! empty sparse on {p.id}")
        payload = p.payload or {}
        if not payload.get("text"):
            print(f"  !! missing text payload on {p.id}")
    print(f"  checked {checked} sample vectors (dims + cosine-normalization)")

    try:
        cnt = client.count(collection_name=name, exact=True).count
        print(f"  exact count    : {cnt}")
    except Exception as exc:
        print(f"  exact count failed: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect Qdrant collections / verify stored embeddings.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--collection", default=None, help="limit to one collection (default: all)")
    ap.add_argument("--scroll", type=int, default=3, help="how many sample points to scroll")
    ap.add_argument("--chunk-id", default=None, help="fetch one point by its chunk_id")
    ap.add_argument("--verify", action="store_true", help="deep safety checks on stored vectors")
    args = ap.parse_args()

    client = QdrantClient(host=args.host, port=args.port)

    if args.chunk_id:
        cid = point_id(args.chunk_id)
        print(f"point id (uuid5 of '{args.chunk_id}') = {cid}")
        pts = client.retrieve(
            collection_name=args.collection, ids=[cid], with_vectors=True, with_payload=True
        )
        if not pts:
            print("  NOT FOUND")
            return
        p = pts[0]
        print("  found.")
        print("  payload.chunk_id:", p.payload.get("chunk_id"))
        print("  payload.doc     :", p.payload.get("doc"))
        print("  payload.level   :", p.payload.get("level"))
        vecs = p.vector if isinstance(p.vector, dict) else {"<default>": p.vector}
        print("  vector names    :", list(vecs.keys()))
        if "sparse" in vecs:
            print(f"  sparse nnz       : {len(vecs['sparse'].indices)}")
        return

    names = [c.name for c in client.get_collections().collections]
    print("collections:", names)
    cols = [args.collection] if args.collection else names
    for n in cols:
        print(f"\n=== {n} ===")
        show_info(client, n)
        if args.scroll:
            pts, _ = client.scroll(collection_name=n, limit=args.scroll, with_payload=True, with_vectors=False)
            print(f"  sample points ({len(pts)}):")
            for p in pts:
                pl = p.payload or {}
                print(f"    - {p.id}  doc={pl.get('doc')!r}  level={pl.get('level')!r}  "
                      f"chunk_id={pl.get('chunk_id')!r}")
    if args.verify:
        for n in cols:
            verify_collection(client, n, sample=max(1, min(30, args.scroll)))
            print()


if __name__ == "__main__":
    main()
