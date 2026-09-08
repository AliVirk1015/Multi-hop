"""
Create a NEW Qdrant collection with BOTH dense + sparse vectors and ingest the
BGE-M3 embeddings — exactly how `legal_hybrid` was built.

Input files (both must cover the same chunk_ids):
  dense  : embeddings_output.json        each item -> {"chunk": {...}, "embedding": [1024 floats]}
  sparse : sparse_embedded_chunks.json   each item -> {...chunk fields..., "sparse_vector": {token_id: weight}}

Why create the collection up-front with BOTH vector configs?
  On this stack (qdrant-client 1.19 <-> server 1.18.1) you CANNOT add a sparse
  vector to an existing dense-only collection (update_collection is a silent
  no-op). So a hybrid collection is created fresh with:
      vectors_config        = dense (unnamed "" vector, 1024-d cosine)
      sparse_vectors_config = {"sparse": SparseVectorParams(...)}

Point ids are deterministic uuid5(chunk_id) -> re-running never duplicates.

Usage (PowerShell, from the repo root):
  .venv\\Scripts\\python.exe qdrant-store/ingest_hybrid.py --collection my_new_coll
  .venv\\Scripts\\python.exe qdrant-store/ingest_hybrid.py --collection my_new_coll --limit 5   # test
  .venv\\Scripts\\python.exe qdrant-store/ingest_hybrid.py --collection my_new_coll --recreate # rebuild
"""
from __future__ import annotations

import argparse
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

EMBEDDING_DIM = 1024
SPARSE_NAME = "sparse"

# Payload fields surfaced as evidence + indexed on the server for fast filtering.
PAYLOAD_FIELDS = (
    "chunk_id", "doc", "doc_type", "level", "section_no", "section_title",
    "part", "chapter", "parent_id", "is_parent", "tokens", "text", "meta",
)
INDEXED_FIELDS = (
    ("doc", "keyword"), ("doc_type", "keyword"), ("level", "keyword"),
    ("chunk_id", "keyword"), ("parent_id", "keyword"), ("is_parent", "bool"),
)


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id derived from a chunk_id (matches ingest_to_qdrant.py)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def build_point(dense_item: dict, sparse_item: dict) -> PointStruct:
    chunk = dense_item["chunk"]
    dense = dense_item["embedding"]
    if len(dense) != EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding dim {len(dense)} for {chunk.get('chunk_id')}")

    lex = sparse_item["sparse_vector"]  # {token_id(str): weight}
    sparse_vec = SparseVector(
        indices=[int(k) for k in lex],
        values=[float(v) for v in lex.values()],
    )

    payload = {k: chunk.get(k) for k in PAYLOAD_FIELDS}
    payload["meta"] = payload.get("meta") or {}

    # "":   unnamed default vector = dense   |   "sparse": named sparse vector
    return PointStruct(
        id=point_id(chunk["chunk_id"]),
        vector={"": dense, SPARSE_NAME: sparse_vec},
        payload=payload,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Create + ingest a NEW hybrid (dense+sparse) Qdrant collection.")
    ap.add_argument("--collection", default="legal_hybrid_v2", help="new collection name")
    ap.add_argument("--dense", default="qdrant-store/embeddings_output.json")
    ap.add_argument("--sparse", default="qdrant-store/sparse_embedded_chunks.json")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="only ingest first N (testing)")
    ap.add_argument("--recreate", action="store_true", help="drop the collection first if it exists")
    args = ap.parse_args()

    client = QdrantClient(host=args.host, port=args.port)

    dense = {it["chunk"]["chunk_id"]: it for it in json.load(open(args.dense, encoding="utf-8"))}
    sparse = {it["chunk_id"]: it for it in json.load(open(args.sparse, encoding="utf-8"))}
    common = [cid for cid in dense if cid in sparse]
    if args.limit:
        common = common[: args.limit]
    print(f"dense={len(dense)} sparse={len(sparse)} overlapping={len(common)} (limit={args.limit or 'all'})")

    if client.collection_exists(args.collection):
        if not args.recreate:
            print(f"Collection '{args.collection}' already exists — use --recreate or a new name.")
            return
        print(f"Deleting '{args.collection}' ...")
        client.delete_collection(args.collection)

    print(f"Creating '{args.collection}' (dense 1024-d cosine + sparse '{SPARSE_NAME}') ...")
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),  # unnamed dense
        sparse_vectors_config={SPARSE_NAME: SparseVectorParams(index=SparseIndexParams(on_disk=True))},
    )

    for field, schema in INDEXED_FIELDS:
        client.create_payload_index(
            collection_name=args.collection, field_name=field, field_schema=schema
        )

    points = [build_point(dense[cid], sparse[cid]) for cid in common]
    for i in range(0, len(points), args.batch_size):
        batch = points[i : i + args.batch_size]
        client.upsert(collection_name=args.collection, points=batch)
        print(f"  upserted {min(i + args.batch_size, len(points))}/{len(points)}")

    info = client.get_collection(args.collection)
    print("\nDone.")
    print(f"Collection : {args.collection}")
    print(f"Points     : {info.points_count}")
    print(f"Dense size : {info.config.params.vectors.size}")
    print(f"Sparse     : {list(info.config.params.sparse_vectors.keys()) if info.config.params.sparse_vectors else None}")


if __name__ == "__main__":
    main()
