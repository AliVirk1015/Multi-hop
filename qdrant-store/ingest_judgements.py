"""
ingest_judgements.py — create the ``judgements`` collection and load the BGE-M3
hybrid (dense + sparse) embeddings from judgement_embeddings.json.

Input format — ONE merged file (dense + sparse together, per chunk):
    [
      {
        "id":              "2021LHC3627:preamble:1",
        "chunk_data":      { chunk_id, doc, doc_type, level, section_no,
                             section_title, part, chapter, parent_id,
                             is_parent, tokens, text, meta },
        "dense_embedding":  [1024 floats],                     # BGE-M3 dense
        "sparse_embedding": {token_id: weight}                 # BGE-M3 lexical
      }, ...
    ]

Collection layout (same recipe as ingest_hybrid.py, verified on this stack):
    vectors_config        = unnamed dense "" vector, 1024-d, COSINE
    sparse_vectors_config = {"sparse": SparseVectorParams(index=...)}
  * You CANNOT add a sparse vector to an existing dense-only collection here
    (update_collection is a silent no-op on qdrant-client 1.19 <-> server 1.18.1),
    so the hybrid collection must be created up-front with BOTH configs.

Point ids are deterministic uuid5(chunk_id) -> re-running never duplicates.

Usage (PowerShell, from the repo root — Qdrant must be running on :6333):
    .venv\Scripts\python.exe qdrant-store/ingest_judgements.py
    .venv\Scripts\python.exe qdrant-store/ingest_judgements.py --limit 5     # test only 5
    .venv\Scripts\python.exe qdrant-store/ingest_judgements.py --recreate    # drop + rebuild
"""
from __future__ import annotations

import argparse
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

EMBEDDING_DIM = 1024          # BAAI/bge-m3 dense output size
SPARSE_NAME = "sparse"

# Payload fields surfaced as evidence + indexed for fast filtering.
PAYLOAD_FIELDS = (
    "chunk_id", "doc", "doc_type", "level", "section_no", "section_title",
    "part", "chapter", "parent_id", "is_parent", "tokens", "text", "meta",
)
INDEXED_FIELDS = (
    ("doc", "keyword"), ("doc_type", "keyword"), ("level", "keyword"),
    ("chunk_id", "keyword"), ("parent_id", "keyword"), ("is_parent", "bool"),
)


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id derived from a chunk_id (matches the other ingest scripts)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def build_point(item: dict) -> PointStruct:
    chunk = item["chunk_data"]
    dense = item["dense_embedding"]
    if len(dense) != EMBEDDING_DIM:
        raise ValueError(f"Unexpected dense dim {len(dense)} for {chunk.get('chunk_id')}")
    lex = item["sparse_embedding"]  # {token_id(str): weight}
    if not lex:
        raise ValueError(f"Empty sparse vector for {chunk.get('chunk_id')}")
    sparse_vec = SparseVector(
        indices=[int(k) for k in lex],
        values=[float(v) for v in lex.values()],
    )
    payload = {k: chunk.get(k) for k in PAYLOAD_FIELDS}
    payload["meta"] = payload.get("meta") or {}
    return PointStruct(
        id=point_id(chunk["chunk_id"]),
        vector={"": dense, SPARSE_NAME: sparse_vec},   # "": dense  |  "sparse": sparse
        payload=payload,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create + ingest the hybrid 'judgements' collection from judgement_embeddings.json."
    )
    ap.add_argument("--file", default="qdrant-store/judgement_embeddings.json")
    ap.add_argument("--collection", default="judgements")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="only ingest first N (testing)")
    ap.add_argument("--recreate", action="store_true", help="drop the collection first if it exists")
    args = ap.parse_args()

    data = json.load(open(args.file, encoding="utf-8"))
    print(f"loaded {len(data)} embedded chunks from {args.file}")
    if args.limit:
        data = data[: args.limit]

    client = QdrantClient(host=args.host, port=args.port)

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

    points = [build_point(it) for it in data]
    for i in range(0, len(points), args.batch_size):
        batch = points[i : i + args.batch_size]
        client.upsert(collection_name=args.collection, points=batch)
        print(f"  upserted {min(i + args.batch_size, len(points))}/{len(points)}")

    info = client.get_collection(args.collection)
    v = info.config.params.vectors
    dense_desc = f"{v.size}-d {v.distance}" if not isinstance(v, dict) else "multi"
    print("\nDone.")
    print(f"Collection : {args.collection}")
    print(f"Points     : {info.points_count}")
    print(f"Dense      : {dense_desc}")
    print(f"Sparse     : {list(info.config.params.sparse_vectors.keys()) if info.config.params.sparse_vectors else None}")


if __name__ == "__main__":
    main()
