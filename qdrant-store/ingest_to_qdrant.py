import argparse
import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    PayloadSchemaType,
)

EMBEDDING_DIM = 1024  # BAAI/bge-m3 output size


def load_chunks(path: str):

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    stripped = content.lstrip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if stripped.startswith("{"):
            # Missing opening bracket -> patch it and retry
            fixed = "[" + stripped
            return json.loads(fixed)
        raise


def build_point(item: dict) -> PointStruct:
    chunk = item["chunk"]
    vector = item["embedding"]

    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding dim {len(vector)} for chunk_id={chunk.get('chunk_id')}"
        )

    # Qdrant point IDs must be an unsigned int or a UUID.
    # chunk_id (e.g. "Anti Money Laundring 2010:section:1") is a great unique
    # human-readable key, so keep it in the payload and derive a stable UUID
    # for the point ID (deterministic -> re-running ingestion won't duplicate).
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))

    payload = {
        "chunk_id": chunk.get("chunk_id"),
        "doc": chunk.get("doc"),
        "doc_type": chunk.get("doc_type"),
        "level": chunk.get("level"),
        "section_no": chunk.get("section_no"),
        "section_title": chunk.get("section_title"),
        "part": chunk.get("part"),
        "chapter": chunk.get("chapter"),
        "parent_id": chunk.get("parent_id"),
        "is_parent": chunk.get("is_parent"),
        "tokens": chunk.get("tokens"),
        "text": chunk.get("text"),
        "meta": chunk.get("meta", {}),
    }

    return PointStruct(id=point_id, vector=vector, payload=payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="embeddings_output.json")
    parser.add_argument("--collection", default="legal_chunks")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting (fresh start).",
    )
    args = parser.parse_args()

    print(f"Loading chunks from {args.file} ...")
    data = load_chunks(args.file)
    print(f"Loaded {len(data)} chunks.")

    client = QdrantClient(host=args.host, port=args.port)

    exists = client.collection_exists(args.collection)
    if exists and args.recreate:
        print(f"Deleting existing collection '{args.collection}' ...")
        client.delete_collection(args.collection)
        exists = False

    if not exists:
        print(f"Creating collection '{args.collection}' (dim={EMBEDDING_DIM}, cosine) ...")
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        # Index payload fields you'll want to filter on (domain routing,
        # doc-level filtering, hierarchical retrieval, etc.)
        for field, schema in [
            ("doc", PayloadSchemaType.KEYWORD),
            ("doc_type", PayloadSchemaType.KEYWORD),
            ("level", PayloadSchemaType.KEYWORD),
            ("chunk_id", PayloadSchemaType.KEYWORD),
            ("parent_id", PayloadSchemaType.KEYWORD),
            ("is_parent", PayloadSchemaType.BOOL),
        ]:
            client.create_payload_index(
                collection_name=args.collection,
                field_name=field,
                field_schema=schema,
            )
    else:
        print(f"Collection '{args.collection}' already exists — upserting into it.")

    points = [build_point(item) for item in data]

    print(f"Upserting {len(points)} points in batches of {args.batch_size} ...")
    for i in range(0, len(points), args.batch_size):
        batch = points[i : i + args.batch_size]
        client.upsert(collection_name=args.collection, points=batch)
        print(f"  upserted {min(i + args.batch_size, len(points))}/{len(points)}")

    info = client.get_collection(args.collection)
    print("\nDone.")
    print(f"Collection: {args.collection}")
    print(f"Points count: {info.points_count}")
    print(f"Vector size: {info.config.params.vectors.size}")


if __name__ == "__main__":
    main()