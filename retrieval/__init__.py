"""
Retrieval layer for the Qdrant hybrid retrieval pipeline.

R1 ships `QdrantHybridStore` — dense (BGE-M3) + sparse (BGE-M3 lexical)
hybrid search with metadata filtering against the `legal_hybrid` collection.
"""
from .qdrant_store import QdrantHybridStore, point_id

__all__ = ["QdrantHybridStore", "point_id"]
