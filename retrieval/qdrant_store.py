
from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Callable, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ----------------------------------------------------------------------
# Defaults (override via env vars or constructor args)
# ----------------------------------------------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_hybrid")
# Multi-collection retrieval: search statutes (legal_hybrid) AND judgements,
# so case law shows up in the evidence pool too. Override with a
# comma-separated list, e.g. QDRANT_COLLECTIONS="legal_hybrid,judgements".
QDRANT_COLLECTIONS = [
    c.strip()
    for c in os.getenv("QDRANT_COLLECTIONS", "legal_hybrid,judgements").split(",")
    if c.strip()
]

DENSE_VECTOR_NAME = ""            # unnamed default vector
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_TOP_K = 20
DEFAULT_PREFETCH_LIMIT = 50       # per-modality candidates fed into fusion
FUSION = qm.Fusion.RRF

# Payload fields indexed on the server (fast filtering).
INDEXED_FIELDS = ("doc", "doc_type", "level", "chunk_id", "parent_id", "is_parent")

# Payload keys surfaced into evidence records (AgentState-compatible shape).
EVIDENCE_FIELDS = (
    "chunk_id", "doc", "doc_type", "level", "section_no", "section_title",
    "part", "chapter", "parent_id", "is_parent", "tokens", "text", "meta",
)

QueryType = Union[str, dict]                       # dict = pre-encoded vectors
FilterSpec = Union[dict, qm.Filter, None]          # dict = field -> value(s)


def point_id(chunk_id: str) -> str:
    """Deterministic Qdrant point id derived from a chunk_id (matches ingest)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantHybridStore:
    def __init__(
        self,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        collection_name: Optional[str] = None,
        collection_names: Optional[list[str]] = None,
        query_encoder: Optional[Callable[[str], dict]] = None,
    ):
        self.client = QdrantClient(host=host, port=port)
        if collection_names is not None:
            self.collection_names = list(collection_names)
        elif collection_name is not None:
            self.collection_names = [collection_name]
        else:
            self.collection_names = list(QDRANT_COLLECTIONS)
        # Skip collections that don't exist yet (e.g. judgements not ingested)
        # so the default config stays safe on a fresh Qdrant.
        existing = [c for c in self.collection_names if self.client.collection_exists(c)]
        missing = [c for c in self.collection_names if c not in existing]
        if missing:
            print(f"[qdrant_store] skipping missing collection(s): {missing}", file=sys.stderr)
        self.collection_names = existing if existing else self.collection_names
        self.collection_name = self.collection_names[0] if self.collection_names else QDRANT_COLLECTION
        self.query_encoder = query_encoder

    # ------------------------------------------------------------------
    # Primary API — hybrid search
    # ------------------------------------------------------------------
    def search(
        self,
        query: QueryType,
        *,
        filters: FilterSpec = None,
        top_k: int = DEFAULT_TOP_K,
        prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    ) -> list[dict]:
        """
        Hybrid dense+sparse search with optional metadata filtering.

        query:
          - dict with "dense" (list[float]) and "sparse" (SparseVector or
            {"indices": [...], "values": [...]}) — pre-encoded vectors; or
          - str text — encoded via `query_encoder` (must be configured).

        filters:
          - dict {field: value}  -> exact match (value) or MatchAny (list)
          - qdrant Filter
          - None

        Returns evidence records sorted by fused (RRF) score descending.
        """
        dense, sparse = self._resolve_vectors(query)
        flt = self._build_filter(filters)

        # Hybrid dense+sparse RRF is run per collection; the per-collection
        # candidate lists are then merged (deduped) so one query returns
        # evidence from BOTH statutes (legal_hybrid) and judgements.
        merged: list[dict] = []
        seen: set = set()
        for coll in self.collection_names:
            for ev in self._search_one(coll, dense, sparse, flt, top_k, prefetch_limit):
                key = ev.get("chunk_id") or ev.get("point_id")
                if key in seen:
                    continue
                seen.add(key)
                ev["collection"] = coll
                merged.append(ev)
        # Fused (RRF) score descending. The caller reranks across the merged
        # pool anyway, so this order only shapes the candidate pool.
        merged.sort(key=lambda e: e.get("score") or 0.0, reverse=True)
        return merged

    def _search_one(
        self,
        collection_name: str,
        dense: list,
        sparse: qm.SparseVector,
        flt: Optional[qm.Filter],
        top_k: int,
        prefetch_limit: int,
    ) -> list[dict]:
        """Hybrid dense+sparse search (RRF fusion) inside a single collection."""
        prefetch = [
            # using=None -> default (unnamed) dense vector
            qm.Prefetch(query=dense, using=None, limit=prefetch_limit, filter=flt),
            qm.Prefetch(query=sparse, using=SPARSE_VECTOR_NAME, limit=prefetch_limit, filter=flt),
        ]
        res = self.client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=FUSION),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return self._format_scored(res.points)

    # ------------------------------------------------------------------
    # Deterministic lookups (payload filters, no vectors needed)
    # ------------------------------------------------------------------
    def exact_lookup(
        self,
        doc: Optional[str] = None,
        section_no: Optional[Any] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """Return chunks matching doc (+ optional section_no) via payload filter."""
        conds = []
        if doc is not None:
            conds.append(qm.FieldCondition(key="doc", match=qm.MatchValue(value=doc)))
        if section_no is not None:
            conds.append(qm.FieldCondition(key="section_no", match=qm.MatchValue(value=str(section_no))))
        flt = qm.Filter(must=conds) if conds else None
        points, _next = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=flt,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [self._evidence_from_payload(p.payload or {}, score=None, point_id=str(p.id)) for p in points]

    def get_by_chunk_id(self, chunk_id: str) -> Optional[dict]:
        """Fetch a single chunk by its deterministic point id (any collection)."""
        pid = point_id(chunk_id)
        for coll in self.collection_names:
            records = self.client.retrieve(
                collection_name=coll,
                ids=[pid],
                with_payload=True,
                with_vectors=False,
            )
            if records:
                r = records[0]
                ev = self._evidence_from_payload(r.payload or {}, score=None, point_id=str(r.id))
                ev["collection"] = coll
                return ev
        return None

    def get_children(self, parent_chunk_id: str, top_k: int = 20) -> list[dict]:
        """Child chunks of a parent (small-to-big expansion hop), any collection."""
        flt = qm.Filter(
            must=[qm.FieldCondition(key="parent_id", match=qm.MatchValue(value=parent_chunk_id))]
        )
        out: list[dict] = []
        for coll in self.collection_names:
            points, _next = self.client.scroll(
                collection_name=coll,
                scroll_filter=flt,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                ev = self._evidence_from_payload(p.payload or {}, score=None, point_id=str(p.id))
                ev["collection"] = coll
                out.append(ev)
        return out

    # ------------------------------------------------------------------
    # Diagnostics / lifecycle
    # ------------------------------------------------------------------
    def collection_info(self):
        """Per-collection info. Single dict when one collection is searched,
        else {"collections": [...]}."""
        infos = []
        for coll in self.collection_names:
            info = self.client.get_collection(coll)
            infos.append({
                "collection": coll,
                "points_count": info.points_count,
                "payload_indexes": list((info.payload_schema or {}).keys()),
                "dense_size": getattr(info.config.params.vectors, "size", None),
                "sparse_enabled": info.config.params.sparse_vectors is not None,
            })
        return infos[0] if len(infos) == 1 else {"collections": infos}

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass

    def __enter__(self) -> "QdrantHybridStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve_vectors(self, query: QueryType) -> tuple[list, qm.SparseVector]:
        if isinstance(query, str):
            if self.query_encoder is None:
                raise ValueError(
                    "query is a string but no query_encoder is configured; "
                    'pass a pre-encoded dict {"dense": [...], "sparse": SparseVector} '
                    "or set query_encoder (R2: BGEM3FlagModel)."
                )
            vectors = self.query_encoder(query)
        elif isinstance(query, dict):
            vectors = query
        else:
            raise TypeError(f"query must be str or dict, got {type(query).__name__}")

        dense = vectors["dense"]
        if hasattr(dense, "tolist"):          # numpy array -> list
            dense = dense.tolist()

        sparse = vectors["sparse"]
        if not isinstance(sparse, qm.SparseVector):
            sparse = qm.SparseVector(
                indices=[int(i) for i in sparse["indices"]],
                values=[float(v) for v in sparse["values"]],
            )
        return dense, sparse

    def _build_filter(self, filters: FilterSpec) -> Optional[qm.Filter]:
        if filters is None:
            return None
        if isinstance(filters, qm.Filter):
            return filters
        if not isinstance(filters, dict):
            raise TypeError(f"filters must be a dict or qdrant Filter, got {type(filters).__name__}")

        must = []
        for key, value in filters.items():
            if isinstance(value, (list, tuple, set)):
                must.append(
                    qm.FieldCondition(key=str(key), match=qm.MatchAny(any=[str(v) for v in value]))
                )
            else:
                must.append(qm.FieldCondition(key=str(key), match=qm.MatchValue(value=value)))
        return qm.Filter(must=must) if must else None

    def _format_scored(self, scored_points) -> list[dict]:
        return [
            self._evidence_from_payload(p.payload, score=p.score, point_id=str(p.id))
            for p in scored_points
        ]

    def _evidence_from_payload(
        self,
        payload: dict,
        score: Optional[float] = None,
        point_id: Optional[str] = None,
    ) -> dict:
        payload = payload or {}
        ev: dict[str, Any] = {k: payload.get(k) for k in EVIDENCE_FIELDS}
        ev["meta"] = ev.get("meta") or {}
        ev["score"] = score
        ev["point_id"] = point_id
        ev["domain"] = "judgment" if ev.get("doc_type") in ("judgment", "judgement") else "statute"
        return ev
