from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Graph.neo4j_store import get_config  # ensures .env is loaded

# canonical act-name lookup used by record shaping (import from ontology)
from Graph import ontology as O  # noqa: E402


class KnowledgeGraphClient:
    """Thin Cypher client over the loaded legal knowledge graph."""

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, database: Optional[str] = None,
                 driver: Optional[Any] = None) -> None:
        cfg = get_config()
        self.uri = uri or cfg["uri"]
        self.user = user or cfg["user"]
        self.password = password or cfg["password"]
        self.database = database or cfg["database"]
        if driver is not None:
            self.driver = driver
        else:
            try:
                from neo4j import GraphDatabase
            except ImportError:
                raise RuntimeError("neo4j driver not installed — run: pip install neo4j")
            if not self.uri or not self.password:
                raise RuntimeError("Neo4j not configured — set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD in .env")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._opened = driver is None

    def close(self) -> None:
        if self._opened:
            self.driver.close()

    def __enter__(self) -> "KnowledgeGraphClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # low-level
    # ------------------------------------------------------------------ #
    def _run(self, query: str, params: Optional[dict] = None) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            res = session.run(query, params or {})
            return [dict(r) for r in res]

    def _run_single(self, query: str, params: Optional[dict] = None) -> Optional[dict]:
        with self.driver.session(database=self.database) as session:
            rec = session.run(query, params or {}).single()
            return dict(rec) if rec else None

    # ------------------------------------------------------------------ #
    # 1) Qdrant -> graph join: chunk_id -> its Section(s)
    # ------------------------------------------------------------------ #
    def sections_for_chunks(self, chunk_ids: list[str], limit: int = 25) -> list[dict]:
        if not chunk_ids:
            return []
        rows = self._run(
            "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids "
            "MATCH (c)-[:BELONGS_TO]->(s:Section) "
            "RETURN DISTINCT s.id AS id, s.act AS act, s.section_no AS section_no, "
            "s.title AS title, s.doc AS doc, s.chunk_id AS chunk_id, c.chunk_id AS via_chunk "
            "LIMIT $limit",
            {"chunk_ids": list(chunk_ids), "limit": limit},
        )
        return [self._section_record(r, via_chunk=r.get("via_chunk")) for r in rows]

    # ------------------------------------------------------------------ #
    # 2) exact Section lookup
    # ------------------------------------------------------------------ #
    def get_section(self, act: str, section_no: str) -> Optional[dict]:
        row = self._run_single(
            "MATCH (s:Section {act: $act, section_no: $section_no}) "
            "RETURN s.id AS id, s.act AS act, s.section_no AS section_no, "
            "s.title AS title, s.doc AS doc, s.chunk_id AS chunk_id LIMIT 1",
            {"act": act, "section_no": str(section_no)},
        )
        return self._section_record(row) if row else None

    # ------------------------------------------------------------------ #
    # 3) BFS expansion over REFERENCES (both directions), bounded
    # ------------------------------------------------------------------ #
    def expand_sections(self, seed_ids: list[str], hops: int = 1, max_nodes: int = 40) -> list[dict]:
        if not seed_ids:
            return []
        hops = max(1, int(hops))
        visited: set[str] = set(seed_ids)
        frontier = list(seed_ids)
        found: list[dict] = []
        for _ in range(hops):
            if not frontier or len(found) >= max_nodes:
                break
            rows = self._run(
                "MATCH (s:Section) WHERE s.id IN $frontier "
                "MATCH (s)-[:REFERENCES]-(n:Section) "
                "RETURN DISTINCT n.id AS id, n.act AS act, n.section_no AS section_no, "
                "n.title AS title, n.doc AS doc, n.chunk_id AS chunk_id",
                {"frontier": frontier},
            )
            next_frontier: list[str] = []
            for r in rows:
                if r["id"] in visited or len(found) >= max_nodes:
                    continue
                visited.add(r["id"])
                found.append(self._section_record(r))
                next_frontier.append(r["id"])
            frontier = next_frontier
        return found

    # ------------------------------------------------------------------ #
    # 4) judgments citing a Section (case law on a provision)
    # ------------------------------------------------------------------ #
    def judgments_citing_section(self, section_id: str, limit: int = 20) -> list[dict]:
        rows = self._run(
            "MATCH (s:Section {id: $id})<-[:CITES]-(j:Judgment) "
            "RETURN DISTINCT j.id AS id, j.case_no AS case_no, j.doc AS doc, "
            "j.court AS court, j.outcome AS outcome, j.date AS date "
            "LIMIT $limit",
            {"id": section_id, "limit": limit},
        )
        out = []
        for r in rows:
            out.append({
                "kg_id": r["id"],
                "doc": r.get("doc"),
                "doc_type": "judgement",
                "case_no": r.get("case_no"),
                "court": r.get("court"),
                "outcome": r.get("outcome"),
                "date": r.get("date"),
                "source": "kg:CITES",
            })
        return out

    # ------------------------------------------------------------------ #
    # 5) orchestrator: full graph expansion from retrieved chunk_ids
    # ------------------------------------------------------------------ #
    def expand_evidence(self, chunk_ids: list[str], hops: int = 1,
                        max_nodes: int = 40) -> dict:
        """Qdrant chunks -> their Sections -> REFERENCES neighbors + citing judgments.

        Returns {"sections": [...], "judgments": [...]} where each record is
        shaped like RAG evidence plus kg_id + source provenance.
        """
        seeds = self.sections_for_chunks(chunk_ids, limit=max_nodes)
        seed_ids = [s["kg_id"] for s in seeds]
        sections = self.expand_sections(seed_ids, hops=hops, max_nodes=max_nodes)
        judgments: list[dict] = []
        # case law only for the directly-hit sections (1 hop from chunks)
        for sid in seed_ids:
            judgments.extend(self.judgments_citing_section(sid))
        # dedupe judgments by kg_id
        seen: set = set()
        uniq_judg = []
        for j in judgments:
            if j["kg_id"] in seen:
                continue
            seen.add(j["kg_id"])
            uniq_judg.append(j)
        return {"seed_sections": seeds, "sections": sections, "judgments": uniq_judg}

    # ------------------------------------------------------------------ #
    # shaping helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _section_record(r: dict, via_chunk: Optional[str] = None) -> dict:
        return {
            "kg_id": r.get("id"),
            "chunk_id": r.get("chunk_id") or via_chunk,  # Section's own chunk or the joining chunk
            "doc": r.get("doc"),
            "act": r.get("act"),
            "doc_type": "statute",
            "level": "section",
            "section_no": r.get("section_no"),
            "section_title": r.get("title"),
            "text": None,  # text lives in Qdrant; graph keeps titles only
            "source": "kg",
        }

    # ------------------------------------------------------------------ #
    # sanity / counts
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        rows = self._run(
            "MATCH (n) WITH labels(n)[0] AS label RETURN label, count(*) AS c ORDER BY c DESC"
        )
        return {r["label"]: r["c"] for r in rows}

    def canonical_acts(self) -> list[str]:
        return sorted({s["act"] for s in self._run("MATCH (s:Section) RETURN DISTINCT s.act AS act")})


# convenience alias matching ontology import usage
def connect(*args: Any, **kwargs: Any) -> KnowledgeGraphClient:
    return KnowledgeGraphClient(*args, **kwargs)
