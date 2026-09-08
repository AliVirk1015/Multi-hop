from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Graph import ontology as O

BATCH_SIZE = 500


# --------------------------------------------------------------------------- #
# Tiny .env loader (no dependency; never overrides real env vars)
# --------------------------------------------------------------------------- #
def _load_env(path: str = ".env") -> None:
    """Load .env into os.environ, OVERRIDING any existing OS env vars.

    .env is the source of truth for this project (so stale OS env vars from a
    previous session cannot silently override your .env credentials).
    """
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key:
            os.environ[key] = value


_load_env()


def get_config() -> dict:
    uri = os.getenv("NEO4J_URI", "").strip()
    return {
        "uri": uri,
        "user": os.getenv("NEO4J_USERNAME", "neo4j").strip(),
        "password": os.getenv("NEO4J_PASSWORD", ""),
        "database": os.getenv("NEO4J_DATABASE", "neo4j").strip(),
    }


def connect() -> Any:
    cfg = get_config()
    if not cfg["uri"] or not cfg["password"]:
        raise RuntimeError(
            "Neo4j not configured: set NEO4J_URI and NEO4J_PASSWORD (and NEO4J_USERNAME) in .env"
        )
    try:
        from neo4j import GraphDatabase
    except ImportError:
        raise RuntimeError("neo4j driver not installed — run: pip install neo4j")
    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    # verify connectivity eagerly so failures surface immediately
    try:
        driver.verify_connectivity()
    except Exception as exc:
        driver.close()
        raise RuntimeError(
            f"Neo4j connection/auth failed for {cfg['uri']} (user={cfg['user']}). "
            f"Underlying: {type(exc).__name__}. "
            f"Fix: in the AuraDB console (instance -> Settings -> Connection details) confirm "
            f"the username and get/reset the exact password, put it in NEO4J_PASSWORD in .env, "
            f"then re-run --check. Never share the password with anyone."
        ) from exc
    return driver


# --------------------------------------------------------------------------- #
# Schema (constraints + indexes) derived from the ontology
# --------------------------------------------------------------------------- #
def schema_statements() -> list[str]:
    stmts: list[str] = []
    for spec in O.NODES:
        label = spec.label
        # identity: our stable uuid5 id
        stmts.append(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.id IS UNIQUE")
        # semantic unique key from the ontology
        if len(spec.key) == 1:
            stmts.append(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.{spec.key[0]} IS UNIQUE")
        else:
            props = ", ".join(f"n.{k}" for k in spec.key)
            stmts.append(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE ({props}) IS UNIQUE")
    # Qdrant join key
    stmts.append("CREATE INDEX IF NOT EXISTS FOR (n:`Chunk`) ON (n.point_id)")
    return stmts


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _clean_props(props: dict) -> dict:
    return {k: v for k, v in props.items() if v is not None}


def load_nodes(driver: Any, nodes_path: Path, batch: int = BATCH_SIZE) -> int:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for line in open(nodes_path, encoding="utf-8"):
        row = json.loads(line)
        props = _clean_props(row.get("props", {}))
        by_label[row["label"]].append({"id": row["id"], **props})

    total = 0
    with driver.session(database=get_config()["database"]) as session:
        for label, rows in by_label.items():
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                query = (
                    f"UNWIND $rows AS r\n"
                    f"MERGE (n:`{label}` {{id: r.id}})\n"
                    f"ON CREATE SET n += r"
                )
                session.run(query, rows=chunk).consume()
                total += len(chunk)
    return total


def load_edges(driver: Any, edges_path: Path, batch: int = BATCH_SIZE) -> int:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for line in open(edges_path, encoding="utf-8"):
        e = json.loads(line)
        groups[(e["type"], e["from"]["label"], e["to"]["label"])].append(
            {"from_id": e["from"]["id"], "to_id": e["to"]["id"]}
        )

    total = 0
    with driver.session(database=get_config()["database"]) as session:
        for (rtype, flabel, tlabel), rows in groups.items():
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                query = (
                    f"UNWIND $rows AS r\n"
                    f"MATCH (a:`{flabel}` {{id: r.from_id}})\n"
                    f"MATCH (b:`{tlabel}` {{id: r.to_id}})\n"
                    f"MERGE (a)-[rel:`{rtype}`]->(b)"
                )
                session.run(query, rows=chunk).consume()
                total += len(chunk)
    return total


def reset(driver: Any) -> None:
    with driver.session(database=get_config()["database"]) as session:
        for spec in O.NODES:
            session.run(f"MATCH (n:`{spec.label}`) DETACH DELETE n").consume()


# --------------------------------------------------------------------------- #
# Verify / check
# --------------------------------------------------------------------------- #
def verify_counts(driver: Any) -> tuple[dict, dict]:
    with driver.session(database=get_config()["database"]) as session:
        node_counts = {}
        for spec in O.NODES:
            res = session.run(f"MATCH (n:`{spec.label}`) RETURN count(n) AS c")
            node_counts[spec.label] = res.single()["c"]
        edge_counts = {}
        for rel in O.RELS:
            res = session.run(f"MATCH ()-[r:`{rel.type}`]->() RETURN count(r) AS c")
            edge_counts[rel.type] = res.single()["c"]
    return node_counts, edge_counts


def _cmd_diag() -> int:
    """Print connection config diagnostics — password value NEVER printed."""
    cfg = get_config()
    pw = cfg["password"]
    print(f"NEO4J_URI      : {cfg['uri']}")
    print(f"NEO4J_USERNAME : {cfg['user']!r}")
    print(f"NEO4J_DATABASE : {cfg['database']!r}")
    print(f"password set   : {bool(pw)}  (length {len(pw)})")
    chars = ["=", "#", " ", "\t", '"', "'", ";", ","]
    print("password contains:", {c: (c in pw) for c in chars})
    print("password alnum   :", pw.isalnum() if pw else False)
    return 0


def _cmd_check(driver: Any) -> int:
    print(f"[neo4j] connected: {get_config()['uri']} (db={get_config()['database']})")
    with driver.session(database=get_config()["database"]) as session:
        res = session.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes RETURN name, labelsOrTypes")
        constraints = [r["name"] for r in res]
        print(f"[neo4j] existing constraints ({len(constraints)}): {constraints}")
    print("[neo4j] check OK — connection is valid (read-only)")
    return 0


def _cmd_verify(driver: Any) -> int:
    summary_path = Path("Graph/assembly_summary.json")
    summary = json.load(open(summary_path, encoding="utf-8")) if summary_path.exists() else None
    node_counts, edge_counts = verify_counts(driver)
    print("[neo4j] node counts in DB:")
    for label in sorted(node_counts):
        expected = summary["node_counts"].get(label) if summary else None
        mark = "" if expected is None or expected == node_counts[label] else f"  (summary says {expected})"
        print(f"    {label:<12} {node_counts[label]}{mark}")
    print("[neo4j] edge counts in DB:")
    for rtype in sorted(edge_counts):
        expected = summary["edge_counts"].get(rtype) if summary else None
        mark = "" if expected is None or expected == edge_counts[rtype] else f"  (summary says {expected})"
        print(f"    {rtype:<14} {edge_counts[rtype]}{mark}")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 5: load the assembled graph into Neo4j")
    ap.add_argument("--nodes", default="Graph/nodes.jsonl")
    ap.add_argument("--edges", default="Graph/edges.jsonl")
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only connectivity + schema check")
    mode.add_argument("--verify", action="store_true", help="compare Neo4j counts vs summary")
    mode.add_argument("--diag", action="store_true", help="print config diagnostics (password masked) — no connection")
    mode.add_argument("--load", action="store_true", help="create schema + load nodes + edges (default)")
    ap.add_argument("--reset", action="store_true", help="(with --load) detach-delete ontology labels first")
    args = ap.parse_args(argv)

    if args.diag:
        return _cmd_diag()

    driver = connect()
    try:
        if args.check:
            return _cmd_check(driver)
        if args.verify:
            return _cmd_verify(driver)
        # default: load
        if args.reset:
            print("[neo4j] --reset: detaching existing ontology nodes ...")
            reset(driver)
        print("[neo4j] creating constraints + indexes ...")
        with driver.session(database=get_config()["database"]) as session:
            for stmt in schema_statements():
                session.run(stmt).consume()
        print(f"[neo4j] loading nodes from {args.nodes} ...")
        n_nodes = load_nodes(driver, Path(args.nodes), batch=args.batch)
        print(f"[neo4j] loaded {n_nodes} nodes")
        print(f"[neo4j] loading edges from {args.edges} ...")
        n_edges = load_edges(driver, Path(args.edges), batch=args.batch)
        print(f"[neo4j] loaded {n_edges} edges")
        print("[neo4j] verifying ...")
        _cmd_verify(driver)
        print("[neo4j] load complete")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
