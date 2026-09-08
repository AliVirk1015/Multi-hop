from __future__ import annotations
import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Graph import ontology as O

DNS = uuid.NAMESPACE_DNS


def node_id(label: str, key: tuple) -> str:
    """Stable uuid5 id. Chunk ids match Qdrant point ids (uuid5(DNS, chunk_id))."""
    if label == "Chunk":
        return str(uuid.uuid5(DNS, key[0]))  # key[0] == chunk_id
    return str(uuid.uuid5(DNS, "graph:" + label + ":" + ":".join(str(k) for k in key)))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def assemble(data: dict, drop_section_text: bool = False, allow_dangling: bool = False) -> dict:
    nodes_in = data.get("nodes", [])
    edges_in = data.get("edges", [])

    nodes: list[dict] = []
    key_to_id: dict[tuple[str, tuple], str] = {}
    id_counts: Counter = Counter()
    duplicate_nodes: list[dict] = []

    for n in nodes_in:
        label = n["label"]
        key = tuple(n["key"])
        spec = O.NODE_SPECS_BY_LABEL.get(label)
        if spec is None:
            raise ValueError(f"unknown node label in graph: {label}")
        props = dict(n["props"])
        if drop_section_text and label == "Section":
            props.pop("text", None)
        nid = node_id(label, key)
        id_counts[nid] += 1
        if (label, key) in key_to_id:
            duplicate_nodes.append(n)
            continue  # keep first occurrence
        key_to_id[(label, key)] = nid
        node = {
            "id": nid,
            "label": label,
            "key_props": {k: props.get(k) for k in spec.key},
            "props": props,
            "tier": n.get("tier", ""),
            "source_file": n.get("source_file", ""),
            "chunk_ids": n.get("chunk_ids", []),
        }
        if label == "Chunk":
            node["point_id"] = nid  # == Qdrant point id
            node["props"]["point_id"] = nid
        nodes.append(node)

    # ---- resolve edges ----------------------------------------------------- #
    dangling: list[dict] = []
    edges: list[dict] = []
    for e in edges_in:
        fk = (e["from"]["label"], tuple(e["from"]["key"]))
        tk = (e["to"]["label"], tuple(e["to"]["key"]))
        fid = key_to_id.get(fk)
        tid = key_to_id.get(tk)
        if fid is None or tid is None:
            dangling.append(
                {
                    "type": e["type"],
                    "from": e["from"],
                    "to": e["to"],
                    "unresolved_from": fid is None,
                    "unresolved_to": tid is None,
                }
            )
            continue
        edges.append(
            {
                "type": e["type"],
                "from": {"label": e["from"]["label"], "id": fid, "key": e["from"]["key"]},
                "to": {"label": e["to"]["label"], "id": tid, "key": e["to"]["key"]},
                "tier": e.get("tier", ""),
                "source": e.get("source", ""),
                "chunk_ids": e.get("chunk_ids", []),
            }
        )

    validation = {
        "nodes_in": len(nodes_in),
        "nodes_out": len(nodes),
        "duplicate_node_keys_skipped": len(duplicate_nodes),
        "duplicate_node_ids": {str(i): c for i, c in id_counts.items() if c > 1},
        "edges_in": len(edges_in),
        "edges_out": len(edges),
        "dangling_edges": len(dangling),
        "dangling_samples": dangling[:10],
    }

    if validation["dangling_edges"] and not allow_dangling:
        raise RuntimeError(
            f"assembly aborted: {validation['dangling_edges']} dangling edge(s) "
            f"(endpoints not in the node set). Inspect validation['dangling_samples'] "
            f"or pass --allow-dangling to force."
        )

    return {"nodes": nodes, "edges": edges, "validation": validation}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 4: graph assembly -> Neo4j-ready JSONL")
    ap.add_argument("--graph", default="Graph/graph_extracted.json")
    ap.add_argument("--nodes", default="Graph/nodes.jsonl")
    ap.add_argument("--edges", default="Graph/edges.jsonl")
    ap.add_argument("--summary", default="Graph/assembly_summary.json")
    ap.add_argument("--drop-section-text", action="store_true",
                    help="drop Section.text (the Chunk node already holds it)")
    ap.add_argument("--allow-dangling", action="store_true",
                    help="write edges even if some endpoints do not resolve")
    args = ap.parse_args(argv)

    data = json.load(open(args.graph, encoding="utf-8"))
    print(f"[build] loaded {args.graph}  (generated_at={data.get('generated_at')})")

    result = assemble(data, drop_section_text=args.drop_section_text,
                      allow_dangling=args.allow_dangling)
    nodes, edges, validation = result["nodes"], result["edges"], result["validation"]

    _write_jsonl(Path(args.nodes), nodes)
    _write_jsonl(Path(args.edges), edges)
    summary = {
        "generated_at": data.get("generated_at"),
        "validation": validation,
        "node_counts": dict(Counter(n["label"] for n in nodes)),
        "edge_counts": dict(Counter(e["type"] for e in edges)),
        "id_scheme": {
            "Chunk": "uuid5(DNS, chunk_id) == Qdrant point_id",
            "other": "uuid5(DNS, graph:<Label>:<key...>)",
        },
    }
    Path(args.summary).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[build] wrote {args.nodes}  ({len(nodes)} nodes)")
    print(f"[build] wrote {args.edges}  ({len(edges)} edges)")
    print(f"[build] wrote {args.summary}")
    print()
    print("node counts:")
    for label, n in sorted(summary["node_counts"].items()):
        print(f"    {label:<12} {n}")
    print("edge counts:")
    for etype, n in sorted(summary["edge_counts"].items()):
        print(f"    {etype:<14} {n}")
    print()
    print("validation:")
    for k, v in validation.items():
        if k == "dangling_samples":
            print(f"    {k}: {len(v)} shown")
        else:
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
