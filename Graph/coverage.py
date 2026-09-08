from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

TIER2_EDGE_TYPES = ("CITES", "CONCERNS", "DECIDED_BY", "INVOLVES")


def _pct(n: int, d: int) -> str:
    return f"{n} ({n / d:.0%})" if d else f"{n}"


def report(data: dict) -> None:
    nodes = data["nodes"]
    edges = data["edges"]
    by_label: dict[str, list] = {}
    for n in nodes:
        by_label.setdefault(n["label"], []).append(n)

    judges = by_label.get("Judgment", [])
    total = len(judges)

    print("=" * 70)
    print("EXTRACTED GRAPH — ENRICHMENT COVERAGE REPORT")
    print("=" * 70)
    print("\nNode counts:")
    for label in sorted(by_label):
        print(f"    {label:<12} {len(by_label[label])}")
    print("\nEdge counts:")
    for etype, n in sorted(Counter(e["type"] for e in edges).items()):
        print(f"    {etype:<14} {n}")

    # per-judgment flags
    flags: dict = {}
    for n in judges:
        flags[tuple(n["key"])] = {"cites": 0, "concerns": 0, "decided": 0, "involves": 0}
    for e in edges:
        if e["type"] not in TIER2_EDGE_TYPES:
            continue
        fk, tk = tuple(e["from"]["key"]), tuple(e["to"]["key"])
        if e["type"] == "CITES" and fk in flags:
            flags[fk]["cites"] += 1
        elif e["type"] == "CONCERNS" and fk in flags:
            flags[fk]["concerns"] += 1
        elif e["type"] == "INVOLVES" and fk in flags:
            flags[fk]["involves"] += 1
        elif e["type"] == "DECIDED_BY":
            if fk in flags:
                flags[fk]["decided"] += 1
            if tk in flags:  # direction can be Judgment->Court
                flags[tk]["decided"] += 1

    t2_props = [n for n in judges if n["props"].get("citation") or n["props"].get("date")]

    print(f"\nTier-2 enrichment coverage  (of {total} Judgment nodes):")
    print(f"    citation/date enriched : {_pct(len(t2_props), total)}")
    print(f"    has CITES (-> section) : {_pct(sum(1 for v in flags.values() if v['cites']), total)}")
    print(f"    has CONCERNS (-> offence): {_pct(sum(1 for v in flags.values() if v['concerns']), total)}")
    print(f"    has DECIDED_BY (-> court): {_pct(sum(1 for v in flags.values() if v['decided']), total)}")
    print(f"    has INVOLVES (-> party) : {_pct(sum(1 for v in flags.values() if v['involves']), total)}")

    print("\nSkips / misses:")
    if data.get("skipped"):
        for k, n in sorted(data["skipped"].items()):
            print(f"    {k:<38} {n}")
    else:
        print("    (none)")

    # judgment nodes with zero Tier-2 enrichment at all
    unenriched = [
        n["key"][0] for n in judges
        if not (n["props"].get("citation") or n["props"].get("date"))
        and all(flags[tuple(n["key"])][k] == 0 for k in ("cites", "concerns", "decided", "involves"))
    ]
    if unenriched:
        print(f"\nJudgments with NO enrichment at all: {len(unenriched)}")
        for u in unenriched[:15]:
            print(f"    {u}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Enrichment coverage report")
    ap.add_argument("--graph", default="Graph/graph_extracted.json")
    args = ap.parse_args(argv)
    data = json.load(open(args.graph, encoding="utf-8"))
    report(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
