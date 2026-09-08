"""
Graph/profile.py — STEP 1: Data profiling (the real data schema).

Loads `chunks.json` (statutes) and `judgement_chunks.json` (judgments) and
produces two artifacts:

  * Graph/schema_report.json  — machine-readable schema + stats + quirks
  * Graph/SCHEMA.md           — human-readable "real data schema" document

This is the FOUNDATION for Step 2 (ontology design): it tells us which
fields exist, their types/coverage/cardinality, per-document structure, and
the data quirks we must design around (chunk_id sequential index vs the real
section_no, amendment metadata, judgment case-header parsing, etc.).

Design notes
------------
* Pure stdlib — no external dependencies (json, collections, statistics).
* Never infers section numbers from the chunk_id integer: the chunk_id number
  is a SEQUENTIAL index (offset from the real section number); the payload
  field `section_no` is authoritative.
* Tolerant of missing fields / non-dict records (skips them, reports counts).

Usage
-----
    .venv/Scripts/python.exe Graph\\profile.py \\
        --chunks chunks.json --judgements judgement_chunks.json \\
        --out-json Graph/schema_report.json --out-md Graph/SCHEMA.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _type_name(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def _is_empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _freeze(v: Any) -> str:
    """Hashable-ish repr for counting unique values of mixed types."""
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)


def _truncate(s: str, n: int = 140) -> str:
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _records(data: Any, path: Path) -> tuple[list[dict], str]:
    """Normalise input to a list of dict records; report the top-level shape."""
    if isinstance(data, list):
        shape = "list"
        recs = [r for r in data if isinstance(r, dict)]
        skipped = len(data) - len(recs)
    elif isinstance(data, dict):
        shape = "dict"
        recs = list(data.values()) if data else []
        skipped = 0
    else:
        raise ValueError(f"{path}: unsupported top-level JSON type {type(data).__name__}")
    if skipped:
        print(f"  [warn] {path}: skipped {skipped} non-dict records")
    return recs, shape


def _num(s: Any) -> int | None:
    """Parse an int from a string/int field; returns None if not numeric."""
    if s is None:
        return None
    try:
        return int(float(str(s)))
    except (ValueError, TypeError):
        return None


def _pctiles(vals: list[int]) -> dict[str, int]:
    if not vals:
        return {"min": None, "median": None, "p95": None, "max": None}
    s = sorted(vals)
    n = len(s)

    def q(p: float) -> int:
        k = max(0, min(n - 1, math.ceil(p * n) - 1))
        return s[k]

    return {"min": s[0], "median": q(0.5), "p95": q(0.95), "max": s[-1], "count": n}


# --------------------------------------------------------------------------- #
# Field-level schema analysis
# --------------------------------------------------------------------------- #
def _analyze_fields(records: list[dict]) -> dict:
    keys: set[str] = set()
    for r in records:
        keys.update(r.keys())
    fields: dict[str, dict] = {}
    for key in sorted(keys):
        vals = [r.get(key) for r in records]
        nonempty = [v for v in vals if not _is_empty(v)]
        types = Counter(_type_name(v) for v in nonempty)
        uniq_vals = set()
        for v in nonempty:
            uniq_vals.add(_freeze(v))
        fields[key] = {
            "primary_type": types.most_common(1)[0][0] if types else "null",
            "all_types": dict(types),
            "coverage": round(len(nonempty) / len(records), 4) if records else 0.0,
            "non_empty": len(nonempty),
            "nulls": len(records) - len(nonempty),
            "unique": len(uniq_vals),
            "example": _truncate(nonempty[0]) if nonempty else None,
        }
    return fields


def _categorical_counts(records: list[dict], keys: Iterable[str], max_values: int = 60) -> dict:
    out: dict[str, dict] = {}
    for key in keys:
        cnt: Counter = Counter()
        for r in records:
            v = r.get(key)
            if not _is_empty(v):
                cnt[v] += 1
        out[key] = {"total_values": len(cnt), "counts": dict(cnt.most_common(max_values))}
    return out


def _per_doc_counts(records: list[dict]) -> dict:
    cnt: Counter = Counter()
    for r in records:
        doc = r.get("doc")
        if not _is_empty(doc):
            cnt[doc] += 1
    return {"docs": len(cnt), "per_doc": dict(cnt.most_common())}


def _meta_analysis(records: list[dict]) -> dict:
    keys: Counter = Counter()
    samples: dict[str, Any] = {}
    for r in records:
        meta = r.get("meta")
        if isinstance(meta, dict):
            for k, v in meta.items():
                keys[k] += 1
                if k not in samples:
                    samples[k] = v
    return {"meta_keys": dict(keys), "samples": {k: _truncate(v) for k, v in samples.items()}}


def _tokens_stats(records: list[dict]) -> dict:
    vals = [_num(r.get("tokens")) for r in records]
    vals = [v for v in vals if v is not None]
    return _pctiles(vals)


# --------------------------------------------------------------------------- #
# Quirk detection
# --------------------------------------------------------------------------- #
def _statute_quirks(records: list[dict]) -> list[dict]:
    quirks: list[dict] = []

    # 1) chunk_id sequential index vs real section_no
    mismatches: list[dict] = []
    checked = 0
    for r in records:
        if r.get("level") != "section":
            continue
        cid = r.get("chunk_id") or ""
        seq = _num(cid.rsplit(":", 1)[-1] if ":" in cid else None)
        sec = _num(r.get("section_no"))
        if seq is not None and sec is not None:
            checked += 1
            if seq != sec:
                mismatches.append(
                    {"chunk_id": cid, "chunk_seq": seq, "section_no": sec, "doc": r.get("doc")}
                )
    if checked:
        quirks.append(
            {
                "id": "chunk_id_vs_section_no",
                "checked_sections": checked,
                "mismatch_count": len(mismatches),
                "mismatch_ratio": round(len(mismatches) / checked, 4),
                "note": (
                    "chunk_id number is a SEQUENTIAL index, not the real section number. "
                    "Always key on payload section_no, never the chunk_id integer."
                ),
                "sample": mismatches[:5],
            }
        )

    # 2) duplicate section_no within one document (legit parent/child splits vs true dups)
    dup_sections: dict[tuple, list[str]] = defaultdict(list)
    for r in records:
        if r.get("level") == "section":
            key = (r.get("doc"), r.get("section_no"))
            if key[1] is not None:
                dup_sections[key].append(r.get("chunk_id"))
    multi = {k: v for k, v in dup_sections.items() if len(v) > 1}
    quirks.append(
        {
            "id": "duplicate_section_no_in_doc",
            "docs_with_dup_section_numbers": len(multi),
            "note": (
                "Multiple chunks can share (doc, section_no) — legitimately, when a large "
                "section is split into parent + paragraph children. Not a real duplicate."
            ),
            "sample": [{"doc": k[0], "section_no": k[1], "chunk_ids": v[:4]} for k, v in list(multi.items())[:5]],
        }
    )

    # 3) amendment docs carry amends metadata
    amendments = [r for r in records if str(r.get("doc_type", "")).lower() == "amendment"]
    with_meta = [r for r in amendments if isinstance(r.get("meta"), dict) and r["meta"].get("amends_section")]
    quirks.append(
        {
            "id": "amendment_metadata",
            "amendment_docs": len(amendments),
            "with_amends_section_meta": len(with_meta),
            "note": (
                "doc_type='amendment' + meta.amends_section is a ready-made AMENDED_BY "
                "relationship (Tier 0 — reuse it, don't re-extract)."
            ),
        }
    )

    # 4) broken parent links
    broken = _broken_parents(records)
    quirks.append(
        {
            "id": "broken_parent_links",
            "count": len(broken),
            "note": (
                "parent_id values that do not resolve to a chunk_id in the same file. "
                "These edges cannot be built unless the parent lives in the other file "
                "(rare) or the link is genuinely dangling."
            ),
            "sample": broken[:5],
        }
    )
    return quirks


def _judgement_quirks(records: list[dict]) -> list[dict]:
    quirks: list[dict] = []

    # section_no in judgments is a per-document paragraph counter, NOT a statute section
    sec_samples = [
        {"chunk_id": r.get("chunk_id"), "section_no": r.get("section_no")}
        for r in records[:6]
    ]
    quirks.append(
        {
            "id": "judgment_section_no_is_para_counter",
            "note": (
                "In judgment chunks, section_no is a paragraph/order counter, unrelated to "
                "statute sections. Do NOT create Section nodes from judgment chunks."
            ),
            "sample": sec_samples,
        }
    )

    # meta.context case-header parse: "[Case: X vs Y | Court | Case No] [Para N]"
    parsed = 0
    for r in records:
        ctx = (r.get("meta") or {}).get("context") if isinstance(r.get("meta"), dict) else None
        if isinstance(ctx, str) and "Case:" in ctx and "|" in ctx:
            parsed += 1
    quirks.append(
        {
            "id": "case_header_context",
            "chunks_with_case_header": parsed,
            "note": (
                "meta.context embeds the case header as '[Case: parties | Court | Case No]'. "
                "This is the structured seed for Tier 2 (LLM) Judgment/Court/Party extraction."
            ),
        }
    )

    broken = _broken_parents(records)
    quirks.append(
        {
            "id": "broken_parent_links",
            "count": len(broken),
            "note": "parent_id values not resolving to a chunk_id in this file.",
            "sample": broken[:5],
        }
    )
    return quirks


def _broken_parents(records: list[dict]) -> list[dict]:
    ids = {r.get("chunk_id") for r in records}
    broken = []
    for r in records:
        p = r.get("parent_id")
        if p is not None and p not in ids:
            broken.append({"chunk_id": r.get("chunk_id"), "parent_id": p, "doc": r.get("doc")})
    return broken


# --------------------------------------------------------------------------- #
# Per-file analysis
# --------------------------------------------------------------------------- #
def _analyze_file(path: Path, kind: str) -> dict:
    print(f"[profile] loading {path} ...")
    data = _load_json(path)
    recs, shape = _records(data, path)
    report: dict[str, Any] = {
        "path": str(path),
        "kind": kind,
        "record_count": len(recs),
        "top_level_type": shape,
        "fields": _analyze_fields(recs),
        "categorical": _categorical_counts(
            recs,
            [k for k in ("doc_type", "level", "part", "chapter") if k in recs[0] or any(k in r for r in recs)],
        ),
        "per_doc": _per_doc_counts(recs),
        "structure": {
            "parent_id": _categorical_counts(recs, ["parent_id"])["parent_id"]["total_values"],
            "with_parent": sum(1 for r in recs if not _is_empty(r.get("parent_id"))),
            "is_parent_true": sum(1 for r in recs if r.get("is_parent") is True),
            "is_parent_false": sum(1 for r in recs if r.get("is_parent") is False),
        },
        "meta": _meta_analysis(recs),
        "tokens": _tokens_stats(recs),
        "quirks": (_statute_quirks(recs) if kind == "statute" else _judgement_quirks(recs)),
    }
    return report


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _render_file_md(rep: dict) -> str:
    name = Path(rep["path"]).name
    kind = rep["kind"].title()
    out: list[str] = [f"## {name} ({kind})", ""]
    out.append(f"- **Records:** {rep['record_count']}  |  top-level: `{rep['top_level_type']}`")
    out.append(f"- **Documents:** {rep['per_doc']['docs']}")
    out.append("")

    # field schema
    out.append("### Field schema")
    out.append("")
    rows = []
    for f, s in rep["fields"].items():
        rows.append(
            [
                f"`{f}`",
                s["primary_type"],
                f"{s['coverage']:.0%}",
                str(s["unique"]),
                _truncate(s["example"], 60) if s["example"] else "—",
            ]
        )
    out.append(_md_table(["Field", "Type", "Coverage", "Unique", "Example"], rows))
    out.append("")

    # categorical
    out.append("### Categorical distributions")
    out.append("")
    for key, info in rep["categorical"].items():
        counts = info["counts"]
        top = " · ".join(f"{k}: {v}" for k, v in list(counts.items())[:12])
        out.append(f"- **`{key}`** ({info['total_values']} distinct): {top}")
    out.append("")

    # per-doc
    out.append("### Per-document chunk counts")
    out.append("")
    rows = [[doc, str(n)] for doc, n in rep["per_doc"]["per_doc"].items()]
    out.append(_md_table(["Document", "Chunks"], rows))
    out.append("")

    # structure
    out.append("### Structure (parent / child)")
    out.append("")
    st = rep["structure"]
    out.append(
        f"- distinct `parent_id` values: {st['parent_id']}  ·  chunks with a parent: {st['with_parent']}  ·  "
        f"`is_parent` true: {st['is_parent_true']}  ·  false: {st['is_parent_false']}"
    )
    out.append("")

    # tokens
    t = rep["tokens"]
    out.append("### Token stats (per chunk)")
    out.append("")
    if t.get("count"):
        out.append(
            f"- min **{t['min']}** · median **{t['median']}** · p95 **{t['p95']}** · max **{t['max']}** (n={t['count']})"
        )
    out.append("")

    # meta
    out.append("### `meta` analysis")
    out.append("")
    for k, v in rep["meta"]["meta_keys"].items():
        out.append(f"- `{k}` (present in {v}) — sample: {_truncate(rep['meta']['samples'].get(k, ''), 80)}")
    out.append("")

    # quirks
    out.append("### Detected quirks")
    out.append("")
    for q in rep["quirks"]:
        out.append(f"- **{q['id']}**: {q['note']}")
        for k, v in q.items():
            if k in ("id", "note"):
                continue
            if isinstance(v, list) and v:
                out.append(f"  - {k}: `{_truncate(v, 160)}`")
            elif not isinstance(v, list):
                out.append(f"  - {k}: `{v}`")
    out.append("")
    return "\n".join(out)


def _render_md(report: dict) -> str:
    out = [
        "# Knowledge Graph — Step 1: Real Data Schema Report",
        "",
        f"_Generated: {report['generated_at']}_",
        "",
        "This is the **data schema** (raw material). Step 2 derives the **ontology** "
        "(node/relationship types) from it — the two are different artifacts.",
        "",
    ]
    for name, rep in report["inputs"].items():
        out.append(_render_file_md(rep))
    out.append(
        "## Implications for Step 2 (ontology design)\n"
        "\n"
        "- Structure edges (`CONTAINS`/`PART_OF`/`PARENT_OF`) are fully derivable from "
        "`doc`/`part`/`chapter`/`section_no`/`parent_id` — Tier 0, zero LLM.\n"
        "- `AMENDED_BY` comes free from `doc_type=amendment` + `meta.amends_section`.\n"
        "- Judgment `Court`/`Party`/`case_no` are **not** structured fields — extract from "
        "`meta.context` + preamble text (Tier 2).\n"
        "- Never key statute nodes on the `chunk_id` integer — use `section_no`.\n"
    )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 1: data profiling for Graph RAG")
    ap.add_argument("--chunks", default="chunks.json")
    ap.add_argument("--judgements", default="judgement_chunks.json")
    ap.add_argument("--out-json", default="Graph/schema_report.json")
    ap.add_argument("--out-md", default="Graph/SCHEMA.md")
    args = ap.parse_args(argv)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {},
    }
    for label, p, kind in (
        ("chunks", args.chunks, "statute"),
        ("judgements", args.judgements, "judgement"),
    ):
        report["inputs"][label] = _analyze_file(Path(p), kind)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_render_md(report), encoding="utf-8")

    print(f"[profile] wrote {out_json}")
    print(f"[profile] wrote {out_md}")
    # quick console summary
    for label, rep in report["inputs"].items():
        nq = sum(1 for q in rep["quirks"] if (isinstance(q.get("mismatch_count"), int) and q["mismatch_count"]) or (q.get("id") == "broken_parent_links" and q.get("count")))
        print(f"[profile] {label}: {rep['record_count']} records, {rep['per_doc']['docs']} docs, {nq} flagged quirks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
