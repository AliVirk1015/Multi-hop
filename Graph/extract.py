from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Graph import ontology as O


ACT_REFERENCE_ALIASES: dict[str, str] = {
    "prevention of electronic crimes act": "Prevention of Electronic Crimes Act 2016",
    "electronic crimes act": "Prevention of Electronic Crimes Act 2016",
    "code of criminal procedure": "Code of Criminal Procedure 1898",
    "qanun-e-shahadat": "Qanun-e-Shahadat Order 1984",
    "criminal procedure": "Code of Criminal Procedure 1898",
    "penal code": "Pakistan Penal Code 1860",
    "money laundering act": "Anti-Money Laundering Act 2010",
    "money laundering": "Anti-Money Laundering Act 2010",
    "telecommunication": "Pakistan Telecommunication (Re-organization) Act 1996",
    "payment systems": "Payment Systems and Electronic Fund Transfers Act 2007",
    "electronic transactions ordinance": "Electronic Transactions Ordinance 2002",
    "electronic transaction ordinance": "Electronic Transactions Ordinance 2002",
    "electronic transactions": "Electronic Transactions Ordinance 2002",
    "investigation for fair trial": "Investigation for Fair Trial Act 2013",
    "fair trial act": "Investigation for Fair Trial Act 2013",
    "consumer protection": "Consumer Protection Regulations 2009",
    "peca": "Prevention of Electronic Crimes Act 2016",
    "crpc": "Code of Criminal Procedure 1898",
    "ppc": "Pakistan Penal Code 1860",
    "cert": "CERT Rules 2023",
}


def resolve_act_reference(text: str) -> Optional[str]:
    """Return the canonical Act name if `text` mentions a known act, else None."""
    low = text.lower()
    for key in sorted(ACT_REFERENCE_ALIASES, key=len, reverse=True):
        if key in low:
            return ACT_REFERENCE_ALIASES[key]
    return None


# --------------------------------------------------------------------------- #
# Tier 1 regexes
# --------------------------------------------------------------------------- #
_REF_TOKEN_RE = re.compile(r"\b(?:section|sections|sec|s)\.?\s+(\d+[A-Za-z]?)", re.IGNORECASE)
_REF_ACT_TAIL_RE = re.compile(r"\bof\s+(?:the\s+)?([A-Z][\w&'(),.\- ]{2,60})", re.IGNORECASE)
_MULTI_CONT_RE = re.compile(r"[,\s]*(?:and\s*|,|&)\s*(\d+[A-Za-z]?)", re.IGNORECASE)


def _expand_section_numbers(text: str, first: str, end: int) -> list[str]:
    nums = [first]
    tail = text[end: end + 80]
    for m in _MULTI_CONT_RE.finditer(tail):
        nums.append(m.group(1))
        if len(nums) >= 5:
            break
    return nums


def extract_cross_refs(text: str, default_act: str) -> list[tuple[str, str]]:
    """Extract (act, section_no) cross-references from section text.

    Handles "section N of <Act>", bare "section N" (-> same act), and
    "sections 20, 21 and 24". Never resolves the target act to something
    unknown — returns only canonical names (or the default act).
    """
    refs: list[tuple[str, str]] = []
    for m in _REF_TOKEN_RE.finditer(text):
        first, end = m.group(1), m.end()
        act = default_act
        tail = text[end: end + 90]
        om = _REF_ACT_TAIL_RE.search(tail)
        if om:
            cand = resolve_act_reference(om.group(1))
            if cand:
                act = cand
        for sec in _expand_section_numbers(text, first, end):
            refs.append((act, sec))
    return refs


# --------------------------------------------------------------------------- #
# Graph store
# --------------------------------------------------------------------------- #
def _nk(label: str, key: tuple) -> tuple[str, tuple]:
    return (label, key)


class Graph:
    """In-memory extracted graph: deduped nodes (label,key) + edges."""

    def __init__(self) -> None:
        self.nodes: dict[tuple[str, tuple], dict] = {}
        self.edges: dict[tuple, dict] = {}
        self.skipped: Counter = Counter()  # provenance of skips/misses

    # ---- nodes ------------------------------------------------------------- #
    def add_node_raw(self, label: str, props: dict, tier: str, source_file: str,
                     chunk_ids: Optional[list] = None) -> Optional[tuple]:
        spec = O.NODE_SPECS_BY_LABEL.get(label)
        if spec is None:
            raise KeyError(f"unknown node label: {label}")
        if any(props.get(k) is None or props.get(k) == "" for k in spec.key):
            return None
        key = tuple(props[k] for k in spec.key)
        nk = _nk(label, key)
        if nk in self.nodes:
            for k, v in props.items():
                if self.nodes[nk]["props"].get(k) is None:
                    self.nodes[nk]["props"][k] = v
            if chunk_ids:
                self.nodes[nk]["chunk_ids"].update(chunk_ids)
            return nk
        self.nodes[nk] = {
            "label": label, "key": key, "props": dict(props), "tier": tier,
            "source_file": source_file, "chunk_ids": set(chunk_ids or []),
        }
        return nk

    def add_node_from_record(self, label: str, record: dict, tier: str = "0",
                             source_file: str = "", collection: Optional[str] = None):
        node = O.build_node(label, record)
        if node is None:
            return None
        props = dict(node["props"])
        if collection is not None:
            props["collection"] = collection
        return self.add_node_raw(label, props, tier, source_file, [record.get("chunk_id")])

    # ---- edges ------------------------------------------------------------- #
    def add_edge(self, type_: str, from_nk: Optional[tuple], to_nk: Optional[tuple],
                 tier: str, source: str, chunk_id: Optional[str] = None) -> Optional[tuple]:
        if from_nk is None or to_nk is None:
            return None
        ekey = (type_, from_nk, to_nk, source)
        if ekey in self.edges:
            if chunk_id:
                self.edges[ekey]["chunk_ids"].add(chunk_id)
            return ekey
        self.edges[ekey] = {
            "type": type_, "from": from_nk, "to": to_nk,
            "tier": tier, "source": source,
            "chunk_ids": {chunk_id} if chunk_id else set(),
        }
        return ekey

    # ---- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        nodes = [
            {
                "label": n["label"], "key": list(n["key"]), "props": n["props"],
                "tier": n["tier"], "source_file": n["source_file"],
                "chunk_ids": sorted(n["chunk_ids"]),
            }
            for n in self.nodes.values()
        ]
        edges = [
            {
                "type": e["type"],
                "from": {"label": e["from"][0], "key": list(e["from"][1])},
                "to": {"label": e["to"][0], "key": list(e["to"][1])},
                "tier": e["tier"], "source": e["source"],
                "chunk_ids": sorted(e["chunk_ids"]),
            }
            for e in self.edges.values()
        ]
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "node_counts": dict(Counter(n["label"] for n in nodes)),
            "edge_counts": dict(Counter(e["type"] for e in edges)),
            "skipped": dict(self.skipped),
            "nodes": nodes,
            "edges": edges,
        }


# --------------------------------------------------------------------------- #
# Tier 0 — statutes (structure)
# --------------------------------------------------------------------------- #
def extract_tier0_statutes(g: Graph, recs: list[dict]) -> None:
    for r in recs:
        doc = r.get("doc")
        act = O.canonical_act_name(doc)
        if not act:
            g.skipped["no_act_name"] += 1
            continue
        doc_type = str(r.get("doc_type", "")).lower()
        level = r.get("level")
        section_no = r.get("section_no")
        chunk_id = r.get("chunk_id")

        act_key = g.add_node_from_record("Act", r, tier="0", source_file="chunks")
        chunk_key = g.add_node_from_record("Chunk", r, tier="0", source_file="chunks", collection="statutes")

        is_statute_section = (
            level == "section"
            and doc_type in ("statute", "rules", "amendment", "order")
            and section_no not in (None, "")
        )
        if is_statute_section:
            sec_key = g.add_node_from_record("Section", r, tier="0", source_file="chunks")
            if sec_key:
                g.add_edge("CONTAINS", act_key, sec_key, "0", "hierarchy:act->section", chunk_id)
                g.add_edge("BELONGS_TO", chunk_key, sec_key, "0", "chunk.section_no", chunk_id)

                part, chapter = r.get("part"), r.get("chapter")
                if part:
                    part_key = g.add_node_raw("Part", {"act": act, "name": str(part)}, "0", "chunks", [chunk_id])
                    g.add_edge("CONTAINS", act_key, part_key, "0", "hierarchy:act->part", chunk_id)
                    g.add_edge("CONTAINS", part_key, sec_key, "0", "hierarchy:part->section", chunk_id)
                if chapter:
                    ch_key = g.add_node_raw("Chapter", {"act": act, "name": str(chapter)}, "0", "chunks", [chunk_id])
                    g.add_edge("CONTAINS", act_key, ch_key, "0", "hierarchy:act->chapter", chunk_id)
                    g.add_edge("CONTAINS", ch_key, sec_key, "0", "hierarchy:chapter->section", chunk_id)
        elif section_no not in (None, ""):
            # non-section chunk -> its logical Section (if that Section node exists)
            sec_key = _nk("Section", (act, str(section_no)))
            if sec_key in g.nodes:
                g.add_edge("BELONGS_TO", chunk_key, sec_key, "0", "chunk.section_no", chunk_id)
            else:
                g.skipped["belongs_to_no_section_node"] += 1

        # PARENT_OF (parent chunk -> this chunk)
        parent = r.get("parent_id")
        if parent:
            parent_key = _nk("Chunk", (parent,))
            if parent_key in g.nodes:
                g.add_edge("PARENT_OF", parent_key, chunk_key, "0", "parent_id", chunk_id)
            else:
                g.skipped["parent_not_in_graph"] += 1

        # AMENDED_BY (hybrid Tier 0 + Tier 1)
        if doc_type == "amendment":
            _extract_amended_by(g, r, act, chunk_id)


def _extract_amended_by(g: Graph, r: dict, amending_act: str, chunk_id: Optional[str]) -> None:
    meta = r.get("meta")
    amends_section = meta.get("amends_section") if isinstance(meta, dict) else None
    if not amends_section:
        m = re.search(r"Amendment of section\s+([\dA-Za-z\-]+)", r.get("text", ""), re.IGNORECASE)
        if m:
            amends_section = m.group(1)
    if not amends_section:
        g.skipped["amendment_no_section"] += 1
        return
    amended_act = resolve_act_reference(r.get("text", ""))
    if not amended_act:
        g.skipped["amendment_no_act"] += 1
        return
    sec_key = _nk("Section", (amended_act, str(amends_section)))
    if sec_key not in g.nodes:
        g.skipped["amendment_section_not_in_graph"] += 1
        return
    amending_act_key = _nk("Act", (amending_act,))
    if amending_act_key not in g.nodes:
        amending_act_key = g.add_node_raw(
            "Act", {"name": amending_act, "doc": r.get("doc"), "doc_type": "amendment"},
            "0", "chunks", [chunk_id],
        )
    g.add_edge("AMENDED_BY", sec_key, amending_act_key, "0", "amendment meta/text", chunk_id)


# --------------------------------------------------------------------------- #
# Tier 1 — regex cross-references
# --------------------------------------------------------------------------- #
def extract_tier1_references(g: Graph, recs: list[dict]) -> None:
    for r in recs:
        if r.get("level") != "section":
            continue
        act = O.canonical_act_name(r.get("doc"))
        sec = r.get("section_no")
        if not act or sec in (None, ""):
            continue
        from_key = _nk("Section", (act, str(sec)))
        if from_key not in g.nodes:
            continue
        for ref_act, ref_sec in extract_cross_refs(r.get("text", ""), act):
            to_key = _nk("Section", (ref_act, ref_sec))
            if to_key in g.nodes:
                g.add_edge("REFERENCES", from_key, to_key, "1", "regex cross-ref", r.get("chunk_id"))
            elif ref_act != act:
                act_key = _nk("Act", (ref_act,))
                if act_key in g.nodes:
                    g.add_edge("REFERENCES", from_key, act_key, "1", "regex cross-ref(act)", r.get("chunk_id"))
                else:
                    g.skipped["ref_act_not_in_graph"] += 1
            else:
                g.skipped["ref_section_not_in_graph"] += 1


# --------------------------------------------------------------------------- #
# Tier 0 — judgments (nodes + Court/Party seeds from the case header)
# --------------------------------------------------------------------------- #
_VS_SPLIT_RE = re.compile(r"\s+(?:vs|v)\.?\s+|\s+versus\s+", re.IGNORECASE)
_TRAILING_RE = re.compile(r",?\s*(?:etc\.?|&?\s*others)\s*$", re.IGNORECASE)


def _split_parties(parties_str: str) -> list[str]:
    if not parties_str:
        return []
    out = []
    for side in _VS_SPLIT_RE.split(parties_str):
        side = _TRAILING_RE.sub("", side).strip(" ,;:") if side else ""
        if side:
            out.append(side)
    return out


def extract_tier0_judgments(g: Graph, judg_recs: list[dict]) -> None:
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in judg_recs:
        by_doc[r.get("doc")].append(r)

    for doc, recs in by_doc.items():
        chunk_ids = [r.get("chunk_id") for r in recs]
        # judgment node from preamble (fallback: first record)
        preamble = next((r for r in recs if r.get("level") == "preamble"), recs[0])
        jkey = g.add_node_from_record("Judgment", preamble, tier="0", source_file="judgements")
        if jkey is None:
            jkey = g.add_node_raw(
                "Judgment", {"case_no": Path(doc).stem, "doc": doc},
                "0", "judgements", chunk_ids,
            )
        # enrich outcome from meta.disposition wherever it appears
        for r in recs:
            meta = r.get("meta")
            if isinstance(meta, dict) and meta.get("disposition"):
                g.nodes[jkey]["props"]["outcome"] = meta["disposition"]

        # per-chunk: Chunk node + PARENT_OF
        for r in recs:
            ckey = g.add_node_from_record("Chunk", r, tier="0", source_file="judgements", collection="judgements")
            parent = r.get("parent_id")
            if parent:
                parent_key = _nk("Chunk", (parent,))
                if parent_key in g.nodes:
                    g.add_edge("PARENT_OF", parent_key, ckey, "0", "parent_id", r.get("chunk_id"))
                else:
                    g.skipped["judgment_parent_not_in_graph"] += 1

        # Court seed + DECIDED_BY (free, from header)
        header = O._parse_case_header(preamble)
        court = header.get("court")
        if court:
            court_key = g.add_node_raw("Court", {"name": court}, "0", "judgements", chunk_ids)
            g.add_edge("DECIDED_BY", jkey, court_key, "0", "case header", chunk_ids[0] if chunk_ids else None)
        # Party seed + INVOLVES (free, from header)
        for party in _split_parties(header.get("parties", "")):
            pkey = g.add_node_raw("Party", {"name": party}, "0", "judgements", chunk_ids)
            g.add_edge("INVOLVES", jkey, pkey, "0", "case header", chunk_ids[0] if chunk_ids else None)


# --------------------------------------------------------------------------- #
# Tier 2 — optional LLM enrichment for judgments
# --------------------------------------------------------------------------- #
TIER2_SYSTEM = (
    "You extract a legal knowledge graph from a Pakistani court judgment. "
    'Return ONLY valid JSON matching this schema: '
    '{"parties": ["..."], "court": "...", "case_number": "...", "citation": "...", '
    '"date": "...", "offences": ["..."], '
    '"cited_sections": [{"act": "...", "section": "..."}], '
    '"cited_cases": ["..."]}. '
    "Rules:\n"
    "- offences must be SHORT offence names (e.g. \"fraud\", \"hacking\", "
    "\"criminal breach of trust\") — do NOT include \"Section N of Act X\" there; "
    "put section references only in cited_sections.\n"
    "- cited_sections: act = full or common act name (e.g. \"Pakistan Penal Code\", "
    "\"The Prevention of Electronic Crimes Act, 2016\"), section = the exact section number.\n"
    "- Use exact section numbers and act names as they appear in the text. "
    "- If a field is absent, use an empty string or empty list.\n"
    "- Output raw JSON only — no markdown fences, no prose."
)


def _llm_available() -> bool:
    try:
        from llm_client import llm_reachable
        return bool(llm_reachable())
    except Exception:
        return False


def extract_tier2_judgments(g: Graph, judg_recs: list[dict], max_docs: int = 0) -> None:
    try:
        from llm_client import llm_chat_json
    except ImportError:
        print("[tier2] llm_client not importable; skipping LLM enrichment")
        return

    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in judg_recs:
        by_doc[r.get("doc")].append(r)
    docs = list(by_doc.items())
    if max_docs:
        docs = docs[:max_docs]

    for doc, recs in docs:
        preamble = next((r for r in recs if r.get("level") == "preamble"), recs[0])
        jkey = _nk("Judgment", (O._parse_case_header(preamble).get("case_no") or Path(doc).stem,))
        if jkey not in g.nodes:
            jkey = g.add_node_raw("Judgment", {"case_no": Path(doc).stem, "doc": doc},
                                  "2", "judgements", [r.get("chunk_id") for r in recs])
        text = "\n\n".join(r.get("text", "") for r in recs[:6])[:8000]
        # gpt-oss-20b is a reasoning model: needs headroom for reasoning tokens
        # before the visible JSON, else the output truncates and json.loads fails.
        try:
            res = llm_chat_json(TIER2_SYSTEM, text, max_tokens=2200)
        except Exception as exc:  # pragma: no cover - depends on live gateway
            print(f"[tier2] LLM call failed for {doc}: {exc}")
            continue
        if not res:
            g.skipped["tier2_empty_response"] += 1
            continue

        # enrich judgment props
        for k in ("citation", "date", "court", "parties"):
            if res.get(k):
                g.nodes[jkey]["props"][k] = str(res[k])

        # DECIDED_BY
        if res.get("court"):
            court_key = g.add_node_raw("Court", {"name": str(res["court"])}, "2", "judgements")
            g.add_edge("DECIDED_BY", jkey, court_key, "2", "llm", None)
        # INVOLVES
        for party in res.get("parties", []) or []:
            pkey = g.add_node_raw("Party", {"name": str(party)}, "2", "judgements")
            g.add_edge("INVOLVES", jkey, pkey, "2", "llm", None)
        # CONCERNS (skip entries that are section citations, not offence names)
        for off in res.get("offences", []) or []:
            off = str(off).strip()
            if not off or re.search(r"\bsection|sec\.?\s*\d", off, re.IGNORECASE):
                g.skipped["tier2_offence_is_section_ref"] += 1
                continue
            okey = g.add_node_raw("Offence", {"name": off}, "2", "judgements")
            g.add_edge("CONCERNS", jkey, okey, "2", "llm", None)
        # CITES (resolve to sections that exist in the graph)
        for cite in res.get("cited_sections", []) or []:
            act_name = resolve_act_reference(str(cite.get("act", "")))
            sec = str(cite.get("section", "")).strip()
            if not act_name or not sec:
                g.skipped["tier2_cite_unresolved_act"] += 1
                continue
            to_key = _nk("Section", (act_name, sec))
            if to_key in g.nodes:
                g.add_edge("CITES", jkey, to_key, "2", "llm", None)
            else:
                g.skipped["tier2_cite_section_not_in_graph"] += 1
    print("[tier2] LLM judgment enrichment complete")



# Main

def _load_records(path: Path) -> list[dict]:
    return [r for r in json.load(open(path, encoding="utf-8")) if isinstance(r, dict)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 3: entity/relationship extraction")
    ap.add_argument("--chunks", default="chunks.json")
    ap.add_argument("--judgements", default="judgement_chunks.json")
    ap.add_argument("--out", default="Graph/graph_extracted.json")
    ap.add_argument("--tier2", action="store_true", help="run LLM judgment enrichment (needs reachable gateway)")
    ap.add_argument("--tier2-max-docs", type=int, default=0, help="limit Tier 2 to N judgment docs (0 = all)")
    args = ap.parse_args(argv)

    g = Graph()
    chunk_recs = _load_records(Path(args.chunks))
    judg_recs = _load_records(Path(args.judgements))

    print("[extract] Tier 0 statutes ...")
    extract_tier0_statutes(g, chunk_recs)
    print("[extract] Tier 1 cross-references ...")
    extract_tier1_references(g, chunk_recs)
    print("[extract] Tier 0 judgments ...")
    extract_tier0_judgments(g, judg_recs)

    if args.tier2:
        if _llm_available():
            print("[extract] Tier 2 LLM judgment enrichment ...")
            extract_tier2_judgments(g, judg_recs, max_docs=args.tier2_max_docs)
        else:
            print("[extract] --tier2 requested but LLM gateway unreachable; skipping (run later with "
                  "LLM_BASE_URL/LLM_MODEL configured, or re-run without --tier2 for the offline graph)")

    data = g.to_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[extract] wrote {out}")
    print("[extract] node counts:")
    for label, n in sorted(data["node_counts"].items()):
        print(f"    {label:<10} {n}")
    print("[extract] edge counts:")
    for etype, n in sorted(data["edge_counts"].items()):
        print(f"    {etype:<12} {n}")
    print("[extract] skips/misses:")
    for k, n in sorted(data["skipped"].items()):
        print(f"    {k:<34} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
