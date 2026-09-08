
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Prop:
    """A single property on a node.

    source is one of:
      * a record field name            -> record.get(name)
      * "canonical_act_name(doc)"      -> ACT_ALIASES-resolved act name
      * "const:<literal>"              -> a constant value
    """
    name: str
    dtype: str = "str"
    source: str = ""
    required: bool = False


@dataclass(frozen=True)
class NodeSpec:
    label: str
    key: tuple[str, ...]            # unique-key property names (composite ok)
    props: tuple[Prop, ...]
    tier: str                       # "0" deterministic | "1" regex | "2" LLM
    description: str
    filter_rule: str = ""           # human-readable filter (which records apply)
    source_file: str = "chunks"     # "chunks" | "judgements" | "both"


@dataclass(frozen=True)
class RelSpec:
    type: str
    description: str
    from_labels: tuple[str, ...]
    to_labels: tuple[str, ...]
    tier: str
    cardinality: str                # "1:1" | "1:N" | "M:N"
    source: str                     # how the edge is derived
    filter_rule: str = ""


# --------------------------------------------------------------------------- #
# Act-name canonicalisation (dedupe) — REVIEW: confirm official titles
# --------------------------------------------------------------------------- #
ACT_ALIASES: dict[str, str] = {
    "THE CODE OF CRIMINAL PROCEDURE, 1898.pdf": "Code of Criminal Procedure 1898",
    "Pakistan Penal Code.pdf": "Pakistan Penal Code 1860",
    "THE QANUN-E-SHAHADAT, 1984.pdf": "Qanun-e-Shahadat Order 1984",
    "Anti Money Laundring 2010.pdf": "Anti-Money Laundering Act 2010",  # typo in source
    "PECA 2016.pdf": "Prevention of Electronic Crimes Act 2016",
    "PAKISTAN TELECOMMUNICATION (Reorg) 1996.pdf": "Pakistan Telecommunication (Re-organization) Act 1996",
    "PAYMENT SYSTEMS AND ELECTRONIC FUND.pdf": "Payment Systems and Electronic Fund Transfers Act 2007",
    "PECA ru2018-ocr (1).pdf": "Prevention of Electronic Crimes Investigation Rules 2018",
    "Electronic Transaction Ordinance 2002.pdf": "Electronic Transactions Ordinance 2002",
    "Investigation of Fair Trail Act 2013.pdf": "Investigation for Fair Trial Act 2013",  # "Trail" typo in source
    "Peca-Act-2025 AMD.pdf": "Prevention of Electronic Crimes (Amendment) Act 2025",
    # --- REVIEW: official titles below are my best derivation; verify before trusting ---
    "CERT 2023.pdf": "CERT Rules 2023",
    "RemovalBlockingofUnlawfulOnlineContentRules2021 (1)UPDATED.pdf": "Removal and Blocking of Unlawful Online Content Rules 2021",
    "Pakistan Consumer Protection Regulation 2009 UPDATED.pdf": "Consumer Protection Regulations 2009",
    "ncciaorder (1)UPDATED.pdf": "NCIIA Order",
}


def canonical_act_name(doc: Optional[str]) -> Optional[str]:
    """Map a document filename to the canonical Act name (for Act nodes / dedupe)."""
    if not doc:
        return None
    return ACT_ALIASES.get(doc) or Path(doc).stem.strip()


# --------------------------------------------------------------------------- #
# THE ONTOLOGY — node types
# --------------------------------------------------------------------------- #
NODES: tuple[NodeSpec, ...] = (
    NodeSpec(
        label="Act",
        key=("name",),
        props=(
            Prop("name", "str", "canonical_act_name(doc)", required=True),
            Prop("doc", "str", "doc"),
            Prop("doc_type", "str", "doc_type"),
        ),
        tier="0",
        description="A statute / rules / amendment / order instrument (one per source PDF).",
        filter_rule="records from chunks.json only (doc_type in statute|rules|amendment|order)",
        source_file="chunks",
    ),
    NodeSpec(
        label="Part",
        key=("act", "name"),
        props=(
            Prop("act", "str", "canonical_act_name(doc)", required=True),
            Prop("name", "str", "part", required=True),
        ),
        tier="0",
        description="A Part within an Act (Roman numeral, e.g. I..IX). Only present when part != null.",
        filter_rule="part field not null (41% coverage in chunks.json)",
        source_file="chunks",
    ),
    NodeSpec(
        label="Chapter",
        key=("act", "name"),
        props=(
            Prop("act", "str", "canonical_act_name(doc)", required=True),
            Prop("name", "str", "chapter", required=True),
        ),
        tier="0",
        description="A Chapter within an Act (56 distinct chapter ids).",
        filter_rule="chapter field not null (87% coverage in chunks.json)",
        source_file="chunks",
    ),
    NodeSpec(
        label="Section",
        key=("act", "section_no"),
        props=(
            Prop("act", "str", "canonical_act_name(doc)", required=True),
            Prop("section_no", "str", "section_no", required=True),
            Prop("title", "str", "section_title"),
            Prop("text", "str", "text"),
            Prop("tokens", "int", "tokens"),
            Prop("chunk_id", "str", "chunk_id"),  # primary parent chunk for Qdrant join
        ),
        tier="0",
        description="A provision/section of an act. Key (act, section_no) verified UNIQUE.",
        filter_rule="level == 'section' AND source is chunks.json (NEVER judgement chunks — "
                    "their section_no is a paragraph counter)",
        source_file="chunks",
    ),
    NodeSpec(
        label="Chunk",
        key=("chunk_id",),
        props=(
            Prop("chunk_id", "str", "chunk_id", required=True),
            Prop("doc", "str", "doc"),
            Prop("doc_type", "str", "doc_type"),
            Prop("level", "str", "level"),
            Prop("section_no", "str", "section_no"),
            Prop("parent_id", "str", "parent_id"),
            Prop("is_parent", "bool", "is_parent"),
            Prop("tokens", "int", "tokens"),
            Prop("text", "str", "text"),
            Prop("collection", "str", "const:statutes"),  # overridden for judgements
        ),
        tier="0",
        description="Every source chunk. Provides the chunk_id join back to Qdrant vectors.",
        filter_rule="all records from both files; chunk_id is unique per file and formats differ",
        source_file="both",
    ),
    NodeSpec(
        label="Judgment",
        key=("case_no",),
        props=(
            Prop("case_no", "str", "case_no(meta.context)", required=True),  # Tier 0 header parse
            Prop("doc", "str", "doc"),
            Prop("citation", "str", "citation"),              # Tier 2 enrichment
            Prop("court", "str", "court(meta.context)"),      # Tier 0 when header present; Tier 2 enriches
            Prop("parties", "str", "parties(meta.context)"),  # Tier 0 when header present
            Prop("date", "str", "date"),                      # Tier 2 enrichment
            Prop("outcome", "str", "outcome(meta)"),          # Tier 0 free via meta.disposition
        ),
        tier="0",
        description="One node per judgment document (82 docs). case_no/court/parties/outcome are "
                    "free via the case-header + disposition metadata (Tier 0).",
        filter_rule="records from judgement_chunks.json only; identify by doc basename",
        source_file="judgements",
    ),
    NodeSpec(
        label="Court",
        key=("name",),
        props=(Prop("name", "str", "name", required=True),),
        tier="2",
        description="Court that decided a case (e.g. Lahore High Court). Extracted by LLM from preamble.",
        filter_rule="none — LLM-derived, referenced by Judgment.DECIDED_BY",
        source_file="judgements",
    ),
    NodeSpec(
        label="Party",
        key=("name",),
        props=(
            Prop("name", "str", "name", required=True),
            Prop("role", "str", "role"),
        ),
        tier="2",
        description="A litigant (petitioner / respondent / complainant). LLM-extracted, canonicalized.",
        filter_rule="none — LLM-derived, referenced by Judgment.INVOLVES",
        source_file="judgements",
    ),
    NodeSpec(
        label="Offence",
        key=("name",),
        props=(Prop("name", "str", "name", required=True),),
        tier="2",
        description="An offence mentioned/charged in a judgment (e.g. fraud, hacking). LLM-extracted.",
        filter_rule="none — LLM-derived, referenced by Judgment.CONCERNS",
        source_file="judgements",
    ),
)

# --------------------------------------------------------------------------- #
# THE ONTOLOGY — relationship types
# --------------------------------------------------------------------------- #
RELS: tuple[RelSpec, ...] = (
    RelSpec(
        type="CONTAINS",
        description="Structural containment in the statute hierarchy (Act→Part/Chapter→Section).",
        from_labels=("Act", "Part", "Chapter"),
        to_labels=("Part", "Chapter", "Section"),
        tier="0",
        cardinality="1:N",
        source="part/chapter fields + level==section on chunks.json records",
    ),
    RelSpec(
        type="PARENT_OF",
        description="Small-to-big chunk chain (parent chunk → child chunk) via parent_id.",
        from_labels=("Chunk",),
        to_labels=("Chunk",),
        tier="0",
        cardinality="1:N",
        source="parent_id field (96 parents / 274 children in chunks.json; 61/163 in judgements)",
    ),
    RelSpec(
        type="BELONGS_TO",
        description="A chunk belongs to its logical Section (uses section_no, Tier 0).",
        from_labels=("Chunk",),
        to_labels=("Section",),
        tier="0",
        cardinality="N:1",
        source="chunk.section_no on statute chunks (never judgment chunks)",
    ),
    RelSpec(
        type="AMENDED_BY",
        description="A section is amended by an amending Act.",
        from_labels=("Section",),
        to_labels=("Act",),
        tier="0",
        cardinality="N:1",
        source="meta.amends_section on doc_type=amendment chunks (4/32 present); "
               "fallback Tier 1 regex on the remaining 28",
        filter_rule="only for doc_type=amendment docs (e.g. Peca-Act-2025 AMD)",
    ),
    RelSpec(
        type="REFERENCES",
        description="A section explicitly references another section/act ('section N of <Act>', 'this Act').",
        from_labels=("Section",),
        to_labels=("Section", "Act"),
        tier="1",
        cardinality="M:N",
        source="regex cross-reference extraction on section text (the old Graph-Lite logic, rebuilt here)",
    ),
    RelSpec(
        type="CITES",
        description="A judgment cites / relies on a statutory section.",
        from_labels=("Judgment",),
        to_labels=("Section",),
        tier="2",
        cardinality="M:N",
        source="LLM extraction from judgment paragraphs (structured JSON)",
    ),
    RelSpec(
        type="DECIDED_BY",
        description="A judgment was decided by a court.",
        from_labels=("Judgment",),
        to_labels=("Court",),
        tier="2",
        cardinality="N:1",
        source="LLM extraction from preamble / meta.context",
    ),
    RelSpec(
        type="INVOLVES",
        description="A judgment involves a party (petitioner/respondent/complainant).",
        from_labels=("Judgment",),
        to_labels=("Party",),
        tier="2",
        cardinality="N:M",
        source="LLM extraction from preamble / meta.context",
    ),
    RelSpec(
        type="CONCERNS",
        description="A judgment concerns an offence.",
        from_labels=("Judgment",),
        to_labels=("Offence",),
        tier="2",
        cardinality="N:M",
        source="LLM extraction from judgment text",
    ),
)

# --------------------------------------------------------------------------- #
# Deterministic case-header parser (Tier 0) — the chunker emits
# meta.context as "[Case: parties | Court | Case No] [Para N]". The format is
# NOT consistent across docs (2-part, 3-part, sometimes court-first), so we
# classify each "|"-separated segment by pattern rather than by position.
# --------------------------------------------------------------------------- #
_CASE_HEADER_RE = re.compile(r"\[Case:\s*(.*?)\]\s*(?:\[Para\s+[^\]]*\])?")

_CASE_COURT_RE = re.compile(
    r"(?:high court|district court|session court|sessions court|family court|"
    r"supreme court|federal shariat court|civil court|trial court|banking court|"
    r"special court|judicial magistrate|additional session|competent court)",
    re.IGNORECASE,
)
_CASE_NUMBER_RE = re.compile(
    r"(?:no\.?\s*[:.]?\s*\d|\b(?:19|20)\d\d\b|misc\.?|crl\.?|bail\s+application|"
    r"writ\s+petition|petition|criminal\s+bail|revision|appeal|complaint|fir\b|"
    r"cr\.\s*misc|criminal\s+revision)",
    re.IGNORECASE,
)


def _parse_case_header(record: dict) -> dict:
    """Parse meta.context into {parties, court, case_no}; empty dict if absent.

    Segment classifier: a segment matching a court name (and not a case-number
    pattern) is the court; a segment matching a case-number pattern is the case
    number; everything else is treated as party text.
    """
    meta = record.get("meta")
    ctx = meta.get("context") if isinstance(meta, dict) else None
    if not isinstance(ctx, str):
        return {}
    m = _CASE_HEADER_RE.search(ctx)
    if not m:
        return {}
    parts = [p.strip() for p in m.group(1).split("|") if p.strip()]
    out: dict[str, str] = {}
    if len(parts) == 1:
        out["case_no"] = parts[0]
        return out

    court: Optional[str] = None
    case_no: Optional[str] = None
    party_parts: list[str] = []
    for seg in parts:
        is_court = _CASE_COURT_RE.search(seg) and not _CASE_NUMBER_RE.search(seg)
        is_case = _CASE_NUMBER_RE.search(seg)
        if is_court and court is None:
            court = seg
        elif is_case and case_no is None:
            case_no = seg
        else:
            party_parts.append(seg)
    if court:
        out["court"] = court
    if case_no:
        out["case_no"] = case_no
    if party_parts:
        out["parties"] = " | ".join(party_parts)
    return out


# --------------------------------------------------------------------------- #
# Resolver: turn a spec + record into concrete nodes/keys
# --------------------------------------------------------------------------- #
def _resolve_expr(expr: str, record: dict, act_name: Optional[str]) -> Any:
    if expr.startswith("const:"):
        return expr[len("const:"):]
    if expr == "canonical_act_name(doc)":
        return act_name
    if expr == "case_no(meta.context)":
        h = _parse_case_header(record)
        return h.get("case_no") or Path(record.get("doc") or "").stem or None
    if expr == "court(meta.context)":
        return _parse_case_header(record).get("court")
    if expr == "parties(meta.context)":
        return _parse_case_header(record).get("parties")
    if expr == "outcome(meta)":
        meta = record.get("meta")
        return meta.get("disposition") if isinstance(meta, dict) else None
    return record.get(expr)


def build_node(label: str, record: dict) -> Optional[dict]:
    """Build a single node dict {label, props} from a source record, or None if it
    does not satisfy the spec (missing required key property or filter)."""
    spec = NODE_SPECS_BY_LABEL.get(label)
    if spec is None:
        raise KeyError(f"unknown node label: {label}")
    act_name = canonical_act_name(record.get("doc"))
    props: dict[str, Any] = {}
    for p in spec.props:
        v = _resolve_expr(p.source, record, act_name)
        if v is not None and v != "" and v != [] and v != {}:
            props[p.name] = v
    # require every key property to be present
    if any(props.get(k) is None for k in spec.key):
        return None
    return {"label": label, "props": props}


def node_key(label: str, record: dict) -> Optional[dict]:
    """Return just the unique-key dict {prop: value} for a record, or None."""
    node = build_node(label, record)
    if node is None:
        return None
    spec = NODE_SPECS_BY_LABEL[label]
    return {k: node["props"][k] for k in spec.key}


NODE_SPECS_BY_LABEL: dict[str, NodeSpec] = {n.label: n for n in NODES}
REL_SPECS_BY_TYPE: dict[str, RelSpec] = {r.type: r for r in RELS}


# --------------------------------------------------------------------------- #
# Serialisation + validation
# --------------------------------------------------------------------------- #
def ontology_to_dict() -> dict:
    def prop_dict(p: Prop) -> dict:
        return {"name": p.name, "dtype": p.dtype, "source": p.source, "required": p.required}

    def node_dict(n: NodeSpec) -> dict:
        return {
            "label": n.label,
            "key": list(n.key),
            "props": [prop_dict(p) for p in n.props],
            "tier": n.tier,
            "description": n.description,
            "filter_rule": n.filter_rule,
            "source_file": n.source_file,
        }

    def rel_dict(r: RelSpec) -> dict:
        return {
            "type": r.type,
            "description": r.description,
            "from_labels": list(r.from_labels),
            "to_labels": list(r.to_labels),
            "tier": r.tier,
            "cardinality": r.cardinality,
            "source": r.source,
            "filter_rule": r.filter_rule,
        }

    return {
        "nodes": [node_dict(n) for n in NODES],
        "relationships": [rel_dict(r) for r in RELS],
        "act_aliases": ACT_ALIASES,
        "notes": {
            "section_key": "(act, section_no) — verified unique in chunks.json",
            "never_key_on_chunk_id_integer": (
                "chunk_id number is a sequential index, not the real section; "
                "95.6% of section chunks mismatch section_no"
            ),
            "judgment_metadata_free": (
                "case_no/court/parties/outcome are Tier 0 via case-header + meta.disposition; "
                "citation/date and enrichment are Tier 2 LLM"
            ),
            "amendment_gap": "only 4/32 amendment chunks carry meta.amends_section; 28 need Tier 1 regex",
            "judgment_noise_filter": "23 mislabeled judgement chunks (doc_type/level) must be filtered from Section building",
        },
    }


def _load_records(path: Path) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    return [r for r in data if isinstance(r, dict)]


def check_against_data(chunks_path: Path, judgements_path: Path) -> None:
    """Validate every node spec's key is resolvable against the real chunk data."""
    print("=" * 78)
    print("ONTOLOGY VALIDATION against real data (key resolvability)")
    print("=" * 78)
    chunk_recs = _load_records(chunks_path)
    judg_recs = _load_records(judgements_path)
    print(f"  chunks.json: {len(chunk_recs)} records   judgement_chunks.json: {len(judg_recs)} records")
    print("-" * 78)
    print(f"{'Node':<12}{'Key':<32}{'Resolvable':>10}{'Coverage':>10}")
    print("-" * 78)
    for spec in NODES:
        recs = chunk_recs if spec.source_file in ("chunks", "both") else judg_recs
        if spec.source_file == "both":
            recs = chunk_recs + judg_recs
        # approximate the filter: for 'chunks' only, skip judgement noise naturally by file split
        n_key = 0
        for r in recs:
            if node_key(spec.label, r) is not None:
                n_key += 1
        cov = n_key / len(recs) if recs else 0.0
        key = "+".join(spec.key)
        print(f"{spec.label:<12}{key:<32}{n_key:>10}{cov:>9.0%}")
    print("-" * 78)
    print("Note: coverage is over the spec.source_file's records, before LLM enrichment;")
    print("Court/Party/Offence show 0% because Tier 2 LLM fields aren't in the JSON yet.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 2: ontology design")
    ap.add_argument("--chunks", default="chunks.json")
    ap.add_argument("--judgements", default="judgement_chunks.json")
    ap.add_argument("--out-json", default="Graph/ontology.json")
    args = ap.parse_args(argv)

    check_against_data(Path(args.chunks), Path(args.judgements))

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ontology_to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ontology] wrote {out}")
    print(f"[ontology] {len(NODES)} node types, {len(RELS)} relationship types")
    return 0


if __name__ == "__main__":
    sys.exit(main())
