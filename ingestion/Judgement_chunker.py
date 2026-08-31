"""
chunking.py — PDF extraction + structure-aware hierarchical chunking
=====================================================================

Target corpora: Pakistan legal PDFs (statutes, gazette rules, amendment
acts, court orders) AND judicial judgements (high-court / Supreme Court
judgment & order sheets, reported-case headnote digests).

Design
------
One pipeline, multiple *document profiles*, selected by a detection ladder,
with a graceful paragraph fallback so no document ever fails:

    statute    : Part -> Chapter -> Section -> (sub-section/clause)
    rules      : Rule N / numbered item          (S.R.O. notifications)
    amendment  : "Amendment of section N of Act X"
    judgement  : masthead/headnote -> meta + headnotes + numbered paras
    order      : single-document chunk           (e.g. IHC order sheet)
    paragraph  : layout/paragraph fallback       (OCR-garbled PDFs)

The `judgement` profile is checked BEFORE the statute ladder, because the
numbered paragraphs of a judgment ("12. In view of the above ...") would
otherwise trigger the numbered-section heuristic of the statute profile.
It is content-based (masthead + parties + "decided on" scoring), so it also
wins over the filename-based `order` check — an IHC order *sheet* deserves
paragraph-level chunks, not one giant chunk.

Judgement parsing produces, in document order:

    preamble : court, case no., parties, judges, counsel   (meta["keep"])
    headnote : "(a) Act Name---S.Section X --- holding"    (meta["keep"],
                                                              meta["is_headnote"])
    para     : one numbered paragraph per block; interludes between
               numbering restarts and the closing signature block are kept
               attached to their neighbouring paragraph

Every profile feeds a single normalisation pass ("parent-child / small-to-big"):

    1. skip the CONTENTS table (it duplicates every heading)          [statutes]
    2. merge tiny heading stubs (< min_chunk_tokens) into a neighbour
       (blocks flagged meta["keep"] are never merged away)            [judgement]
    3. split giant sections (> section_split_threshold_tokens) into leaf
       children with paragraph-level overlap
    4. emit the full section as a *parent* chunk (small-to-big expansion),
       children carry `parent_id`

Judgement chunks additionally get a *contextual prefix* baked into their text:

    [Case: Sheraz Khan vs The State | Lahore High Court |
     Crl.Misc.No.44216-B/2021] [Para 5]

so every leaf embeds its provenance (bge-m3 has plenty of headroom for this).
Document-level metadata (court, case numbers, parties, judges, decided date,
statutes cited from headnotes, disposition) is extracted once per document and
stored on `ExtractedDocument.judgement_meta`; per-chunk disposition hits are
flagged as meta["disposition"].

Dependencies: pypdf (extraction). Everything else is stdlib.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader


@dataclass
class ChunkingConfig:
    """Tunables, sized from a real audit of the corpus.

    Measured section-body token distributions (median / p95 / max):
      PPC   85 / 568 / 3348 ;  CrPC 19 / 541 / 24782 ;  PECA 7 / -- / 6321
    So the fixed-window fallacy is avoided: sections are the atomic unit,
    tiny stubs are merged, giants are split. bge-m3 (8192-token inputs)
    comfortably fits both the 600-token leaves and the contextual prefixes.
    """

    min_chunk_tokens: int = 20          # below this a section is a "stub" -> merge
    section_split_threshold_tokens: int = 500   # above this -> split into leaves
    max_chunk_tokens: int = 600         # leaf chunk ceiling
    parent_max_tokens: int = 2000       # cap for the small-to-big parent chunk
    min_structure_coverage: float = 0.30  # if parsed blocks cover <30% of text -> fallback


@dataclass
class ExtractedDocument:
    name: str
    path: str
    title: str
    pages: List[str]
    text: str
    doc_type: str          # statute | rules | amendment | judgement | order | paragraph
    needs_ocr: bool
    judgement_meta: Optional[dict] = None   # populated when doc_type == "judgement"


@dataclass
class Block:
    """A structural unit found by the parser (one section/rule/para/heading)."""

    level: str
    num: Optional[str]
    title: str
    body: str
    part: Optional[str]
    chapter: Optional[str]
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    doc_name: str
    doc_type: str
    level: str
    text: str
    section_no: Optional[str] = None
    section_title: Optional[str] = None
    part: Optional[str] = None
    chapter: Optional[str] = None
    meta: dict = field(default_factory=dict)
    chunk_id: str = ""
    parent_id: Optional[str] = None
    is_parent: bool = False
    tokens: int = 0

    @property
    def heading(self) -> str:
        parts = [p for p in (self.section_no, self.section_title) if p]
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc": self.doc_name,
            "doc_type": self.doc_type,
            "level": self.level,
            "section_no": self.section_no,
            "section_title": self.section_title,
            "part": self.part,
            "chapter": self.chapter,
            "parent_id": self.parent_id,
            "is_parent": self.is_parent,
            "tokens": self.tokens,
            "text": self.text,
            "meta": self.meta,
        }


# ---------------------------------------------------------------------------
# Heading patterns (line-anchored)
# ---------------------------------------------------------------------------

# NOTE: PART/CHAPTER are also used with `.search()` on the full text for
# doc-type detection, hence re.MULTILINE so ^ / $ match per line.
_PART_RE = re.compile(r"^\s*PART\s+([IVXLC\d]+)\s*$", re.IGNORECASE | re.MULTILINE)
_CHAPTER_RE = re.compile(
    r"^\s*CHAPTER\s+([IVXLC\d]+(?:-?[A-Z])?)\s*$", re.IGNORECASE | re.MULTILINE
)
# "12. Short title and commencement."  — num may carry a letter: 7A, 22A, 2A.
# Also tolerate amendment footnote markers that prefix the number:
#   "1[2. Definitions.— In this Act ..."  (common in Pakistani statutes)
_SECTION_RE = re.compile(
    r"^\s*(?:\d+\s*\[)?\s*(?P<num>\d{1,3}[A-Z]?)\s*\.\s+(?P<title>[A-Z][^\n]{2,150})\s*$"
)
# loose section-heading scan (for doc-type detection on docs w/o chapters, e.g. AML)
_SEC_HITS_RE = re.compile(
    r"^\s*\d{1,3}[A-Z]?\.\s+[A-Z][^\n]{2,90}\s*$", re.MULTILINE
)
# Rules/amendments put heading + body on ONE line: "1. Short title and commencement.—(1) ..."
_RULE_RE = re.compile(
    r"^\s*(?P<num>\d{1,3}[A-Z]?)\s*\.\s+(?P<title>[A-Z][^.\u2014\u2013-]{2,90})\s*\.\s*[\u2014\u2013-]"
)
# OCR variant (letter-spaced scans): bare "2." line, title on the NEXT line
_BARE_NUM_RE = re.compile(r"^\s*(\d{1,3}[A-Z]?)\s*\.\s*$")
_TITLE_LINE_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 ,'&()\-/.]{2,120})\.?\s*$")
_AMEND_RE = re.compile(r"Amendment of (?:section|sub-section)", re.IGNORECASE)
_RULES_CALLED_RE = re.compile(
    r"these\s+(?:rules|regulations)\b.{0,40}?(?:may|shall)\s+be\s+called",
    re.IGNORECASE | re.DOTALL,
)
_PAGE_NO_RE = re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
_PURE_NUM_RE = re.compile(r"^\s*-?\d+\s*-?\s*$")


# ---------------------------------------------------------------------------
# Judgement patterns
# ---------------------------------------------------------------------------

# Mastheads seen on court-issued sheets: IHC/LHC "ORDER SHEET" /
# "JUDGMENT SHEET", SC appeal papers, Sindh's inverted
# "IN THE HIGH COURT OF SINDH, KARACHI" and PLJ's "[Lahore High Court, Lahore]".
_JUDGEMENT_MASTHEAD_RE = re.compile(
    r"(?:JUDGMENT|ORDER)\s*SHEET\.?"
    r"|(?:LAHORE|ISLAMABAD|SINDH|PESHAWAR|BALOCHISTAN)\s+HIGH\s+COURT"
    r"|HIGH\s+COURT\s+OF\s+SINDH"
    r"|SUPREME\s+COURT\s+OF\s+PAKISTAN",
    re.IGNORECASE,
)
_DECIDED_ON_RE = re.compile(r"decided\s+on\s*:?\s*(?P<date>[^\n]{3,60})", re.IGNORECASE)
# Headnote digest signature: "(a)" style index items carrying statute refs:
# "(a)Constitution of Pakistan, 1973---S.Article 199 ..."  (PLD/digilawyer
# style uses triple dashes; PLJ reports use "--" right after the Act name).
_HEADNOTE_SPLIT_RE = re.compile(r"(?m)^\(([a-z])\)\s*")
_STAT_CITE_RE = re.compile(r"(?m)^\([a-z]\)\s*(?P<act>.{3,140}?)\s*[-\u2014\u2013]{2,}")
_STAT_CITE_ALT_RE = re.compile(
    r"(?m)^(?P<act>[A-Z][^\n]{3,130}?)\s*[-\u2014\u2013]{2,}\s*(?=Ss?\.)"
)
# Distinctive statute-reference shorthand used in headnotes ("S.295-A",
# "Ss. 497 & 497(5)", "S.Section 22-A", "S.Article 10-A").
_SEC_REF_RE = re.compile(r"\bSs?\.\s*(?:Section\s+|Article\s+)?\d")
# PLD-style judgment openings put the authoring judge straight into the
# narrative across an em-dash: "ISHTIAQ IBRAHIM, C.J.---By invoking ..."
_JUDGE_DASH_OPENING_RE = re.compile(r"(?:C\.J\.|HCJ\b|J\.)\s*[-\u2014\u2013]{2,}\s*[A-Z]")
_COURT_PATTERNS = [
    (re.compile(r"SUPREME\s+COURT\s+OF\s+PAKISTAN", re.I), "Supreme Court of Pakistan"),
    (re.compile(r"LAHORE\s+HIGH\s+COURT", re.I), "Lahore High Court"),
    (re.compile(r"ISLAMABAD\s+HIGH\s+COURT", re.I), "Islamabad High Court"),
    (re.compile(r"SINDH\s+HIGH\s+COURT|HIGH\s+COURT\s+OF\s+SINDH", re.I), "Sindh High Court"),
    (re.compile(r"PESHAWAR\s+HIGH\s+COURT", re.I), "Peshawar High Court"),
    (re.compile(r"BALOCHISTAN\s+HIGH\s+COURT", re.I), "Balochistan High Court"),
]
# "Justice Jamal Khan Mandokhail" / "Mr Justice Aamer Farooq, HCJ"
_JUSTICE_RE = re.compile(
    r"\b(?:Mr\.?\s*)?Justice\s+(?:Mr\.?\s*)?[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}"
)
# Signature blocks, both shapes:
#   "( MUHAMMAD AZAM KHAN) \n JUDGE"      (name line + JUDGE line)
#   "MUHAMMAD AZAM KHAN, J."              (single line)
_SIG_NAME_RE = re.compile(
    r"^[ \t]*\W{0,3}\(?\s*(?P<n>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})\s*\)?"
    r"(?:\s*,\s*|\s*\)?\s*\n\s*)(?:HCJ\b|C\.J\.|J\.|JUDGE)",
    re.MULTILINE,
)
_SHEET_DATE_RE = re.compile(r"\b\d{1,2}[.-]\d{1,2}[.-]\d{2,4}\b")
_MONTH_DATE_RE = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s*,?\s*\d{4}")
_VS_LINE_RE = re.compile(
    r"^[ \t]*(?P<a>[^\n]{3,90}?)\s*:?\s*\n"
    r"[ \t]*(?:Versus|VERSUS|Vs\.?|VS\.?|v\.?|versus)\s*:?[ \t]*\n"
    r"[ \t]*(?P<b>[^\n]{3,90})",
    re.MULTILINE,
)
# Sindh order sheets label the parties instead of using a "Versus" line:
#   "APPLICANT  : Saleem Khalid son of ..., \n through Mr. Adnan Ali, Advocate."
_PARTY_LBL_RE = re.compile(
    r"(?mi)^\s*(?:APPLICANT|PETITIONER|APPELLANT)\s*(?:name)?\s*[:\uFF1A]?\s*(?P<n>[^\n]{3,90})"
)
_RESP_LBL_RE = re.compile(
    r"(?mi)^\s*RESPONDENTS?\s*(?:name)?\s*[:\uFF1A]?\s*(?P<n>[^\n]{3,90})"
)
# "Present: Tariq Saleem Sheikh, J." / bare "Tariq Saleem Sheikh, J." lines
_NAME_J_RE = re.compile(
    r"(?m)^(?:Present\s*:?\s*\n?\s*)?"
    r"(?P<n>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3})\s*,\s*(?:HCJ|C\.J\.|J\.)\s*$"
)
# Case numbers across every observed shape:
#   "W.P. No. 1553 of 2026" / "Crl.Misc.No.44216-B/2021" /
#   "CRIMINAL REVISIONS NOS. 173 OF 2024" / "Criminal Revision No.163/2023"
_CASE_NO_RE = re.compile(
    r"(?P<label>[A-Za-z][A-Za-z ./&()]{0,45}?"
    r"N[Oo]s?\.\s*\.?\s*)"
    r"(?P<num>\d{1,6}(?:-[A-Z]+)?)"
    r"\s*(?:[oO][fF]|/)\s*"
    r"(?P<year>\d{4})"
)
# Caption variant where the line OPENS with the number token, honourific-less:
#   "NO.128 OF 2024"  /  "No. 56 of 2017"
_CASE_NO_BARE_RE = re.compile(
    r"^N[Oo]s?\.\s*\.?\s*(?P<num>\d{1,6}(?:-[A-Z]+)?)\s*(?:[oO][fF]|/)\s*(?P<year>\d{4})"
)
_DISPOSITION_RE = re.compile(
    r"(?:leave to appeal\s+(?:is\s+)?refused"
    r"|(?:petitions?|appeals?|bail\s+application|revision(?:\s+petition)?)"
    r"\s+(?:is|was|are|were|stands?)?\s*(?:accordingly\s+|hereby\s+)?dismissed"
    r"|dismiss(?:ed)?\s+(?:the\s+)?instant\s+(?:bail\s+)?(?:application|petition)"
    r"|revision\s+(?:petition\s+)?(?:is\s+)?(?:accepted|dismissed)"
    r"|bail\s+(?:was\s+|is\s+|after\s+arrest\s+|post[\s-]*arrest\s+)?(?:granted|confirmed|accepted)"
    r"|conviction\s+(?:and\s+sentence\s+)?(?:is\s+)?maintained"
    r"|sentence\s+(?:is\s+)?(?:reduced|maintained|enhanced)"
    r"|impugned\s+(?:order|judgment)\s+(?:is\s+)?(?:set[\s-]*aside|quashed|upheld)"
    r"|file be consigned to record"
    r")",
    re.IGNORECASE,
)
# Numbered judgement paragraphs: "12. In view of the above ..." — anchored so
# dates ("27.07.2021", "3.3.2025. Mr. Imran Khan...") never match (a digit
# right after the dot fails the lookahead). Separators tolerate NBSPs and a
# line-wrap straight after the dot, both common in PDF extraction.
_PARA_MARK_RE = re.compile(
    r"(?m)^[^\S\n]{0,14}(\d{1,3})\.[^\S\n]*(?:\n[^\S\n]*)?(?=[^\d\s])"
)
_MIN_PARA_RUN = 4        # consecutive N, N+1, N+2, N+3 marks -> structured body

# Per-line boilerplate (download watermarks, form codes, timestamps, digilawyer
# AI artifacts). These are *scrubbed inline* so a watermark glued to a content
# line does not cost the whole line.
_BOILERPLATE_RES = [
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"digilawyer|jurisdiction\s*=|Ark\s+is\s+generating", re.IGNORECASE),
    re.compile(r"\d{1,2}/\d{1,2}/\d{4},\s*\d{1,2}:\d{2}"),
    re.compile(r"Form\s*No[:.]?", re.IGNORECASE),
]
# Case-number labels that betray a mid-sentence capture rather than a real
# case caption line ("... in Criminal Revision No 163/2023", "FIR No 45/2016")
# are filtered at extraction time via the lowercase-word check below.


def _looks_like_judgement(text: str) -> bool:
    """Content-scored detection, run FIRST in the ladder.

    Judgements must be caught before the statute profile: their numbered
    paragraphs trip _SEC_HITS_RE. The masthead is checked in a tight 800-char
    window (true mastheads sit at the very top), so statutes that merely
    discuss 'High Court' powers deep in their preamble stay below the bar.
    """
    masthead_zone = text[:800]
    head = text[:1500]
    score = 0
    if _JUDGEMENT_MASTHEAD_RE.search(masthead_zone):
        score += 3
    if _DECIDED_ON_RE.search(head):
        score += 2
    # "(a)" headnote items with statute refs — digest signature, never statutes
    if re.search(r"\([a-z]\)\s*[^\n]{3,140}[-\u2014\u2013]{2,}", text[:4000]):
        score += 3
    # PLD/PLJ author-judge dash opening: "ISHTIAQ IBRAHIM, C.J.---By invoking..."
    # Very strong signature — no statute/rule text contains this construction.
    if _JUDGE_DASH_OPENING_RE.search(head):
        score += 3
    if re.search(r"\bversus\b|\bVs\.?\b|\bv\.\s", head, re.I):
        score += 1
    if re.search(r"Crl|Criminal|W\.?P\.?|Petition|Appeal|Bail", head, re.I):
        score += 1
    if re.search(
        r"\bPetitioner\b|\bRespondent\b|\bAppellant\b|\bApplicant\b|\bComplainant\b",
        head,
    ):
        score += 1
    return score >= 4


def _extract_judgement_meta(text: str) -> dict:
    """Document-level metadata, attached to every chunk via the context prefix."""
    meta: dict = {
        "court": None, "case_nos": [], "parties": None, "judges": [],
        "decided_on": None, "statutes_cited": [], "disposition": None,
    }
    head = text[:3000]

    for rx, court in _COURT_PATTERNS:
        if rx.search(head):
            meta["court"] = court
            break

    seen_nums = set()
    # Scan caption-style lines only: real case numbers sit on short standalone
    # lines ("CRIMINAL PETITION NO.128 OF 2024", "Crl.Misc.No.44216-B/2021."),
    # while prose references ("...passed by the IHC in Criminal Revision
    # No.163/2023") live on long or parenthetical lines.
    for line in head.splitlines():
        ls = line.strip()
        if len(ls) > 90 or ls.startswith("("):
            continue
        for m in _CASE_NO_RE.finditer(ls):
            key = (m.group("num"), m.group("year"))
            if key in seen_nums:
                continue
            # Reject prose references. Three tells: a lowercase word opening
            # the captured label ("in Criminal Revision No.163/2023"), a
            # lowercase connector before the match ("... by filing Criminal
            # Revision No.121/2023" — "of" is whitelisted, it is the standard
            # "No. 56 of 2017" connector), or an "FIR" in label/prefix.
            label_clean = re.sub(r"\s+", " ", m.group("label")).strip()
            first_alpha = re.sub(r"[^A-Za-z]", "", label_clean.split(" ")[0]) if label_clean else ""
            # Mid-label lowercase words ("High Court by filing Criminal
            # Revision No.") betray a swallowed prose prefix; real captions
            # are all case-type tokens ("Cr. Bail Application No.").
            mid_words = label_clean.split(" ")[:-1] if len(label_clean.split(" ")) > 1 else []
            pre_raw = re.findall(r"[A-Za-z]+", ls[: m.start()])
            if (
                not first_alpha
                or (len(first_alpha) > 1 and first_alpha.islower())
                or any(len(w) > 1 and w.islower() and w.isalpha() for w in mid_words)
                or (pre_raw and pre_raw[-1].lower() != "of"
                    and len(pre_raw[-1]) > 1 and pre_raw[-1].islower())
                or "fir" in label_clean.lower().split()
                or "fir" in (w.lower() for w in pre_raw)
            ):
                continue
            seen_nums.add(key)
            no = f"{label_clean} {m.group('num')}/{m.group('year')}".strip()
            if len(no) <= 60:
                meta["case_nos"].append(no)
            if len(meta["case_nos"]) >= 8:
                break
        # Caption lines that open straight with the number token.
        mb = _CASE_NO_BARE_RE.match(ls)
        if mb:
            key = (mb.group("num"), mb.group("year"))
            if key not in seen_nums:
                seen_nums.add(key)
                meta["case_nos"].append(f"No.{mb.group('num')}/{mb.group('year')}")
        if len(meta["case_nos"]) >= 8:
            break
        if len(meta["case_nos"]) >= 8:
            break

    def _clean_party(p: str) -> Optional[str]:
        p = re.split(r"\s*\u2026+\s*|\s*-{2,}\s*", p)[0]  # cut "…" / "--" role markers
        p = re.sub(r"\s+", " ", p).strip(" .,\u2026-")
        return (p[:80] or None)

    vm = _VS_LINE_RE.search(head)
    if vm:
        a, b = _clean_party(vm.group("a")), _clean_party(vm.group("b"))
        if a and b:
            meta["parties"] = f"{a} vs {b}"
    else:
        pm, rm = _PARTY_LBL_RE.search(head), _RESP_LBL_RE.search(head)
        if pm:
            a = re.split(r",\s*(?:through|via)\b|\s+through\s+", pm.group("n"))[0]
            a = _clean_party(a)
            b = None
            if rm:
                b = re.split(r",\s*(?:through|via)\b|\s+through\s+", rm.group("n"))[0]
                b = _clean_party(b)
            if a and b:
                meta["parties"] = f"{a} vs {b}"
            elif a:
                meta["parties"] = a

    judges: List[str] = []
    for m in _JUSTICE_RE.finditer(text[:4000]):
        name = re.sub(r"\s+", " ", m.group(0)).replace("Mr. ", "").strip()
        if name not in judges:
            judges.append(name)
    for rx in (_NAME_J_RE,):
        for m in rx.finditer(text[:4000] + "\n" + text[-900:]):
            name = m.group("n").title()
            if name not in judges:
                judges.append(name)
    for m in _SIG_NAME_RE.finditer(text[-900:]):
        name = m.group("n").title()
        if name not in judges:
            judges.append(name)
    meta["judges"] = judges[:6]

    dm = _DECIDED_ON_RE.search(head)
    if dm:
        raw_date = dm.group("date")
        sm = _SHEET_DATE_RE.search(raw_date) or _MONTH_DATE_RE.search(raw_date)
        val = sm.group(0).strip() if sm else raw_date.strip(" .,")
        if val and not re.search(r"NaN|Invalid", val, re.I):
            meta["decided_on"] = val
    if not meta["decided_on"]:
        dates = _SHEET_DATE_RE.findall(text)
        if dates:
            meta["decided_on"] = dates[-1]

    lower_seen = set()
    for rx in (_STAT_CITE_RE, _STAT_CITE_ALT_RE):
        for m in rx.finditer(text):
            act = re.sub(r"\s+", " ", m.group("act")).strip(" .,;\u2014-")
            if act and act.lower() not in lower_seen:
                lower_seen.add(act.lower())
                meta["statutes_cited"].append(act)
            if len(meta["statutes_cited"]) >= 12:
                break
        if len(meta["statutes_cited"]) >= 12:
            break

    disp = None
    region = text[-1200:] if len(text) > 1200 else text
    for m in _DISPOSITION_RE.finditer(region):
        disp = m.group(0)
    if disp:
        meta["disposition"] = re.sub(r"\s+", " ", disp).lower().strip(" .,")
    return meta


def _judgement_context(jmeta: Optional[dict], para_num: Optional[str]) -> str:
    """Contextual prefix embedded into every chunk's text."""
    if not jmeta:
        return ""
    bits: List[str] = []
    if jmeta.get("parties"):
        bits.append(jmeta["parties"])
    if jmeta.get("court"):
        bits.append(jmeta["court"])
    if jmeta.get("case_nos"):
        bits.append("; ".join(jmeta["case_nos"][:3]))
    if not bits:
        return ""
    ctx = "[Case: " + " | ".join(bits) + "]"
    if para_num:
        ctx += f" [Para {para_num}]"
    return ctx[:250]


def _is_boilerplate_line(s: str) -> bool:
    if s.lower() == "pdf":
        return True
    return any(rx.search(s) for rx in _BOILERPLATE_RES)


def _scrub_boilerplate(s: str) -> str:
    """Remove watermark substrings glued inside otherwise-useful lines."""
    out = s
    for rx in _BOILERPLATE_RES:
        out = rx.sub(" ", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _clean_judgement_text(pages: List[str]) -> str:
    """Judgement-specific cleaning before structural parsing.

    Drops page furniture that generic cleaning cannot know about: download
    watermarks (digilawyer URLs, 'jurisdiction=pakistan 1/12', timestamp
    lines, digilawyer AI placeholders), form codes (HCJD/C-121), and
    *running footers* — exact-match short lines repeated on >=3 pages (e.g.
    'Criminal Revisions Nos. 173 and 180/2024' under every page). Masthead
    detection runs on the RAW text upstream, so stripping here never breaks
    doc-type detection.
    """
    line_counts: dict = {}
    parsed_pages: List[List[str]] = []
    for pg in pages:
        ls: List[str] = []
        for ln in pg.splitlines():
            s = ln.strip()
            if not s:
                continue
            if _PAGE_NO_RE.match(s) or _PURE_NUM_RE.match(s):
                continue
            if _is_boilerplate_line(s):
                continue
            s = _scrub_boilerplate(s)
            if not s:
                continue
            ls.append(s)
            if len(s) <= 90:
                line_counts[s] = line_counts.get(s, 0) + 1
        parsed_pages.append(ls)
    footers = {s for s, c in line_counts.items() if c >= 3}
    out: List[str] = []
    for ls in parsed_pages:
        out.extend(s for s in ls if s not in footers)
    return "\n".join(out)


def _mono_runs(marks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Index ranges of marks whose paragraph numbers ascend strictly by 1.

    Handles numbering restarts (multi-proceeding order sheets): each maximal
    run >= _MIN_PARA_RUN qualifies independently.
    """
    runs: List[Tuple[int, int]] = []
    i, n = 0, len(marks)
    while i < n:
        j = i
        while j + 1 < n and marks[j + 1][1] == marks[j][1] + 1:
            j += 1
        if j - i + 1 >= _MIN_PARA_RUN:
            runs.append((i, j))
        i = j + 1
    return runs


def _split_headnote_items(preamble: str) -> Tuple[List[str], List[str]]:
    """Split the pre-body region into '(x)' headnote items and plain remainder.

    Digest documents interleave: TITLE / case-no / decided-on, then
    "(a) Act---refs---holding", counsel lines, "(b) ...". Everything inside an
    (x)-span becomes a headnote item; segments outside them are plain preamble.
    """
    ms = list(_HEADNOTE_SPLIT_RE.finditer(preamble))
    if not ms:
        return [], ([preamble] if preamble.strip() else [])
    plain_parts = [preamble[: ms[0].start()]]
    head_items: List[str] = []
    for k, m in enumerate(ms):
        end = ms[k + 1].start() if k + 1 < len(ms) else len(preamble)
        head_items.append(preamble[m.end():end])
    return (
        [h for h in head_items if h.strip()],
        [p for p in plain_parts if p.strip()],
    )


def _iter_judgement_blocks(text: str) -> List[Block]:
    """Masthead/preamble -> headnotes -> numbered paragraphs -> tail.

    The body is considered structured when the text contains a monotonic run
    of >= _MIN_PARA_RUN numbered-paragraph markers. That qualifying run only
    *establishes* the body start; from there on EVERY "N." marker becomes a
    split point, so paragraphs beyond a stray marker ("37 ETO 2002 ..." at a
    line start), numbering restarts in multi-proceeding sheets, and trailing
    runs shorter than _MIN_PARA_RUN are all captured. Spurious splits produce
    tiny blocks that `_merge_tiny` folds into their neighbour.

    Text between two markers stays with the preceding paragraph; everything
    after the last marker (signatures, dates, 'Petition dismissed') rides on
    the final paragraph, where the disposition scan will find it.
    """
    marks = [(m.start(), int(m.group(1))) for m in _PARA_MARK_RE.finditer(text)]
    marks = [(p, n) for p, n in marks if n < 500]   # ignore absurd numbers
    runs = _mono_runs(marks)

    blocks: List[Block] = []
    if runs:
        body_start = marks[runs[0][0]][0]
        preamble_text = text[:body_start]
        head_items, plain_parts = _split_headnote_items(preamble_text)
        if plain_parts:
            blocks.append(
                Block("preamble", None, "", "\n".join(plain_parts).strip(),
                      None, None, meta={"keep": True})
            )
        for h in head_items:
            blocks.append(
                Block("headnote", None, "", h.strip(), None, None,
                      meta={"keep": True, "is_headnote": True})
            )
        # Accept a marker as a paragraph boundary when it continues the
        # running sequence (N -> N+1) or restarts numbering downward (fresh
        # proceedings inside one order sheet renumber from 1). Upward jumps
        # ("12." then a line starting "37 ETO 2002") are strays: their text
        # simply stays inside the preceding paragraph.
        bounds: List[Tuple[int, int]] = []
        for pos, num in (m for m in marks if m[0] >= body_start):
            if not bounds or num == bounds[-1][1] + 1 or num < bounds[-1][1]:
                bounds.append((pos, num))
        for k, (pos, num) in enumerate(bounds):
            end = bounds[k + 1][0] if k + 1 < len(bounds) else len(text)
            body = text[pos:end].strip()
            if body:
                blocks.append(Block("para", str(num), "", body, None, None))
    else:
        # Order sheets without reliable numbering: one flowing block; the
        # normaliser splits it into parent + overlapping children.
        if text.strip():
            blocks.append(Block("body", None, "", text.strip(), None, None))
    return blocks


# ---------------------------------------------------------------------------
# 1) PDF extraction from the `data` folder
# ---------------------------------------------------------------------------


def _read_pdf_pages(path: str) -> List[str]:
    """Extract per-page text from a (possibly AES-encrypted) PDF."""
    reader = PdfReader(path)
    if reader.is_encrypted:
        reader.decrypt("")
    return [page.extract_text() or "" for page in reader.pages]


def _clean_lines(text: str) -> List[str]:
    """Drop page-number lines / headers; keep blank lines as structure breaks."""
    out: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if _PAGE_NO_RE.match(s) or _PURE_NUM_RE.match(s):
            continue
        out.append(ln.rstrip())
    return out


def _title_from_filename(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    stem = re.sub(r"\s*\(.*?\)", "", stem)      # strip "(1)UPDATED"
    stem = re.sub(r"[\s_]+", " ", stem).strip()
    return stem


def _looks_garbled(text: str) -> bool:
    """OCR-quality heuristic: many single-char tokens (letter-spaced scans)
    or a word-per-line layout."""
    tokens = text.split()
    if not tokens:
        return False
    singles = sum(1 for t in tokens if len(t) == 1)
    if singles / len(tokens) > 0.25:
        return True
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and len(tokens) / len(lines) < 3:
        return True
    return False


def _looks_like_gazette(text: str) -> bool:
    """Gazette S.R.O. notifications (rules/regulations) carry a distinctive
    header. Score signals from the first ~2k chars only, so a statute that
    merely *mentions* 'S.R.O.' in its body is not misclassified."""
    head = text[:2000]
    score = 0
    if re.search(r"S\.?\s*R\.?\s*O\.?\s*\d", head, re.I):
        score += 2
    if re.search(r"\bNOTIFICATION\b", head, re.I):
        score += 1
    if re.search(r"\bGAZETTE\b", head, re.I) and re.search(r"EXTRAORDINARY", head, re.I):
        score += 1
    if re.search(r"ISLAMABAD,? the \d", head, re.I):
        score += 1
    return score >= 2


def detect_doc_type(text: str, filename: str) -> str:
    """Detection ladder -> which profile to chunk with.

    Order matters:
      * judgements first — content-scored, because their numbered paragraphs
        would otherwise trip the statute heuristic;
      * rules/amendment docs ALSO contain CHAPTER/PART headings and numbered
        section lines, so they must be caught before the statute check.
    """
    if _looks_like_judgement(text):
        return "judgement"
    low = filename.lower()
    if "order" in low or "writ" in low:
        return "order"
    if _AMEND_RE.search(text) or re.search(r"to amend the .+ act", text, re.I):
        return "amendment"
    # Gazette S.R.O. notifications / rules / regulations
    if _looks_like_gazette(text) or _RULES_CALLED_RE.search(text):
        return "rules"
    # Full statutes: PART/CHAPTER headings, or many numbered section headings
    if _PART_RE.search(text) or _CHAPTER_RE.search(text):
        return "statute"
    if len(_SEC_HITS_RE.findall(text)) >= 5:
        return "statute"
    return "paragraph"


def extract_documents(data_dir: str = "data") -> List[ExtractedDocument]:
    """Extract every *.pdf in `data_dir` into an ExtractedDocument.

    Skips unreadable files gracefully (logs to stderr) so one bad PDF never
    aborts the whole batch.
    """
    docs: List[ExtractedDocument] = []
    if not os.path.isdir(data_dir):
        print(f"[chunking] data dir not found: {data_dir}", file=sys.stderr)
        return docs
    for name in sorted(os.listdir(data_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(data_dir, name)
        try:
            pages = _read_pdf_pages(path)
        except Exception as exc:  # AES-encrypted w/o pwd, corrupt, etc.
            print(f"[chunking] SKIP {name}: {exc}", file=sys.stderr)
            continue
        text = "\n".join(pages)
        dtype = detect_doc_type(text, name)
        docs.append(
            ExtractedDocument(
                name=name,
                path=path,
                title=_title_from_filename(name),
                pages=pages,
                text=text,
                doc_type=dtype,
                needs_ocr=_looks_garbled(text),
                judgement_meta=_extract_judgement_meta(text) if dtype == "judgement" else None,
            )
        )
    return docs


# ---------------------------------------------------------------------------
# 2) Structural parsers (one per profile)
# ---------------------------------------------------------------------------


def _iter_statute_blocks(text: str) -> List[Block]:
    """Part -> Chapter -> Section. Section bodies accumulate until the next
    heading; subsection/clause lines inside a body are kept as-is.

    The front CONTENTS table is NOT stripped here: its heading-only copies of
    every section are removed later by `_dedupe_blocks` (the real body version
    of a section always has more text than its contents-table copy).
    """
    lines = _clean_lines(text)
    blocks: List[Block] = []
    part = chapter = None
    cur_num = cur_title = None
    cur_body: List[str] = []

    def close():
        nonlocal cur_num, cur_title, cur_body
        if cur_num is not None:
            blocks.append(
                Block(
                    level="section", num=cur_num, title=cur_title or "",
                    body="\n".join(cur_body).strip(), part=part, chapter=chapter,
                )
            )
        cur_num = cur_title = None
        cur_body = []

    for ln in lines:
        s = ln.strip()
        m = _PART_RE.match(s)
        if m:
            close(); part, chapter = m.group(1), None; continue
        m = _CHAPTER_RE.match(s)
        if m:
            close(); chapter = m.group(1); continue
        m = _SECTION_RE.match(s)
        if m:
            close()
            cur_num, cur_title = m.group("num"), m.group("title").strip()
            # Many acts put the first clause on the same line:
            # "1. Short title, extent and commencement.—(1) This Act may be called..."
            # Split heading from inline body at the em/en-dash marker.
            parts = re.split(r"\s*\.\s*[\u2014\u2013-]\s*", cur_title, maxsplit=1)
            if len(parts) == 2:
                cur_title, first_body = parts[0].strip(), parts[1].strip()
                cur_body = [first_body] if first_body else []
            else:
                cur_body = []
            continue
        if cur_num is not None:
            cur_body.append(ln)
    close()
    return blocks


def _iter_rule_blocks(text: str, level: str) -> List[Block]:
    """Rules / amendment acts: 'N. Short title.—(1) body...' same-line style.

    Also handles the OCR letter-spaced variant where the number, the title and
    the dash land on separate lines ("2." / "Amendment of section 2, Act XL of
    2016." / "—" / body)."""
    lines = _clean_lines(text)
    blocks: List[Block] = []
    cur_num = cur_title = None
    cur_body: List[str] = []

    def close():
        nonlocal cur_num, cur_title, cur_body
        if cur_num is not None:
            meta = {}
            m = re.search(r"Amendment of (?:section|sub-section)\s+(\d+[A-Z]?)", cur_title or "", re.I)
            if m:
                meta["amends_section"] = m.group(1)
            blocks.append(
                Block(level=level, num=cur_num, title=cur_title or "",
                      body="\n".join(cur_body).strip(), part=None, chapter=None, meta=meta)
            )
        cur_num = cur_title = None
        cur_body = []

    i = 0
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        m = _RULE_RE.match(s)  # same-line "N. Title.—(body)"
        if m:
            close()
            cur_num, cur_title = m.group("num"), m.group("title").strip()
            rest = re.sub(r"^[\u2014\u2013-]+\s*", "", s[m.end():].strip())
            cur_body = [rest] if rest else []
            i += 1
            continue
        m2 = _BARE_NUM_RE.match(s)  # OCR variant: "N." alone, title next line
        if m2:
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            tm = _TITLE_LINE_RE.match(lines[j].strip()) if j < n else None
            if tm:
                close()
                cur_num = m2.group(1)
                cur_title = tm.group(1).strip()
                k = j + 1
                while k < n and not lines[k].strip():
                    k += 1
                if k < n and re.match(r"^[\u2014\u2013-]+$", lines[k].strip()):
                    k += 1  # skip a lone dash line
                cur_body = []
                i = k
                continue
        if cur_num is not None:
            cur_body.append(lines[i])
        i += 1
    close()
    return blocks


def _split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if paras:
        return paras
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]


# ---------------------------------------------------------------------------
# 3) Normalisation: parent-child / small-to-big
# ---------------------------------------------------------------------------


def _dedupe_blocks(blocks: List[Block]) -> List[Block]:
    """Drop CONTENTS-table copies of sections; keep num-less blocks.

    Every statute lists each section in its front CONTENTS (heading only, no
    body) and again in the body (heading + body). For each (level, number) keep
    the version with the most body text — i.e. the real one — in first-seen
    (contents) order, which equals document order. Handles chapter-less acts
    (e.g. AML) where there is no CHAPTER anchor to cut on.

    Blocks without a number (judgement preambles / headnotes / flowing bodies)
    carry unique content and are always preserved in place.
    """
    chosen: dict = {}
    for i, b in enumerate(blocks):
        if b.num is None:
            continue
        key = (b.level, b.num)
        size = len(b.body.split())
        cur = chosen.get(key)
        if cur is None or size > cur[1] or (size == cur[1] and i > cur[2]):
            chosen[key] = (b, size, i)
    keep_idx = {v[2] for v in chosen.values()}
    return [b for i, b in enumerate(blocks) if b.num is None or i in keep_idx]


def _merge_tiny(pairs: List[Tuple[Block, str]], min_tokens: int) -> List[Tuple[Block, str]]:
    """Merge heading stubs (< min_tokens) into their neighbour.

    Blocks flagged meta['keep'] (judgement preambles / headnotes) are never
    merged away — they are small but retrieval-valuable in their own right.
    """
    out: List[Tuple[Block, str]] = []
    i, n = 0, len(pairs)
    while i < n:
        block, text = pairs[i]
        if (
            len(text.split()) < min_tokens
            and n > 1
            and not block.meta.get("keep")
        ):
            if i + 1 < n:
                nb, ntext = pairs[i + 1]
                pairs[i + 1] = (nb, text + "\n" + ntext)
                i += 1
                continue
            if out:  # last item -> fold into previous
                pb, ptext = out[-1]
                out[-1] = (pb, ptext + "\n" + text)
                i += 1
                continue
        out.append((block, text))
        i += 1
    return out


def _truncate(text: str, max_tokens: int) -> str:
    toks = text.split()
    return " ".join(toks[:max_tokens]) if len(toks) > max_tokens else text


def _structure_coverage_ok(blocks: List[Block], text: str, cfg: ChunkingConfig) -> bool:
    """Guard shared by all structural profiles: if parsing recovered almost
    nothing, callers fall back so we never emit a single useless chunk or
    silently lose a document."""
    parsed_tokens = sum(len(b.body.split()) + len((b.title or "").split()) for b in blocks)
    total_tokens = len(text.split())
    if not blocks:
        return False
    if not total_tokens:
        return True
    return parsed_tokens / total_tokens >= cfg.min_structure_coverage


class _Chunker:
    """Holds per-document state (id counter) and the finalisation logic."""

    def __init__(self, cfg: ChunkingConfig):
        self.cfg = cfg
        self._counter = 0

    # -- chunk construction -------------------------------------------------
    def make(self, doc: ExtractedDocument, block: Block, text: str,
             level: str, is_parent: bool = False,
             parent_id: Optional[str] = None) -> Chunk:
        self._counter += 1
        cid = f"{os.path.splitext(doc.name)[0]}:{level}:{self._counter}"
        meta = dict(block.meta)
        if is_parent:
            meta["is_parent"] = True
        # Contextual prefix: judgement chunks embed their provenance so a bare
        # paragraph remains answerable in isolation ("which case said this?").
        if doc.doc_type == "judgement":
            ctx = _judgement_context(doc.judgement_meta, block.num)
            if ctx:
                text = ctx + "\n" + text
                meta["context"] = ctx
        return Chunk(
            doc_name=doc.name, doc_type=doc.doc_type, level=level,
            text=text.strip(), section_no=block.num,
            section_title=block.title or None, part=block.part, chapter=block.chapter,
            meta=meta, chunk_id=cid, parent_id=parent_id,
            is_parent=is_parent, tokens=len(text.split()),
        )

    # -- tiny-stub merge + giant-section split ------------------------------
    def finalize(self, doc: ExtractedDocument, blocks: List[Block],
                 dedupe: bool = True) -> List[Chunk]:
        cfg = self.cfg
        # Judgement paragraphs legitimately repeat numbers across proceedings
        # (each judgment in a sheet renumbering from 1), so their parser output
        # is already unique — deduplication would discard real paragraphs.
        if dedupe:
            blocks = _dedupe_blocks(blocks)
        pairs: List[Tuple[Block, str]] = []
        for b in blocks:
            heading = f"{b.num}. {b.title}".strip(" .") if b.title else (b.num or "")
            text = (heading + "\n" + b.body).strip() if heading else b.body.strip()
            if text:
                pairs.append((b, text))
        pairs = _merge_tiny(pairs, cfg.min_chunk_tokens)

        chunks: List[Chunk] = []
        for block, text in pairs:
            if len(text.split()) <= cfg.section_split_threshold_tokens:
                chunks.append(
                    self.make(doc, block, text, level=block.level, is_parent=False)
                )
            else:
                parent = self.make(
                    doc, block, _truncate(text, cfg.parent_max_tokens),
                    level=block.level, is_parent=True,
                )
                children = self._split_long(doc, block, text, parent.chunk_id)
                chunks.append(parent)
                chunks.extend(children)
        return chunks

    # -- shared paragraph packing ---------------------------------------------
    def _expand_paragraphs(self, paras: List[str]) -> List[str]:
        """Sentence-split any paragraph bigger than max_chunk_tokens so a single
        paragraph can never balloon a chunk (e.g. the 24k-token CrPC §560)."""
        cfg = self.cfg
        out: List[str] = []
        for p in paras:
            if len(p.split()) <= cfg.max_chunk_tokens:
                out.append(p)
            else:
                out.extend(self._split_sentences_to_max(p))
        return out

    def _split_sentences_to_max(self, text: str) -> List[str]:
        cfg = self.cfg
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        pieces: List[str] = []
        cur: List[str] = []
        cur_tokens = 0
        for s in sents:
            st = len(s.split())
            if cur and cur_tokens + st > cfg.max_chunk_tokens:
                pieces.append(" ".join(cur))
                cur, cur_tokens = [], 0
            if st > cfg.max_chunk_tokens:  # single sentence still too big
                if cur:
                    pieces.append(" ".join(cur))
                    cur, cur_tokens = [], 0
                words = s.split()
                for i in range(0, len(words), cfg.max_chunk_tokens):
                    pieces.append(" ".join(words[i:i + cfg.max_chunk_tokens]))
            else:
                cur.append(s)
                cur_tokens += st
        if cur:
            pieces.append(" ".join(cur))
        return pieces

    def _pack(self, paras: List[str]):
        """Greedy pack of paragraphs into (paras, carry) yields; the last
        paragraph of a pack is carried into the next as overlap."""
        cfg = self.cfg
        cur: List[str] = []
        cur_tokens = 0
        carry = ""
        for p in paras:
            pt = len(p.split())
            if cur and cur_tokens + pt > cfg.max_chunk_tokens:
                yield list(cur), carry
                carry = cur[-1]
                cur, cur_tokens = [], len(carry.split())
            cur.append(p)
            cur_tokens += pt
        if cur or carry:
            yield list(cur), carry

    # -- giant section -> overlapping leaf children --------------------------
    def _split_long(self, doc: ExtractedDocument, block: Block,
                    text: str, parent_id: str) -> List[Chunk]:
        paras = self._expand_paragraphs(_split_paragraphs(text))
        children: List[Chunk] = []
        for pack, carry in self._pack(paras):
            body = self._join_pack(pack, carry)
            if not body:
                continue
            level = "subsection" if pack and re.match(r"\(\s*\d+[a-z]?\)", pack[0].lstrip()) else "paragraph"
            children.append(
                self.make(doc, block, body, level=level, is_parent=False, parent_id=parent_id)
            )
        return children

    # -- plain paragraph fallback (OCR / unstructured docs) -------------------
    def paragraph_chunks(self, doc: ExtractedDocument, text: str) -> List[Chunk]:
        # Keep the fallback as clean as the structural paths: strip watermark
        # substrings line-by-line before packing.
        text = "\n".join(
            s for s in (_scrub_boilerplate(ln) for ln in text.splitlines()) if s.strip()
        )
        paras = self._expand_paragraphs(_split_paragraphs(text))
        chunks: List[Chunk] = []
        for pack, carry in self._pack(paras):
            body = self._join_pack(pack, carry)
            if not body:
                continue
            chunks.append(
                self.make(doc, Block("paragraph", None, "", body, None, None),
                          body, level="paragraph")
            )
        return chunks

    @staticmethod
    def _join_pack(pack: List[str], carry: str) -> str:
        body = "\n\n".join(pack) if pack else ""
        if carry and body:
            return carry + "\n\n" + body
        return carry if carry else body


# ---------------------------------------------------------------------------
# 4) Dispatcher
# ---------------------------------------------------------------------------


def chunk_document(doc: ExtractedDocument, cfg: Optional[ChunkingConfig] = None) -> List[Chunk]:
    """Chunk one extracted document using its detected profile."""
    cfg = cfg or ChunkingConfig()
    chunker = _Chunker(cfg)
    text = doc.text
    dtype = doc.doc_type

    if dtype == "statute":
        blocks = _iter_statute_blocks(text)
    elif dtype in ("rules", "amendment"):
        blocks = _iter_rule_blocks(text, dtype)
    elif dtype == "judgement":
        jtext = _clean_judgement_text(doc.pages) or text
        blocks = _iter_judgement_blocks(jtext)
        if not _structure_coverage_ok(blocks, jtext, cfg):
            # Fallback must use the CLEANED text — the paragraph packer would
            # otherwise re-import the watermarks we just stripped.
            return chunker.paragraph_chunks(doc, jtext)
        chunks = chunker.finalize(doc, blocks, dedupe=False)
        # Flag operative language wherever it lands (usually the last
        # paragraph chunk carrying the signature block).
        for c in chunks:
            dm = _DISPOSITION_RE.search(c.text)
            if dm:
                c.meta["disposition"] = re.sub(r"\s+", " ", dm.group(0)).lower().strip(" .,")
        return chunks
    else:  # order -> single document chunk; paragraph -> raw fallback
        if dtype == "order":
            block = Block("document", None, "", text, None, None)
            return [chunker.make(doc, block, text, level="document")]
        return chunker.paragraph_chunks(doc, text)

    # guard: if structural parsing recovered almost nothing, fall back so we
    # never emit a single useless chunk or silently lose the document.
    if not _structure_coverage_ok(blocks, text, cfg):
        return chunker.paragraph_chunks(doc, text)
    return chunker.finalize(doc, blocks)


def chunk_documents(data_dir: str = "data",
                    cfg: Optional[ChunkingConfig] = None) -> Tuple[List[ExtractedDocument], List[Chunk]]:
    """Extract all PDFs from `data_dir`, then chunk each one."""
    docs = extract_documents(data_dir)
    all_chunks: List[Chunk] = []
    for d in docs:
        all_chunks.extend(chunk_document(d, cfg))
    return docs, all_chunks


# ---------------------------------------------------------------------------
# 5) CLI
# ---------------------------------------------------------------------------


def _summary(docs: List[ExtractedDocument], chunks: List[Chunk]) -> str:
    by_doc: dict = {}
    for c in chunks:
        by_doc.setdefault(c.doc_name, [0, 0])
        by_doc[c.doc_name][0] += 1
        by_doc[c.doc_name][1] = max(by_doc[c.doc_name][1], c.tokens)
    lines = [f"{d.name[:52]:54} {d.doc_type:10} ocr={str(d.needs_ocr):5} "
             f"chunks={by_doc.get(d.name, [0])[0]:4} max_tok={by_doc.get(d.name, [0, 0])[1]}"
             for d in docs]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Structure-aware hierarchical chunking for legal PDFs")
    ap.add_argument("--data", default="data", help="folder containing the legal PDFs")
    ap.add_argument("--out", default=None, help="optional JSON dump of all chunks")
    ap.add_argument("--min-chunk-tokens", type=int, default=20)
    ap.add_argument("--split-threshold", type=int, default=500)
    ap.add_argument("--max-chunk-tokens", type=int, default=600)
    ap.add_argument("--parent-max-tokens", type=int, default=2000)
    args = ap.parse_args(argv)

    cfg = ChunkingConfig(
        min_chunk_tokens=args.min_chunk_tokens,
        section_split_threshold_tokens=args.split_threshold,
        max_chunk_tokens=args.max_chunk_tokens,
        parent_max_tokens=args.parent_max_tokens,
    )

    docs, chunks = chunk_documents(args.data, cfg)
    print(f"extracted {len(docs)} documents, {len(chunks)} chunks\n")
    print(_summary(docs, chunks))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump([c.to_dict() for c in chunks], fh, indent=2, ensure_ascii=False)
        print(f"\nchunks written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
