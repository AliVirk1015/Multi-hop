"""
chunking.py — PDF extraction + structure-aware hierarchical chunking
=====================================================================

Target corpus: Pakistan legal PDFs in `data/` (statutes, gazette rules,
amendment acts, a court order).

Design
------
One pipeline, multiple *document profiles*, selected by a detection ladder,
with a graceful paragraph fallback so no document ever fails:

    statute    : Part -> Chapter -> Section -> (sub-section/clause)
    rules      : Rule N / numbered item          (S.R.O. notifications)
    amendment  : "Amendment of section N of Act X"
    order      : single-document chunk           (e.g. IHC order sheet)
    paragraph  : layout/paragraph fallback       (OCR-garbled PDFs)

Every profile feeds a single normalisation pass ("parent-child / small-to-big"):

    1. skip the CONTENTS table (it duplicates every heading)
    2. merge tiny heading stubs (< min_chunk_tokens) into a neighbour
    3. split giant sections (> section_split_threshold_tokens) into leaf
       children with paragraph-level overlap
    4. emit the full section as a *parent* chunk (small-to-big expansion),
       children carry `parent_id`

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
    tiny stubs are merged, giants are split.
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
    doc_type: str          # statute | rules | amendment | order | paragraph
    needs_ocr: bool


@dataclass
class Block:
    """A structural unit found by the parser (one section/rule/paragraph)."""

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

    Order matters: rules/amendment docs ALSO contain CHAPTER/PART headings and
    numbered section lines, so they must be caught before the statute check.
    """
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
        docs.append(
            ExtractedDocument(
                name=name,
                path=path,
                title=_title_from_filename(name),
                pages=pages,
                text=text,
                doc_type=detect_doc_type(text, name),
                needs_ocr=_looks_garbled(text),
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
    """Drop CONTENTS-table copies of sections.

    Every statute lists each section in its front CONTENTS (heading only, no
    body) and again in the body (heading + body). For each (level, number) keep
    the version with the most body text — i.e. the real one — in first-seen
    (contents) order, which equals document order. Handles chapter-less acts
    (e.g. AML) where there is no CHAPTER anchor to cut on.
    """
    best: dict = {}
    order: List[Tuple[str, str]] = []
    for i, b in enumerate(blocks):
        if b.num is None:
            continue
        key = (b.level, b.num)
        size = len(b.body.split())
        if key not in best:
            best[key] = (b, size, i)
            order.append(key)
        else:
            _, cur_size, cur_i = best[key]
            if size > cur_size or (size == cur_size and i > cur_i):
                best[key] = (b, size, i)
    return [best[k][0] for k in order]


def _merge_tiny(pairs: List[Tuple[Block, str]], min_tokens: int) -> List[Tuple[Block, str]]:
    """Merge heading stubs (< min_tokens) into their neighbour."""
    out: List[Tuple[Block, str]] = []
    i, n = 0, len(pairs)
    while i < n:
        block, text = pairs[i]
        if len(text.split()) < min_tokens and n > 1:
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
        return Chunk(
            doc_name=doc.name, doc_type=doc.doc_type, level=level,
            text=text.strip(), section_no=block.num,
            section_title=block.title or None, part=block.part, chapter=block.chapter,
            meta=meta, chunk_id=cid, parent_id=parent_id,
            is_parent=is_parent, tokens=len(text.split()),
        )

    # -- tiny-stub merge + giant-section split ------------------------------
    def finalize(self, doc: ExtractedDocument, blocks: List[Block]) -> List[Chunk]:
        cfg = self.cfg
        blocks = _dedupe_blocks(blocks)  # remove CONTENTS-table copies
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
    else:  # order -> single document chunk; paragraph -> raw fallback
        if dtype == "order":
            block = Block("document", None, "", text, None, None)
            return [chunker.make(doc, block, text, level="document")]
        return chunker.paragraph_chunks(doc, text)

    # guard: if structural parsing recovered almost nothing, fall back so we
    # never emit a single useless chunk or silently lose the document.
    parsed_tokens = sum(len(b.body.split()) + len((b.title or "").split()) for b in blocks)
    total_tokens = len(text.split())
    if blocks and total_tokens and parsed_tokens / total_tokens < cfg.min_structure_coverage:
        return chunker.paragraph_chunks(doc, text)
    if not blocks:
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
    ap.add_argument("--min-chunk-tokens", type=int, default=ChunkingConfig.min_chunk_tokens)
    ap.add_argument("--split-threshold", type=int, default=ChunkingConfig.section_split_threshold_tokens)
    ap.add_argument("--max-chunk-tokens", type=int, default=ChunkingConfig.max_chunk_tokens)
    ap.add_argument("--parent-max-tokens", type=int, default=ChunkingConfig.parent_max_tokens)
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
