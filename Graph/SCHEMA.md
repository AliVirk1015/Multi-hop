# Knowledge Graph — Step 1: Real Data Schema Report

_Generated: 2026-08-31T16:17:27+00:00_

This is the **data schema** (raw material). Step 2 derives the **ontology** (node/relationship types) from it — the two are different artifacts.

## chunks.json (Statute)

- **Records:** 1871  |  top-level: `list`
- **Documents:** 15

### Field schema

| Field | Type | Coverage | Unique | Example |
|---|---|---|---|---|
| `chapter` | str | 87% | 56 | 9 |
| `chunk_id` | str | 100% | 1871 | Anti Money Laundring 2010:section:1 |
| `doc` | str | 100% | 15 | Anti Money Laundring 2010.pdf |
| `doc_type` | str | 100% | 4 | statute |
| `is_parent` | bool | 100% | 2 | False |
| `level` | str | 100% | 6 | section |
| `meta` | dict | 5% | 5 | {'is_parent': True} |
| `parent_id` | str | 15% | 96 | Anti Money Laundring 2010:section:2 |
| `part` | str | 41% | 9 | I |
| `section_no` | str | 100% | 698 | 1 |
| `section_title` | str | 100% | 1570 | The following regulators are AML/CFT regulatory authoriti... |
| `text` | str | 100% | 1863 | 1. The following regulators are AML/CFT regulatory author... |
| `tokens` | int | 100% | 550 | 130 |

### Categorical distributions

- **`doc_type`** (4 distinct): statute: 1735 · rules: 103 · amendment: 32 · order: 1
- **`level`** (6 distinct): section: 1507 · paragraph: 252 · rules: 66 · amendment: 23 · subsection: 22 · document: 1
- **`part`** (9 distinct): VI: 172 · IX: 153 · III: 133 · I: 129 · IV: 55 · VII: 38 · II: 36 · V: 31 · VIII: 22
- **`chapter`** (56 distinct): IV: 310 · III: 142 · II: 106 · XLVI: 97 · XVII: 93 · V: 61 · X: 59 · VI: 53 · I: 52 · VII: 44 · XVI-A: 38 · IX: 32

### Per-document chunk counts

| Document | Chunks |
|---|---|
| THE CODE OF CRIMINAL PROCEDURE, 1898.pdf | 592 |
| Pakistan Penal Code.pdf | 517 |
| THE QANUN-E-SHAHADAT, 1984.pdf | 176 |
| Anti Money Laundring 2010.pdf | 112 |
| PECA 2016.pdf | 94 |
| PAKISTAN TELECOMMUNICATION (Reorg) 1996.pdf | 80 |
| PAYMENT SYSTEMS AND ELECTRONIC FUND.pdf | 78 |
| PECA ru2018-ocr (1).pdf | 50 |
| Electronic Transaction Ordinance 2002.pdf | 47 |
| Investigation of Fair Trail Act 2013.pdf | 39 |
| Peca-Act-2025 AMD.pdf | 32 |
| CERT 2023.pdf | 27 |
| RemovalBlockingofUnlawfulOnlineContentRules2021 (1)UPDATED.pdf | 15 |
| Pakistan Consumer Protection Regulation 2009 UPDATED.pdf | 11 |
| ncciaorder (1)UPDATED.pdf | 1 |

### Structure (parent / child)

- distinct `parent_id` values: 96  ·  chunks with a parent: 274  ·  `is_parent` true: 96  ·  false: 1775

### Token stats (per chunk)

- min **20** · median **126** · p95 **656** · max **2000** (n=1871)

### `meta` analysis

- `is_parent` (present in 96) — sample: True
- `amends_section` (present in 4) — sample: 2

### Detected quirks

- **chunk_id_vs_section_no**: chunk_id number is a SEQUENTIAL index, not the real section number. Always key on payload section_no, never the chunk_id integer.
  - checked_sections: `1359`
  - mismatch_count: `1299`
  - mismatch_ratio: `0.9558`
  - sample: `[{'chunk_id': 'Anti Money Laundring 2010:section:8', 'chunk_seq': 8, 'section_no': 3, 'doc': 'Anti Money Laundring 2010.pdf'}, {'chunk_id': 'Anti Money Laund...`
- **duplicate_section_no_in_doc**: Multiple chunks can share (doc, section_no) — legitimately, when a large section is split into parent + paragraph children. Not a real duplicate.
  - docs_with_dup_section_numbers: `0`
- **amendment_metadata**: doc_type='amendment' + meta.amends_section is a ready-made AMENDED_BY relationship (Tier 0 — reuse it, don't re-extract).
  - amendment_docs: `32`
  - with_amends_section_meta: `4`
- **broken_parent_links**: parent_id values that do not resolve to a chunk_id in the same file. These edges cannot be built unless the parent lives in the other file (rare) or the link is genuinely dangling.
  - count: `0`

## judgement_chunks.json (Judgement)

- **Records:** 1199  |  top-level: `list`
- **Documents:** 82

### Field schema

| Field | Type | Coverage | Unique | Example |
|---|---|---|---|---|
| `chapter` | null | 0% | 0 | — |
| `chunk_id` | str | 100% | 1199 | 2021LHC3627:preamble:1 |
| `doc` | str | 100% | 82 | 2021LHC3627.pdf |
| `doc_type` | str | 100% | 3 | judgement |
| `is_parent` | bool | 100% | 2 | False |
| `level` | str | 100% | 7 | preamble |
| `meta` | dict | 98% | 880 | {'keep': True, 'context': '[Case: Sheraz Khan vs The Stat... |
| `parent_id` | str | 14% | 61 | 2021LHC3627:para:8 |
| `part` | null | 0% | 0 | — |
| `section_no` | str | 83% | 43 | 2 |
| `section_title` | str | 1% | 14 | Briefly stated, the facts of the prosecution case are tha... |
| `text` | str | 100% | 1048 | [Case: Sheraz Khan vs The State, etc | Lahore High Court ... |
| `tokens` | int | 100% | 467 | 120 |

### Categorical distributions

- **`doc_type`** (3 distinct): judgement: 1180 · statute: 14 · paragraph: 5
- **`level`** (7 distinct): para: 882 · paragraph: 164 · preamble: 70 · headnote: 56 · section: 14 · body: 9 · subsection: 4
- **`part`** (0 distinct): 
- **`chapter`** (0 distinct): 

### Per-document chunk counts

| Document | Chunks |
|---|---|
| Muhammad_Ayyaz_Bin_Tariq_PECA_638536155567897347.pdf | 73 |
| j7.pdf | 51 |
| j4.pdf | 44 |
| j59.pdf | 44 |
| j61.pdf | 37 |
| j11.pdf | 34 |
| j30.pdf | 32 |
| j2.pdf | 30 |
| j13.pdf | 28 |
| j6.pdf | 28 |
| j27.pdf | 26 |
| j3.pdf | 25 |
| j58.pdf | 25 |
| j10.pdf | 24 |
| MjUzMjU4Y2Ztcy1kYzgz.pdf | 22 |
| j20.pdf | 21 |
| j31.pdf | 21 |
| j35.pdf | 21 |
| j37.pdf | 21 |
| j9.pdf | 21 |
| j69.pdf | 19 |
| MTQ1MjQy.pdf | 15 |
| MjAzNTcwY2Ztcy1kYzgz.pdf | 15 |
| j40.pdf | 15 |
| j66.pdf | 15 |
| 2021LHC3627.pdf | 14 |
| j29.pdf | 14 |
| j33.pdf | 14 |
| j53.pdf | 14 |
| j70.pdf | 14 |
| CRIMINAL_REVISIONS_NOS._173_OF_2024_638756537538516867.pdf | 13 |
| crl.p._128_2024.pdf | 13 |
| j14.pdf | 13 |
| j18.pdf | 13 |
| j19.pdf | 13 |
| j21.pdf | 13 |
| j23.pdf | 13 |
| j54.pdf | 13 |
| 975.pdf | 12 |
| j12.pdf | 12 |
| j24.pdf | 12 |
| j39.pdf | 12 |
| j60.pdf | 12 |
| j8.pdf | 12 |
| Criminal_Miscellaneous_No._201_of_2025_638767915493156673.pdf | 10 |
| MzAyNDEzY2Ztcy1kYzgz.pdf | 10 |
| j36.pdf | 10 |
| j5.pdf | 10 |
| j55.pdf | 10 |
| j64.pdf | 10 |
| Criminal_Miscellaneous_No._255_of_2025_638764333946605900.pdf | 9 |
| Criminal_Miscellaneous_No._256_of_2025_638763628850974550.pdf | 9 |
| Criminal_Revision_No._204_of_2024_638763620485172104.pdf | 9 |
| j63.pdf | 9 |
| Masud_ur_Rehman_Abbasi_bail_637754290732532008.pdf | 8 |
| Mjc1MjI1Y2Ztcy1kYzgz.pdf | 8 |
| MzEwNjgxY2Ztcy1kYzgz (1).pdf | 8 |
| MzEwNjgxY2Ztcy1kYzgz.pdf | 8 |
| j32.pdf | 8 |
| j56.pdf | 8 |
| pdf.pdf | 8 |
| MTYyOTEx.pdf | 7 |
| MjY0MzM2Y2Ztcy1kYzgz.pdf | 7 |
| j28.pdf | 7 |
| j38.pdf | 7 |
| j57.pdf | 7 |
| j62.pdf | 7 |
| MjAzODIxY2Ztcy1kYzgz (1).pdf | 6 |
| MjAzODIxY2Ztcy1kYzgz.pdf | 6 |
| Mjc5Njc1Y2Ztcy1kYzgz.pdf | 6 |
| j17.pdf | 5 |
| j15.pdf | 4 |
| j52.pdf | 4 |
| j67.pdf | 4 |
| j68.pdf | 4 |
| MTQ3NTI0.pdf | 3 |
| j34.pdf | 3 |
| j26.pdf | 2 |
| j51.pdf | 2 |
| j16.pdf | 1 |
| j25.pdf | 1 |
| j65.pdf | 1 |

### Structure (parent / child)

- distinct `parent_id` values: 61  ·  chunks with a parent: 163  ·  `is_parent` true: 61  ·  false: 1138

### Token stats (per chunk)

- min **14** · median **166** · p95 **1154** · max **2022** (n=1199)

### `meta` analysis

- `keep` (present in 177) — sample: True
- `context` (present in 1180) — sample: [Case: Sheraz Khan vs The State, etc | Lahore High Court | Crl.Misc.No. 44216...
- `is_parent` (present in 61) — sample: True
- `is_headnote` (present in 103) — sample: True
- `disposition` (present in 60) — sample: bail was granted

### Detected quirks

- **judgment_section_no_is_para_counter**: In judgment chunks, section_no is a paragraph/order counter, unrelated to statute sections. Do NOT create Section nodes from judgment chunks.
  - sample: `[{'chunk_id': '2021LHC3627:preamble:1', 'section_no': None}, {'chunk_id': '2021LHC3627:para:2', 'section_no': '2'}, {'chunk_id': '2021LHC3627:para:3', 'secti...`
- **case_header_context**: meta.context embeds the case header as '[Case: parties | Court | Case No]'. This is the structured seed for Tier 2 (LLM) Judgment/Court/Party extraction.
  - chunks_with_case_header: `1115`
- **broken_parent_links**: parent_id values not resolving to a chunk_id in this file.
  - count: `0`

## Implications for Step 2 (ontology design)

- Structure edges (`CONTAINS`/`PART_OF`/`PARENT_OF`) are fully derivable from `doc`/`part`/`chapter`/`section_no`/`parent_id` — Tier 0, zero LLM.
- `AMENDED_BY` comes free from `doc_type=amendment` + `meta.amends_section`.
- Judgment `Court`/`Party`/`case_no` are **not** structured fields — extract from `meta.context` + preamble text (Tier 2).
- Never key statute nodes on the `chunk_id` integer — use `section_no`.
