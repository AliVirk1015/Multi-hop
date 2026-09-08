"""
PROJECT_WALKTHROUGH.py
======================

A-to-Z technical walkthrough of the **Legal Multi-Hop Graph-RAG Agent**
(repository folder: ``legal_multihop_agent``).

WHAT THIS FILE IS
-----------------
A self-contained, read-me-first technical documentation / interview-prep
script. It explains the CURRENT codebase strictly from the source: every
function, class, file, config value and execution step below was verified by
reading the actual code. It is documentation, not a program - it imports
nothing and touches no databases, models or secrets.

HOW TO USE IT
-------------
* Read top to bottom: it follows the real flow (data -> index -> graph ->
  runtime -> frontend -> answer).
* Each section follows the same "interview lens" for its components:
  What does it do -> Why it exists -> How it works -> What it receives ->
  What it returns -> What calls it -> What happens next.
* Run it to print a quick table of contents:
      python PROJECT_WALKTHROUGH.py

ACCURACY CONVENTIONS
--------------------
* Where the docs and the code disagree, the CODE wins and the discrepancy is
  called out. README.md and parts of .env.example describe an EARLIER
  architecture (Milvus, a LangGraph "tool agent", Graph-Lite adjacency maps,
  modules like config.py / main.py / agent/ / tools/). None of those exist in
  the current tree and are labelled LEGACY / NOT IN CODE.
* Values labelled (observed) are snapshot numbers from the loaded data/DB,
  not code guarantees.
* No secret, API key or credential value is included anywhere in this file.
"""
from __future__ import annotations

# ==========================================================================
# 1. PROJECT PURPOSE AND OVERALL ARCHITECTURE
# ==========================================================================
SECTION_1 = r"""
1. PROJECT PURPOSE AND OVERALL ARCHITECTURE
===========================================

IN SIMPLE TERMS
---------------
An interviewer or researcher opens a ChatGPT-style web app and asks a question
about Pakistani cyber-crime / criminal law, e.g.:
    "What is the punishment for qatl-i-amd under the Pakistan Penal Code?"
The system (1) searches a vector database of statutes AND case-law judgments,
(2) optionally walks a Neo4j knowledge graph to pull in *related* provisions
and case law a text search would miss, (3) may run a second "hop" when more
context is needed, (4) sends the retrieved evidence to an LLM that writes a
structured, citation-grounded answer, (5) renders it in a clean chat UI with a
"Sources" panel, and (6) keeps the conversation in memory so follow-ups like
"what about an attempt to commit it?" are understood.

WHAT PROBLEM IT SOLVES
----------------------
* Pakistani statutes (PPC, CrPC, PECA, AML, ...) and judgments are long and
  heavily cross-referenced. A lawyer's reasoning usually needs several
  CONNECTED provisions (e.g. PPC 302 qatl-i-amd -> its attempt in 324 ->
  evidence rule in 304) plus relevant case law.
* Plain semantic search returns similar chunks but no structure. This project
  fuses three signals that correct each other:
    1) hybrid dense + sparse vector search   (recall: meaning + exact terms)
    2) cross-encoder re-ranking              (precision)
    3) a hand-built Neo4j knowledge graph    (coverage: structure + citations)
* The LLM is a *synthesizer* (evidence -> cited answer) and an optional *hop
  planner*, never a knowledge source - the main hallucination guard-rail.

TARGET USER / MAIN USE CASE
---------------------------
Primary: a cyber-crime / criminal-law research demo. Answer a natural-language
legal question with cited provisions + case law. It is explicitly not a
substitute for a licensed lawyer (every answer carries a caveat).

HIGH-LEVEL ARCHITECTURE (actual)
--------------------------------
 Documents (PDFs: statutes + judgments)
    |  (offline, once)
    v
 ingestion/chunking.py + ingestion/Judgement_chunker.py
    |  chunks.json (1871 statute chunks) / judgement_chunks.json (1199)
    v
 BGE-M3 embeddings (dense 1024-d + sparse lexical)  -> qdrant-store/*.json
    v
 qdrant-store/ingest_hybrid.py + ingest_judgements.py  ->  Qdrant
    |  legal_hybrid (statutes)  +  judgements (case law)
    |                                |
    |  Graph/ 5-step build  ---------> Neo4j AuraDB (knowledge graph)
    v                                v
 -------------------- RUNTIME (per question) --------------------
 User -> ui.py (Streamlit chat) -> api.answer()
      -> LangGraph multi-hop retriever (retrieval_graph/)
      -> retrieval/: BgeM3QueryEncoder -> QdrantHybridStore -> CrossEncoderReranker
      -> Graph/query.py KnowledgeGraphClient (Neo4j) when kg present
      -> evidence pool + kg_evidence -> llm_client.py (OpenAI-compatible)
      -> cited markdown answer -> ui.py renders + "Sources · N"

TECHNOLOGY STACK (actual imports)
---------------------------------
Python 3.11 venv | LangGraph (langgraph) | Qdrant + qdrant-client (hybrid,
RRF fusion) | BAAI/bge-m3 via FlagEmbedding (dense + SPLADE-style sparse) |
BAAI/bge-reranker-v2-m3 / ms-marco MiniLM cross-encoder | Neo4j AuraDB via the
neo4j driver | OpenAI-compatible LLM via httpx (llm_client.py) | FastAPI +
uvicorn (api.py) | Streamlit (ui.py) | pypdf (PDF text) | stdlib elsewhere.

RUN ENTRY POINTS
----------------
  .venv\Scripts\python.exe -m streamlit run ui.py        # chat UI
  .venv\Scripts\python.exe -m uvicorn api:app --port 8000  # FastAPI
  .venv\Scripts\python.exe main_query.py --query "..."     # CLI
"""

# ==========================================================================
# 2. REPOSITORY / FILE STRUCTURE AND RESPONSIBILITIES
# ==========================================================================
SECTION_2 = r"""
2. REPOSITORY / FILE STRUCTURE AND RESPONSIBILITIES
===================================================
(Real current tree. Legacy files such as config.py, main.py, agent/,
tools/, retrieval/ingest.py, graph_lite* do NOT exist now.)

legal_multihop_agent/
|-- ui.py                 Streamlit chat UI (product name "Verus"). Renders the
|                         chat, keeps conversation in st.session_state, calls
|                         api.answer() in the same process.
|-- api.py                FastAPI backend. answer() = full pipeline entry;
|                         endpoints GET /, GET /health, POST /query;
|                         lifespan warm-up; conversation-history support.
|-- main_query.py         CLI twin of the pipeline: --query / --no-llm /
|                         --multi-hop; defines make_llm_planner() and the
|                         defaults (DEFAULT_TOP_K=20, DEFAULT_RERANK_K=6,
|                         DEFAULT_MAX_HOPS=2).
|-- llm_client.py         Tiny OpenAI-compatible LLM client (httpx). Reads
|                         .env; llm_reachable(), llm_chat(), llm_chat_json();
|                         retries 429/5xx.
|-- requirements.txt      Dependencies (langgraph, qdrant-client, neo4j,
|                         fastapi, uvicorn, streamlit, FlagEmbedding comment,
|                         pypdf, cryptography, mlflow (legacy), ...).
|-- .env.example          Documented env template (MIXED: many vars legacy).
|-- .env                  Real secrets/config (git-ignored).
|-- README.md             *** STALE ***: describes the old Milvus tool-agent.
|-- chunks.json           1871 statute/rules/amendment chunks (ingest output).
|-- judgement_chunks.json 1199 judgment chunks (ingest output).
|
|-- assets/styles.css     Custom CSS for the chat UI (light theme).
|-- .streamlit/config.toml  Streamlit theme + server settings.
|
|-- retrieval/            Runtime retrieval primitives (in-process library).
|   |-- qdrant_store.py   QdrantHybridStore: multi-collection hybrid search
|   |                     (server-side RRF), exact_lookup, get_by_chunk_id,
|   |                     get_children, filter builder, point_id().
|   |-- query_encoder.py  BgeM3QueryEncoder: text -> {"dense","sparse"}.
|   |-- reranker.py       CrossEncoderReranker: query x evidence -> reranked.
|   `-- pipeline.py       retrieve_and_rerank() - the "fast path".
|
|-- retrieval_graph/      LangGraph multi-hop retriever.
|   |-- state.py          RetrievalState (TypedDict) + fresh_state().
|   |-- nodes.py          make_retrieval_nodes(...): all node callables.
|   |-- graph.py          build_retrieval_graph(...): compiles the StateGraph.
|   |-- smoke_test_*.py   Development smoke tests.
|
|-- Graph/                Knowledge-graph build (offline) + runtime client.
|   |-- profile.py        STEP 1 data profiling -> schema_report.json/SCHEMA.md.
|   |-- ontology.py       STEP 2 ontology spec (9 nodes, 9 rels) -> ontology.json.
|   |-- extract.py        STEP 3 extraction (Tier 0/1/2) -> graph_extracted.json.
|   |-- build.py          STEP 4 assembly (uuid5 ids) -> nodes.jsonl/edges.jsonl.
|   |-- neo4j_store.py    STEP 5 create schema + load/verify into Neo4j.
|   |-- query.py          KnowledgeGraphClient (Cypher) - the RUNTIME bridge.
|   |-- coverage.py, smoke_test_query.py        tooling/tests.
|   |-- *.jsonl/json/md   Build artifacts (nodes, edges, schema, summary).
|
|-- ingestion/            Offline PDF -> chunk pipeline.
|   |-- chunking.py         statutes/rules/amendment/order/paragraph -> chunks.json.
|   |-- Judgement_chunker.py judgment profile -> judgement_chunks.json.
|   `-- embedder.py         LEGACY SentenceTransformer embedder (Milvus era;
|                           NOT used by the current Qdrant pipeline).
|
|-- qdrant-store/         Offline embed -> Qdrant + data artifacts.
|   |-- ingest_hybrid.py      create legal_hybrid (dense+sparse) + upsert.
|   |-- ingest_judgements.py  create judgements collection + upsert.
|   |-- ingest_to_qdrant.py   older dense-only ingest (legacy).
|   |-- inspect_qdrant.py     diagnostics.
|   |-- embeddings_output.json / sparse_embedded_chunks.json /
|       judgement_embeddings.json   (BGE-M3 vectors for both corpora).
|
|-- evaluation/           Retrieval evaluation (no LLM).
|   |-- recall5.py          GOLD (8 queries) + Recall@5 of the fast path.
|   |-- kg_eval.py          A/B: baseline vs KG-augmented recall.
|   `-- extraction_audit.py KG edge grounding audit.
|
|-- data/                 Source PDFs (statutes + judgments).
`-- qdrant_storage/       Local Qdrant server files (not code).

RUNTIME IMPORT SPINE (who imports what, actual)
-----------------------------------------------
ui.py            -> api, streamlit
api.py           -> llm_client, retrieval.query_encoder, retrieval.qdrant_store,
                    retrieval.reranker, retrieval_graph.graph, retrieval_graph.state,
                    Graph.query, main_query (make_llm_planner + constants)
main_query.py    -> llm_client, retrieval.query_encoder/.qdrant_store/.reranker,
                    retrieval_graph.graph/.state, Graph.query
retrieval_graph/graph.py -> langgraph.graph, .state, .nodes
retrieval_graph/nodes.py -> .state
retrieval/qdrant_store.py -> qdrant_client (+models), os, uuid, sys
retrieval/query_encoder.py -> FlagEmbedding.BGEM3FlagModel, qdrant_client.models, torch
retrieval/reranker.py -> FlagEmbedding.FlagReranker | sentence_transformers.CrossEncoder
Graph/query.py      -> Graph.neo4j_store.get_config, Graph.ontology, neo4j
Graph/neo4j_store.py -> Graph.ontology
"""

# ==========================================================================
# 3. ENVIRONMENT VARIABLES AND CONFIGURATION
# ==========================================================================
SECTION_3 = r"""
3. ENVIRONMENT VARIABLES AND CONFIGURATION
==========================================
There is NO config.py and NO python-dotenv. Two tiny hand-written loaders:
  * llm_client._load_env()          -> never overrides an OS-set variable
  * Graph/neo4j_store._load_env()   -> DOES override ("the .env is the source
    of truth", so stale OS vars cannot shadow AuraDB credentials)

Variables the CURRENT code actually reads:

[LLM  - llm_client.py]
LLM_BASE_URL   gateway base (OpenRouter / Groq / OpenAI-compatible).
               If empty AND OPENAI_API_KEY is set -> falls back to OpenAI.
LLM_API_KEY    sent as "Authorization: Bearer <key>".
LLM_MODEL      e.g. google/gemma-4-31b-it:free, openai/gpt-oss-20b.
               If missing -> no LLM -> pipeline degrades to evidence-only.
LLM_TIMEOUT    request timeout (default 120 s).
LLM_MAX_RETRIES / LLM_BACKOFF_BASE  429/5xx retry policy (defaults 5 / 1.5).
OPENAI_API_KEY / OPENAI_MODEL       fallback path (default model gpt-4o-mini).

[Qdrant  - retrieval/qdrant_store.py]
QDRANT_HOST (localhost) / QDRANT_PORT (6333).
QDRANT_COLLECTIONS (default "legal_hybrid,judgements")  searched on every
query; missing collections are skipped at init. QDRANT_COLLECTION honoured as
a single-collection fallback.

[Models]
BGE_MODEL (BAAI/bge-m3)               -> query_encoder DEFAULT_MODEL.
RERANKER_MODEL (BAAI/bge-reranker-v2-m3) -> reranker DEFAULT_RERANKER.
RERANKER_BACKEND auto|bge|fast        -> "auto" picks bge on CUDA else "fast"
  (fast = cross-encoder/ms-marco-MiniLM-L-6-v2, ~0.05 s/pair vs ~3.5 s/pair).
HF_HUB_DISABLE_SYMLINKS=1             -> set before FlagEmbedding import
  (Windows: HF Hub symlinks fail with WinError 1314 without Dev Mode).

[Neo4j  - Graph/neo4j_store.py + Graph/query.py]
NEO4J_URI       e.g. neo4j+s://<instance>.databases.neo4j.io (AuraDB).
NEO4J_USERNAME  default "neo4j".
NEO4J_PASSWORD  required; never printed (the --diag command masks it).
NEO4J_DATABASE  default "neo4j".
Missing URI/password -> KnowledgeGraphClient raises a clear RuntimeError;
api.py catches it and sets kg=None so graph expansion is disabled but the
vector path still works.

[Dials that are actually used, not just present]
  * main_query.py: DEFAULT_TOP_K=20, DEFAULT_RERANK_K=6, DEFAULT_MAX_HOPS=2.
  * nodes.py defaults: top_k=20, rerank_top_k=10, confidence_threshold=0.4,
    kg_hops=1, kg_max_nodes=12 (callers pass these in build_retrieval_graph).
  * qdrant_store.py: DEFAULT_PREFETCH_LIMIT=50 per modality; Fusion.RRF.
  * chunking: min_chunk_tokens=20, split_threshold=500, leaf cap 600,
    parent cap 2000, min_structure_coverage 0.30.

[LEGACY vars in .env.example NOT read by any current module]
MLFLOW_*, MILVUS_*, EMBEDDER_BACKEND/BGE_M3_MODEL, SCOPE, RETRIEVAL_TOP_K,
HYBRID_CANDIDATE_LIMIT, RRF_K, MAX_HOPS, MAX_FALLBACK_RETRIES,
MAX_TOOL_CALLS_TOTAL, GRAPH_LITE_MAX_EXPANSIONS, CONFIDENCE_THRESHOLD,
RISK_THRESHOLD, DATA_DIR/STATUTES_DIR/JUDGMENTS_DIR/STORAGE_DIR, PDF_MIN_*.
Leftovers from the old Milvus tool-agent. Harmless; not read by current code.

Security note: all real values live only in the git-ignored .env. No secret
value appears anywhere in this walkthrough.
"""

# ==========================================================================
# 4. APPLICATION STARTUP AND INITIALIZATION
# ==========================================================================
SECTION_4 = r"""
4. APPLICATION STARTUP AND INITIALIZATION
=========================================
Heavy singletons (GB-scale models + connections) are built ONCE and cached.
The three entry points share the same lazy builder, api.ensure_resources().

STARTUP TRACE (ui.py, the primary demo entry):
1. ui.py -> st.set_page_config(...), inject assets/styles.css.
2. import api  -> imports fastapi + retrieval + graph modules. CHEAP: no model
   is loaded at import time.
3. Session starts; st.session_state["backend"] is None, so ui renders the
   "Preparing the legal research engine..." splash and calls the cached
   _warm_backend() (decorated @st.cache_resource).
4. _warm_backend() -> api.ensure_resources(), which lazily builds, in order:
     a. BgeM3QueryEncoder()                 # loads BAAI/bge-m3
     b. QdrantHybridStore(query_encoder=encoder)  # connects, keeps existing
                                                  # collections only
     c. CrossEncoderReranker()              # backend-aware model load
     d. KnowledgeGraphClient()              # Neo4j; ANY failure -> kg=None
     e. build_retrieval_graph(store, encoder, reranker,
            planner=make_llm_planner() if llm_client.llm_reachable() else None,
            max_hops=DEFAULT_MAX_HOPS, kg=kg, kg_hops=1, kg_max_nodes=12)
        # compiles the LangGraph StateGraph (no execution yet)
   Returns status dict {"models","store","kg","llm_reachable"}.
5. ui.py re-runs -> top bar shows a green "Ready" pill + the welcome screen.
6. The app waits for a chat message. Later queries reuse the cached objects.

api.py equivalent: the same steps run inside the FastAPI lifespan
(@asynccontextmanager lifespan) -> uvicorn logs "[api] warming up resources..."
then "[api] ready ...". GET /health reflects the status dict.

Startup failure behaviour (actual):
  * Neo4j down -> kg=None, message printed, app continues.
  * Qdrant / models fail -> ensure_resources() raises -> the UI shows a clean
    "The engine is not available" card (no raw traceback); /health reports
    {"models":"down","store":"down","kg":"down"}.
  * LLM unreachable (llm_reachable()==False) -> planner=None (single-hop
    deterministic expansion only) and answer synthesis is skipped gracefully.

What gets initialised vs what is NOT:
  * Nothing is ingested at startup; ingestion is a separate offline pipeline.
  * No Neo4j schema creation at startup (it is done by Graph/neo4j_store.py
    --load during the build step, not per request).
"""

# ==========================================================================
# 5. DATA INGESTION -> PREPROCESSING -> CHUNKING -> METADATA
# ==========================================================================
SECTION_5 = r"""
5. DATA INGESTION -> PREPROCESSING -> CHUNKING -> METADATA
==========================================================
Ingestion is OFFLINE (run once per corpus change). Two chunkers share one
design: profile detection ladder + structure-aware chunking + a normalisation
pass, then embedding + load to Qdrant, then the graph build (Section 8).

WHERE DOCUMENTS COME FROM
-------------------------
* data/ root: 15 statute-family PDFs (PPC 1860, CrPC 1898, Qanun-e-Shahadat
  1984, AML 2010, PECA 2016, PTA (Reorg) 1996, Payment Systems 2007, ETO
  2002, Investigation for Fair Trial 2013, gazette rules/amendments, IHC
  order sheet, ...).
* data/Judgements*: 82 judgment PDFs (mostly cyber-crime / bail).

TEXT EXTRACTION (pypdf, AES-aware)
----------------------------------
ingestion/chunking.py extracts pages and runs quality heuristics: docs that
are too short / garbled are flagged needs_ocr and skipped (no OCR engine is
present). OCR-free guarantee: 4 scanned PDFs are known to be excluded.

PROFILE DETECTION LADDER (order matters)
----------------------------------------
  judgement (checked first - its numbered paragraphs would otherwise look
             like statute sections)
  order     -> single chunk            (IHC order sheet)
  amendment -> "Amendment of section N of Act X" blocks + meta.amends_section
  rules     -> "Rule N" numbered items (gazette S.R.O. notifications)
  statute   -> Part -> Chapter -> Section (+ sub/clause)
  paragraph -> layout fallback (OCR-garbled PDFs) - no document ever fails

CHUNKING TUNING (ChunkingConfig, sized from a real audit of section-body
tokens: PPC median 85 / max 3348; CrPC max 24782):
  min_chunk_tokens = 20            # below => merge tiny heading stubs
  section_split_threshold_tokens = 500   # above => split into leaf children
  max_chunk_tokens = 600           # leaf ceiling
  parent_max_tokens = 2000         # small-to-big parent cap
  min_structure_coverage = 0.30    # <30% structured -> paragraph fallback

NORMALISATION PASS (every profile)
  1. skip the CONTENTS table (duplicates every heading)
  2. merge tiny stubs into a neighbour
  3. split giants into leaf children (children keep parent_id; the full unit
     is kept as an is_parent=True parent chunk = small-to-big)
  4. emit parent + children

JUDGMENT CHUNKING (ingestion/Judgement_chunker.py) - content-based, emits in
document order: preamble (court/case/parties, meta["keep"]), headnote
(meta["is_headnote"]), numbered paras. Every judgment leaf gets a provenance
prefix baked into its text:
  [Case: Sheraz Khan vs The State | Lahore High Court |
   Crl.Misc.No.44216-B/2021] [Para 5]
so each embedding knows its case/court. Document metadata (court, case
numbers, parties, judges, decided date, cited statutes, disposition) is
extracted per document; per-chunk disposition hits are meta["disposition"].

WHAT A CHUNK LOOKS LIKE (real shape)
------------------------------------
  {
    "chunk_id": "Pakistan Penal Code:section:272",   # SEQUENTIAL INDEX
    "doc": "Pakistan Penal Code.pdf",
    "doc_type": "statute", "level": "section",
    "section_no": "302",                    # REAL section - authoritative
    "section_title": "Punishment of qatl-i-amd",
    "part": null, "chapter": null,
    "parent_id": null, "is_parent": false,
    "tokens": 123, "text": "Whoever causes death ...", "meta": {...}
  }
CRITICAL QUIRK (verified at scale): chunk_id is a SEQUENTIAL index, NOT the
real section number - they mismatch on 95.6% of section chunks. Everything
(graph keys, exact lookups, the ontology) uses payload section_no, never the
chunk_id integer.

OUTPUTS (observed)
------------------
chunks.json: 1871 records / 15 docs.
judgement_chunks.json: 1199 records / 82 docs.

NEXT STEP: embed (Section 6) -> load to Qdrant -> graph build (Section 8).
"""

# ==========================================================================
# 6. EMBEDDINGS AND VECTOR DATABASE / RETRIEVAL
# ==========================================================================
SECTION_6 = r"""
6. EMBEDDINGS AND VECTOR DATABASE / RETRIEVAL
=============================================

6A. EMBEDDING MODEL  (retrieval/query_encoder.py)
-------------------------------------------------
Class: BgeM3QueryEncoder  (callable; wraps FlagEmbedding BGEM3FlagModel)
Model: BAAI/bge-m3  (BGE_MODEL env)
What it produces per text, in ONE call:
  dense  : normalized 1024-d vector (unnamed default vector in Qdrant)
  sparse : "lexical_weights" (BGE-M3 learned / SPLADE-style) -> qm.SparseVector
Settings: normalize_embeddings=True (matches stored cosine vectors); fp16 only
on CUDA; HF_HUB_DISABLE_SYMLINKS set before import (Windows fix).
encode(text) -> {"dense": [...], "sparse": SparseVector}
The dimensionality is sanity-checked against hidden_size == 1024 (a stored
collection made by a different model would raise loudly).
Because the stored vectors were produced by the SAME model, query and index
share one vector space with no external vocabulary to persist.
OFFLINE/LEGACY: ingestion/embedder.py is a Milvus-era SentenceTransformer
embedder; it is NOT used by the current Qdrant pipeline (the collections were
built from the pre-computed BGE-M3 files in qdrant-store/).

6B. VECTOR DATABASE  (retrieval/qdrant_store.py)
------------------------------------------------
Class: QdrantHybridStore. Collections: QDRANT_COLLECTIONS (default
"legal_hybrid,judgements"). Missing collections are skipped at init.
Each collection layout (created up-front by qdrant-store/ingest_*.py):
  vectors_config        = unnamed dense "" 1024-d COSINE
  sparse_vectors_config = {"sparse": SparseVectorParams(index on disk)}
  (cannot add sparse to an existing dense collection on qdrant-client 1.19 <->
   server 1.18.1 - a silent no-op - hence create-with-both.)
Payload = the 13 chunk fields (chunk_id, doc, doc_type, level, section_no,
section_title, part, chapter, parent_id, is_parent, tokens, text, meta).
Indexed payload fields: doc, doc_type, level, chunk_id, parent_id (keyword),
is_parent (bool). Point id = uuid5(DNS, chunk_id)  ==  point_id().

Main search flow (search()):
  User query string -> query_encoder -> {"dense","sparse"}
  -> per collection: Prefetch(dense, using=None, limit=50, filter)
                      + Prefetch(sparse, using="sparse", limit=50, filter)
     combined with qm.FusionQuery(fusion=qm.Fusion.RRF), top_k=20
  -> Qdrant fuses with Reciprocal Rank Fusion (sum 1/(k+rank)) server-side
  -> merge across collections, dedupe by chunk_id, tag ev["collection"],
     sort by fused score desc.
Filters: dict {field: value} -> MatchValue; list -> MatchAny; qm.Filter
accepted; _build_filter DROPS null/empty values (LLM planners emit
{"field": null}) so it can never crash on a null MatchValue.

Other store methods used at runtime:
  exact_lookup(doc, section_no)  -> payload-filter scroll (deterministic)
  get_by_chunk_id(chunk_id)      -> fetch one chunk by uuid5 id (any coll)
  get_children(parent_chunk_id)  -> child chunks (small-to-big hop)
Evidence record shape: {chunk_id, doc, doc_type, level, section_no,
section_title, part, chapter, parent_id, is_parent, tokens, text, meta,
score, point_id, domain} where domain = "judgment" if doc_type in
("judgment","judgement") else "statute".

6C. CROSS-ENCODER RERANKER  (retrieval/reranker.py)
---------------------------------------------------
Class: CrossEncoderReranker.
Backend by RERANKER_BACKEND: auto (bge on GPU else fast) | bge | fast.
  bge  = FlagReranker(BAAI/bge-reranker-v2-m3)   accurate, ~3.5 s/pair CPU
  fast = CrossEncoder(cross-encoder/ms-marco-MiniLM-L-6-v2) ~0.05 s/pair CPU
score(query, texts) -> list[float] (compute_score, normalize=True).
rerank(query, evidence, top_k) -> copies each record, adds rerank_score,
sorts descending, truncates. Input records are NOT mutated.

6D. FAST PATH  (retrieval/pipeline.py)
--------------------------------------
retrieve_and_rerank(store, reranker, query, retrieve_top_k=20,
rerank_top_k=8): store.search(...) then reranker.rerank(...). This is the
non-graph "fast path" used by main_query.py when NOT --multi-hop. The default
demo path instead uses the LangGraph graph (Section 9/10).

RETRIEVAL PIPELINE DIAGRAM
--------------------------
 User query string
      |  BgeM3QueryEncoder.encode()
      v
 dense(1024) + sparse  -> QdrantHybridStore.search()
      |  per collection: dense-prefetch + sparse-prefetch
      |  Qdrant RRF fusion -> top_k=20 -> merge/dedupe across collections
      v
 evidence pool (score=RRF)
      |  CrossEncoderReranker.rerank(query, pool)
      v
 reranked evidence (rerank_score) -> LLM / next hop
"""

# ==========================================================================
# 7. GRAPH / GRAPH-LITE ARCHITECTURE (what actually exists)
# ==========================================================================
SECTION_7 = r"""
7. GRAPH / GRAPH-LITE ARCHITECTURE
==================================
IMPORTANT ACCURACY POINT: this repository does NOT contain a "Graph-Lite"
JSON-adjacency module any more. The old repo notes and README describe a
Graph-Lite adjacency map and a graph_lite retrieval node, but no such file
exists in the current tree. The graph component that DOES exist is a real
Neo4j knowledge graph (Section 8) plus a deterministic parent-child
"small-to-big" expansion inside the retriever (Section 10).

What the graph gives you that plain vector search cannot:
  * STRUCTURE  : Act -> Part -> Chapter -> Section containment (CONTAINS),
                 chunk -> section membership (BELONGS_TO).
  * CITATION   : section -> section references (REFERENCES) and judgment ->
                 section cites (CITES) - i.e. "which other provisions did
                 this one point to, and which cases relied on it".
  * TRAVERSAL  : from a retrieved chunk, follow BELONGS_TO to its Section,
                 then REFERENCES to neighbours -> related provisions that may
                 not be lexically similar but are legally connected.

The three retrieval signals, clarified:
  Vector RAG      : semantic + lexical similarity over chunk text (Qdrant).
  Multi-hop/expand: same vector store, but the retriever can query again with
                    a refined query, or expand parents/children (parent_id).
  Graph (Neo4j)   : explicit legal relationships; a separate retrieval arm
                    that feeds graph-discovered provisions/case law into the
                    same evidence pool (tagged _kg).
"""

# ==========================================================================
# 8. NEO4j INTEGRATION (knowledge graph) - the actual graph
# ==========================================================================
SECTION_8 = r"""
8. NEO4j INTEGRATION
====================

8A. THE 5-STEP BUILD (offline, all in Graph/)
----------------------------------------------
STEP 1 profile.py    : loads chunks.json + judgement_chunks.json, writes
                       schema_report.json + SCHEMA.md (field types, coverage,
                       quirks). Key finding: section key = (act, section_no).
STEP 2 ontology.py   : DECLARATIVE spec (this is the schema contract).
  NODES (9 labels): Act, Part, Chapter, Section, Chunk, Judgment, Court,
                    Party, Offence.
  Unique keys: Section (act, section_no); Chunk (chunk_id); Judgment
               (case_no); Act (name); Part/Chapter (act, name); Court/Party/
               Offence (name).
  RELS (9 types): CONTAINS (Act->Part/Chapter->Section), PARENT_OF (Chunk->
                  Chunk), BELONGS_TO (Chunk->Section), AMENDED_BY (Section->
                  Act), REFERENCES (Section->Section/Act), CITES (Judgment->
                  Section), DECIDED_BY (Judgment->Court), INVOLVES (Judgment->
                  Party), CONCERNS (Judgment->Offence).
  ACT_ALIASES maps raw PDF filenames -> canonical act names (dedupe), e.g.
  "Pakistan Penal Code.pdf" -> "Pakistan Penal Code 1860".
  Tier tags: 0 = deterministic from metadata, 1 = regex, 2 = LLM.
STEP 3 extract.py    : entity/relationship extraction -> graph_extracted.json.
  Tier 0 (offline): Act/Part/Chapter/Section/Chunk/Judgment/Court-seed/
    Party-seed nodes; CONTAINS/PARENT_OF/BELONGS_TO; AMENDED_BY from
    meta.amends_section + "Amendment of section N" regex; DECIDED_BY/INVOLVES
    seeded from a deterministic case-header parser (meta.context
    "[Case: parties | Court | Case No]") + meta.disposition.
  Tier 1 (offline): REFERENCES via regex (resolve_act_reference over
    ACT_REFERENCE_ALIASES + section-number regex). Only references that
    resolve to existing sections/acts become edges; the rest are counted and
    skipped (no dangling targets).
  Tier 2 (optional, needs LLM + --tier2): judgment enrichment (CITES,
    CONCERNS, full DECIDED_BY/INVOLVES, citation/date) via
    llm_client.llm_chat_json. As loaded, only 3 of 82 judgments were
    enriched (so CITES is sparse: 22 edges).
STEP 4 build.py      : assembly -> nodes.jsonl / edges.jsonl /
                       assembly_summary.json.
  Stable IDs via uuid5(NAMESPACE_DNS, ...):
     Chunk nodes: id = uuid5(DNS, chunk_id)  ==  the Qdrant point id (1:1
                  join between graph and vector store).
     Others     : id = uuid5(DNS, "graph:<Label>:<key...>").
  Validates duplicate ids and dangling edges (fails unless --allow-dangling).
  --drop-section-text removes Section.text (the Chunk node already has it).
STEP 5 neo4j_store.py: load into Neo4j (AuraDB cloud).
  connect() verifies connectivity eagerly and gives a friendly fix message.
  schema_statements() -> one unique constraint per label id + ontology key +
    an index on Chunk.point_id.
  load_nodes/load_edges: batched (500) UNWIND ... MERGE ... ON CREATE SET
    (idempotent). reset() detach-deletes all ontology labels.
  --check / --diag / --verify / --load. --diag prints the password LENGTH and
    character-class booleans, NEVER the value.

OBSERVED LOAD (snapshot, from the loaded graph):
  nodes: Act 15, Chapter 105, Chunk 3070, Court 9, Judgment 75, Offence 8,
         Part 13, Party 79, Section 1507  (total 4881)
  edges: CONTAINS 3660, BELONGS_TO 1735, REFERENCES 1062, PARENT_OF 437,
         DECIDED_BY 66, INVOLVES 120, CITES 22, CONCERNS 9, AMENDED_BY 2
         (total 7116)

8B. THE RUNTIME CLIENT  (Graph/query.py)
----------------------------------------
Class: KnowledgeGraphClient - a thin Cypher client used by the retriever.
What/Why: make Neo4j readable by the RAG pipeline - expand Qdrant hits into
the graph, discover related provisions and citing case law.
Constructor: reads config via Graph.neo4j_store.get_config() (which loads
.env), creates neo4j.GraphDatabase.driver(uri, auth=(user, password)). Raises
a clear RuntimeError if not configured (api.py catches -> kg=None).
Methods (actual Cypher):
  sections_for_chunks(chunk_ids, limit=25)
      MATCH (c:Chunk) WHERE c.chunk_id IN $ids
      MATCH (c)-[:BELONGS_TO]->(s:Section) RETURN s...
      -> Qdrant -> graph JOIN (the retrieval bridge).
  get_section(act, section_no) -> exact Section lookup.
  expand_sections(seed_ids, hops, max_nodes) -> BFS over undirected
      REFERENCES, bounded by hops and max_nodes.
  judgments_citing_section(section_id) -> MATCH (s)<-[:CITES]-(j:Judgment).
  expand_evidence(chunk_ids, hops, max_nodes) -> ORCHESTRATOR returning
      {"seed_sections": [...], "sections": [...], "judgments": [...]}.
  stats() / canonical_acts() for diagnostics.
Record shaping: returned records are shaped like RAG evidence (doc_type,
level, section_no, section_title, chunk_id) plus kg_id and source ("kg" for
sections, "kg:CITES" for judgments). Section.text is None - full text lives
in Qdrant (that is why the retriever re-fetches it by chunk_id).

What calls it: retrieval_graph/nodes.py kg_expand node (Section 10) and
evaluation/kg_eval.py. It is created once in api.ensure_resources() and in
main_query._get_resources().
"""

# ==========================================================================
# 9. LANGGRAPH STATE, NODES, EDGES, ROUTER, CONDITIONAL ROUTING, CYCLES
# ==========================================================================
SECTION_9 = r"""
9. LANGGRAPH ARCHITECTURE
=========================
Builder: retrieval_graph/graph.py -> build_retrieval_graph(store, encoder,
reranker, planner=None, max_hops=3, top_k=20, rerank_top_k=10, kg=None,
kg_hops=1, kg_max_nodes=12). If the store has no query_encoder and one is
passed, it is wired in. Returns graph.compile().

STATE SCHEMA  (retrieval_graph/state.py, RetrievalState TypedDict)
------------------------------------------------------------------
  original_query, current_query, filters
  evidence_pool (deduped across hops), hop_evidence (latest hop)
  current_hop, max_hops, last_score, low_confidence,
  expansion_done, last_expansion_added, decision
  next_query, next_filters, planner_note
  kg_evidence            (graph-discovered provisions + CITES judgments)
  final_evidence         (output)
fresh_state(question, max_hops, filters) returns a clean dict for each run -
the state is per-question; there is no cross-turn LangGraph memory (that
lives in the frontend / api, Section 13).

NODES (all created by make_retrieval_nodes in nodes.py) - What each does:
-----------------------------------------------------------------------
initialize  : reset every runtime field (idempotent start).
router      : DETERMINISTIC stub - sets decision="retrieve". (An LLM intent
              router is NOT implemented; see note below.)
retrieve    : current_hop += 1; store.search(current_query, filters,
              top_k=20) -> hop_evidence; records into evidence_pool;
              last_score = top hit score; low_confidence if no results.
rerank      : reranker.rerank(current_query, hop_evidence, top_k=10);
              last_score = top rerank_score;
              low_confidence = empty OR top rerank_score < 0.4.
expand      : decision="retrieve"; expansion_done=True.
              IF planner is set (LLM reachable):
                 planner(original_query, evidence_pool, hop) ->
                 {"next_query","filters","note"} via llm_chat_json;
                 current_query=next_query; filters cleaned by _clean_filters;
                 last_expansion_added=True.
              ELSE (deterministic parent-child, small-to-big):
                 for each pooled chunk: if parent_id -> fetch parent via
                 get_by_chunk_id; if is_parent -> fetch children via
                 get_children; add (dedup) to pool.
                 Then current_query=original_query, filters=None (broaden).
kg_expand   : ONLY when kg is not None. Maps pool chunk_ids -> Sections
              (BELONGS_TO), walks REFERENCES (kg_hops, bounded kg_max_nodes)
              via kg.expand_evidence(...); for discovered sections it fetches
              FULL TEXT back from Qdrant (get_by_chunk_id) and adds those to
              the rerankable pool tagged _kg/_kg_source="REFERENCES";
              text-less sections -> kg_evidence; CITES judgments ->
              kg_evidence tagged _kg_source="CITES". Never raises: KG errors
              are appended to planner_note and execution continues.
fallback_retrieve : drops filters, resets to original_query (broaden recall).
finalize    : rerank the WHOLE evidence_pool against original_query (top 20)
              -> final_evidence.

CONDITIONAL ROUTERS (conditional edges - actual logic)
------------------------------------------------------
route_decision (after "router"):
    if current_hop >= max_hops -> "sufficient" (finalize)
    else decision ("retrieve"|"expand")
check_quality (after "rerank"):
    current_hop >= max_hops            -> "hop_limit"
    low_confidence                     -> "poor"
    planner is not None                -> "expand"   (LLM-driven hop)
    not expansion_done                 -> "expand"   (run expansion once)
    last_expansion_added               -> "expand"   (still adding evidence)
    else                               -> "good"

GRAPH EDGES (graph.py)
----------------------
  initialize -> router
  router     -conditional-> retrieve | expand | (sufficient) finalize
  retrieve   -> rerank
  rerank     -conditional-> good:   kg_expand if kg else finalize
                            poor:   fallback_retrieve
                            expand: expand
                            hop_limit: kg_expand if kg else finalize
  fallback_retrieve -> retrieve       (cycle back to a broader search)
  expand     -> retrieve              (cycle back for the next hop)
  kg_expand  -> finalize   (when kg present)
  finalize   -> END

CYCLES AND BOUNDING
-------------------
* The poor->fallback_retrieve->retrieve and expand->retrieve edges form
  cycles. They are bounded by max_hops (DEFAULT_MAX_HOPS = 2 in
  main_query.py/api.py; the graph default is 3). route_decision and
  check_quality both short-circuit to "hop_limit"/"sufficient" when
  current_hop >= max_hops, so no infinite loop is possible.
* check_quality's "expand" triggers are also gated by the same hop limit.
* kg_expand runs at most once per question (it sits after rerank, only on the
  good/hop_limit path), bounded by kg_hops=1 and kg_max_nodes=12.

ROUTER / DECISION MAKING - what ACTUALLY exists:
The node is named router but currently only sets decision="retrieve"; the
comment in code says "Optional LLM intent router goes here later". There is
no multi-intent classification, no LLM tool-selection, and NO query
decomposition module in the current code. The real "decisions" are the two
deterministic conditional functions above (route_decision, check_quality)
plus the optional LLM hop planner inside expand.
"""

# ==========================================================================
# 10. QUERY DECOMPOSITION AND MULTI-HOP RETRIEVAL
# ==========================================================================
SECTION_10 = r"""
10. QUERY DECOMPOSITION AND MULTI-HOP RETRIEVAL
===============================================
QUERY DECOMPOSITION: NOT IMPLEMENTED as a module. Complex multi-part
questions are not split into sub-queries by the code. (The only "refinement"
of a question happens through (a) the optional hop planner that proposes the
NEXT retrieval query/filters, and (b) the frontend/backend conversation
contextualizer in api.py - Section 13.)

MULTI-HOP RETRIEVAL (the real mechanism)
----------------------------------------
A "hop" = one retrieve+rerank pass against Qdrant (current_hop += 1 in the
retrieve node).

Why multiple hops: the first pass may be too narrow (needs a broader or
refined query), too weak (low rerank confidence), or legally connected
evidence is only reachable by expanding from what was found (parent-child or
graph REFERENCES).

How the system decides what to do next (per hop, after rerank):
  * low_confidence (empty results, or top rerank_score < 0.4):
        fallback_retrieve -> drop filters, original query, retrieve again.
  * planner available (LLM): check_quality returns "expand"; the planner
        proposes a refined next_query/filters for the next retrieve.
  * no planner: deterministic expand once, then keep expanding only while
        parent-child expansion is still adding NEW evidence
        (last_expansion_added), otherwise "good" -> finish.
  * hop budget: when current_hop >= max_hops everything routes to finalize.

What each hop changes in state:
  current_hop +1, hop_evidence (new), evidence_pool grows (dedup by
  chunk_id), current_query/filters possibly refined, last_score and
  low_confidence recomputed, planner_note filled.

Knowledge-graph hop (kg_expand) - the "graph hop":
  runs AFTER a good/hop-limit rerank when kg is present. It does not issue a
  new vector query; it maps the pooled chunk_ids to Section nodes in Neo4j
  and walks REFERENCES once (bounded). Discovered sections get their full
  text re-fetched from Qdrant and enter the same pool (tagged _kg,
  rerankable). Graph-only findings (text-less sections + CITES judgments)
  land in state["kg_evidence"] and are surfaced in the answer/sources.

When does it STOP (termination):
  1. current_hop reaches max_hops  -> hop_limit -> (kg_expand) -> finalize.
  2. good quality + no more useful expansion -> finalize.
  3. either way, finalize reranks the combined pool once -> final_evidence,
     and the graph ends.

DIAGRAM (one question, KG enabled)
----------------------------------
 original_query
   -> initialize -> router(retrieve)
   -> retrieve(hop1) -> rerank -> check_quality
        | "poor"  -> fallback_retrieve -> retrieve(hop2) -> ...
        | "expand"-> expand (planner OR parent-child) -> retrieve(hopN)
        | "good"/"hop_limit" -> kg_expand (Neo4j) -> finalize -> END
        (hop_limit also enforced by route_decision before retrieve)
"""

# ==========================================================================
# 11. TOOL CALLING AND DECISION-MAKING (what actually exists)
# ==========================================================================
SECTION_11 = r"""
11. TOOL CALLING AND DECISION-MAKING
====================================
ACCURACY: the current codebase has NO function/tool-calling agent. There are
no tool schemas, no make_tools(), no web-search tool, no statute/penalty/
procedure/case-law lookup tools. Those existed in the OLD Milvus architecture
(per stale notes/README) but are NOT in this repository now.

What replaces "tool calling" in the current runtime:
  1. RETRIEVAL PRIMITIVES are plain method calls, not LLM-dispatched tools:
       QdrantHybridStore.search / exact_lookup / get_by_chunk_id /
       get_children   (retrieval/qdrant_store.py)
       CrossEncoderReranker.rerank
       KnowledgeGraphClient.expand_evidence
  2. DECISION-MAKING is deterministic (LangGraph conditional edges) plus one
     OPTIONAL LLM hop planner:
       make_llm_planner() in main_query.py is injected as planner into
       build_retrieval_graph(...) only when llm_client.llm_reachable().
       Called inside the expand node: planner(original_query, evidence_pool,
       hop) -> llm_chat_json -> {"next_query","filters","note"}.
       Its system prompt asks for JSON only: the NEXT retrieval query, an
       optional {"doc_type"/"doc"} filter, and a one-line note.
       Any planner failure falls back to deterministic parent-child expansion
       (planner_note records the error).
  3. There is no "agent loop" choosing between retriever/graph/tool per turn;
     the graph always: retrieve -> rerank -> decide -> (expand|fallback|
     kg_expand) -> finalize.

So the honest interview statement: this project is an AGENTIC-STYLE pipeline
(LangGraph state machine with conditional routing and bounded cycles) but it
does not do LLM tool-calling. If asked "does it call tools?", answer: "The
old design had deterministic legal tools; the current repo implements
retrieval as first-class graph operations instead."
"""

# ==========================================================================
# 12. PROMPT CONSTRUCTION AND LLM INTERACTION
# ==========================================================================
SECTION_12 = r"""
12. PROMPT CONSTRUCTION AND LLM INTERACTION
===========================================
LLM CLIENT (llm_client.py)
--------------------------
* llm_reachable() -> True only if base_url + model are configured (it does
  NOT ping the server).
* llm_chat(system, user, max_tokens=1200, temperature=0.2) -> assistant text
  (single OpenAI-compatible POST /chat/completions; retries 429/5xx with
  exponential backoff + Retry-After).
* llm_chat_json(...) -> llm_chat then json.loads; returns {} on ANY failure
  (the planner relies on this safe fallback).

PROMPT SITES (actual prompts, all built in api.py / main_query.py):

[1] HOP PLANNER - main_query.make_llm_planner() -> planner() in expand node
  System: "You are a Pakistani Cyber Crime legal retrieval planner. Decide
  the NEXT retrieval hop. Reply ONLY with JSON: {\"next_query\": ...,
  \"filters\": {\"doc_type\": ... | null, \"doc\": ... | null} | null,
  \"note\": ...}"
  User: original question + a sample of the evidence pool so far.
  Output parsed by llm_chat_json; filters cleaned by _clean_filters (nulls
  dropped); any failure -> deterministic expansion.

[2] ANSWER SYNTHESIS - api._synthesize(evidence, kg_evidence, query, history)
    (main_query.synthesize_answer is the equivalent without kg/history)
  System: "You are a Pakistani legal research assistant specialising in
  cyber-crime and criminal law. Answer using ONLY the provided evidence and
  cite the exact act + section number for every claim. Structure your answer
  with these markdown headings: ## Direct Answer / ## Legal Basis / ##
  Analysis / ## Practical Guidance / ## Caveats. Be direct and practical. Do
  NOT invent statutes, sections, or facts not in the evidence. If evidence is
  insufficient, say so and advise consulting a licensed lawyer. Respond in
  the same language as the question."
  User: "Question: {query}" + formatted evidence docs ([i] doc - Section N
  (title) [doc_type] + text snippet up to 1200 chars each, top 8) + optional
  "Related case law (from the knowledge graph)" lines (kg_evidence where
  _kg_source == CITES) + optional "Conversation so far ..." block.
  max_tokens=1500.

[3] CONVERSATION CONTEXTUALIZER - api._contextualize_query(query, history)
  System: rewrite a conversational follow-up into ONE standalone searchable
  question (resolve "this article", "that section", "it", "in that context"
  etc.), with two worked examples; reply with only the rewritten question.
  max_tokens=160, temperature=0.0. Runs only when history exists AND the LLM
  is reachable; any failure -> return the raw question.

FINAL PROMPT FLOW (what the LLM actually sees for an answer)
------------------------------------------------------------
  System instructions (headings + citation rule + honesty rule)
  + retrieved provisions text (evidence, top 8)
  + graph case-law summary (CITES judgments)   [when kg_evidence present]
  + conversation history block                 [when history present]
  + "Question: <current user question>"
        |
        v  llm_client.llm_chat  (httpx POST, retries, timeout)
        v
  structured markdown answer  ->  ui.py renders it as the assistant message

Hallucination control (as implemented):
  * "use ONLY the provided evidence" + "do NOT invent statutes/sections".
  * Every claim is asked to carry an exact act + section citation.
  * Sources panel shows the actual retrieved chunks for verification.
  * If the LLM is unreachable, the pipeline returns retrieved evidence with a
    clear note instead of a fabricated answer.
"""

# ==========================================================================
# 13. CONVERSATION MEMORY / HISTORY
# ==========================================================================
SECTION_13 = r"""
13. CONVERSATION MEMORY / HISTORY
=================================
WHERE HISTORY LIVES (frontend)
------------------------------
ui.py stores the full conversation in Streamlit session state:
  st.session_state["messages"] = [{"role","content","sources","case_law",
                                   "caption",...}, ...]
Each user/assistant turn is appended; the loop re-renders messages from this
list every rerun, so the conversation is visible and continuous.

HOW HISTORY REACHES THE BACKEND
-------------------------------
On a new user message, ui._submit builds:
  history = [{"role": m["role"], "content": m["content"]} for m in messages[:-1]]
and passes it to api.answer(question, use_llm=True, multi_hop=True,
history=history). The current question itself is NOT in history.

WHAT THE BACKEND DOES WITH IT
-----------------------------
api.answer():
  1. search_q = _contextualize_query(question, history or [])
     -> if history exists AND LLM reachable, an LLM rewrites the follow-up
        into a standalone question so RETRIEVAL searches the right thing
        (verified: "Does the same apply to its attempt?" after a qatl-i-amd
        answer is rewritten to mention "attempt ... under the Pakistan Penal
        Code (Section 302)").
  2. Retrieval runs on search_q.
  3. _synthesize(evidence, kg_evidence, question, history) includes a
     "Conversation so far ..." block so the LLM resolves "this article" /
     "that case" when writing the answer.

So history influences BOTH retrieval (via contextualization) AND the final
answer prompt. There is no separate long-term memory store; it is
per-session in Streamlit and sent per request.

LIMITATIONS (actual):
  * History is capped implicitly: only the last 6 turns are formatted into
    the prompt (_format_history limit=6).
  * The LangGraph state itself resets per question (fresh_state) - no
    cross-turn state in the graph.
  * Contextualization needs the LLM; if the LLM is down, follow-ups are sent
    verbatim (retrieval may be weaker) but the answer prompt still includes
    the conversation only when synthesis runs.
"""

# ==========================================================================
# 14. STREAMLIT FRONTEND -> BACKEND -> LANGGRAPH -> LLM -> RESPONSE FLOW
# ==========================================================================
SECTION_14 = r"""
14. STREAMLIT FRONTEND -> BACKEND -> LANGGRAPH -> LLM -> RESPONSE
=================================================================
FRONTEND (ui.py) - a professional ChatGPT-style chat, real features:
  * Layout: slim top bar (brand "Verus" + status pill Ready/Preparing/
    Degraded), centered ~820px content column, sidebar (New chat button +
    "Show sources" toggle), pinned chat input at the bottom.
  * Welcome/empty state: "How can I help you?" + 3 clickable suggestion
    chips. It is replaced by the conversation the moment a message arrives.
  * Messages: user = right-aligned soft bubble; assistant = full-width
    readable markdown (headings, tables, lists). Loading = "Thinking..."
    spinner while the backend runs.
  * Sources: each assistant message can open a "Sources · N" expander that
    groups citations into "Provisions" (statutes with section numbers) and
    "Case law" (judgment chunks + graph CITES), with a caption like
    "Deep research · 2 hops · 19.6s".
  * Warm-up: @st.cache_resource _warm_backend() loads everything once behind
    a clean splash; api.ensure_resources() returns a status dict.
  * Errors: backend failures are logged to the console; the user sees a
    clean "Sorry, I couldn't process that request. Please try again." message
    (no raw tracebacks).

FRONTEND -> BACKEND INVOCATION
------------------------------
ui.py imports api (same process - no HTTP between UI and logic). It calls:
  api.answer(question, use_llm=True, multi_hop=True, history=history)
which mirrors exactly what the FastAPI endpoint POST /query does. So the
Streamlit app works WITHOUT running the uvicorn server; if you DO run the
server, ui.py could be pointed at it but the current code calls it in-process.

BACKEND (api.py)
----------------
  QueryRequest {query, use_llm=True, multi_hop=True, history=[]}
  answer():
    1. _contextualize_query(question, history)
    2. multi_hop True -> _graph.invoke(fresh_state(search_q, max_hops=2))
       reads out.final_evidence, out.current_hop, out.kg_evidence
       multi_hop False -> store.search + reranker.rerank (fast path)
    3. llm_on = use_llm and llm_reachable(); if on ->
       _synthesize(evidence, kg_evidence, question, history) -> answer_text
    4. returns {query, mode, hops, llm_used, answer, evidence_count,
       evidence[...], kg_evidence_count, kg_evidence[...]}
  _ev_out truncates text to 300 chars and adds rerank_score/from_graph.
  _kg_out flattens graph records (source/act/section_no/case_no/court/
  outcome/date).

END-TO-END FLOW DIAGRAM (one message)
-------------------------------------
 user types question in ui.py
   -> st.session_state.messages += user turn
   -> history = previous turns
   -> ui._run_backend -> api.answer(question, history=history)
   -> (spinner "Thinking...")
   -> contextualize (optional LLM rewrite)
   -> LangGraph multi-hop over Qdrant (+ reranker, + optional kg_expand)
   -> evidence + kg_evidence
   -> LLM synthesis (api._synthesize) -> cited markdown
   -> api returns dict -> ui builds assistant message + sources
   -> ui renders markdown + "Sources · N" + caption; input stays available
"""

# ==========================================================================
# 15. ERROR HANDLING AND IMPORTANT IMPLEMENTATION DETAILS
# ==========================================================================
SECTION_15 = r"""
15. ERROR HANDLING AND IMPORTANT IMPLEMENTATION DETAILS
=======================================================
WHAT HAPPENS WHEN ... (all actual behaviour):

* LLM unreachable  -> planner=None (deterministic expansion only); answer is
  None; the UI shows the retrieved provisions with a clear "couldn't generate
  a written answer because the language model is unreachable" note. /health
  reports llm_reachable False.
* LLM synthesis fails mid-answer -> api sets answer_text to
  "(LLM synthesis failed: ...)" and the UI detects that prefix and replaces
  it with a graceful note + the sources. Console logs the real error.
* LLM planner emits bad/null filters -> _clean_filters() (nodes.py) and
  _build_filter() (qdrant_store.py) drop null/empty values; a null dict
  becomes None (no filter). This fixed a real Qdrant MatchValue(None) crash.
* Planner throws -> expand() catches, records planner_note, falls back to
  deterministic parent-child expansion.
* Neo4j down / unconfigured -> KnowledgeGraphClient raises; api/main_query
  catch it and set kg=None -> graph expansion disabled, vector path works.
  GET /health shows kg:"down".
* Qdrant down at startup -> ensure_resources raises -> clean "engine not
  available" UI card / status "down" (not a traceback).
* Empty query -> api.answer raises HTTPException 422 "query must be
  non-empty"; ui.py never sends empty strings (chat_input returns None).
* Empty retrieval -> retrieve() sets low_confidence=True -> fallback path;
  if nothing usable remains, _assistant_message marks the turn as an error
  and the UI shows the generic retry message.
* HTTP 429/5xx from the LLM gateway -> llm_client retries with exponential
  backoff + Retry-After (LLM_MAX_RETRIES=5, LLM_BACKOFF_BASE=1.5).
* Windows HF cache symlink failure -> HF_HUB_DISABLE_SYMLINKS=1 set before
  FlagEmbedding import in query_encoder.py and reranker.py.
* Chunk/collection version mismatch -> BgeM3QueryEncoder raises if the model
  hidden_size != 1024; ingest scripts raise on wrong dims or empty sparse.

IMPORTANT IMPLEMENTATION DETAILS (things that matter in an interview)
---------------------------------------------------------------------
* Chunk id vs section number: point ids / graph chunk nodes / retrieval all
  key on chunk_id = uuid5(DNS, chunk_id); the LEGAL identity is (act,
  section_no). The 95.6% mismatch between the chunk_id integer and the real
  section_no is why the ontology never keys on the chunk_id number.
* Qdrant point id == Neo4j Chunk.id == uuid5(DNS, chunk_id): this is the
  single join key between the vector store and the graph.
* Hybrid retrieval = server-side RRF fusion of dense + sparse prefetches;
  evidence is merged across the legal_hybrid and judgements collections and
  later re-ranked once by the cross-encoder (RRF order only shapes the pool).
* kg_expand never raises; KG is a best-effort enrichment arm, so one failing
  service cannot take down the query.
* Section.text was dropped at graph load; full text is re-fetched from Qdrant
  by chunk_id - a deliberate "metadata in graph, text in vector store" split.
* .env loaders: llm_client never overrides OS vars; neo4j_store DOES override
  (this fixed a real "stale OS credentials shadow .env" bug).
"""

# ==========================================================================
# 16. PERFORMANCE, SECURITY AND LIMITATIONS
# ==========================================================================
SECTION_16 = r"""
16. PERFORMANCE, SECURITY AND LIMITATIONS
=========================================
PERFORMANCE (actual design decisions)
-------------------------------------
* Heavy singletons (bge-m3 ~2.3 GB + reranker ~2.27 GB on CPU) are loaded ONCE
  per process and cached (@st.cache_resource / module globals). Startup warm
  ~15-60 s; subsequent multi-hop queries ~15-25 s each (measured), most of
  which is LLM + reranker on CPU.
* RERANKER_BACKEND=auto picks bge-reranker-v2-m3 on GPU and the tiny
  ms-marco MiniLM on CPU (~0.05 s/pair vs ~3.5 s/pair) - a precision/speed
  trade-off chosen because the LLM does final synthesis anyway.
* Hop budget max_hops=2 (DEFAULT_MAX_HOPS) bounds LLM planner calls;
  kg_expand bounded by kg_max_nodes=12 and runs at most once.
* Only top-8 evidence texts are fed to the LLM (each up to 1200 chars) to
  bound prompt size and cost.
* Multi-collection RRF moves fusion to the Qdrant server (one round trip per
  collection instead of client-side re-ranking of every hit).
* Bottlenecks: first-load model time, CPU reranker, per-hop LLM planner
  round-trips, free-tier gateway rate limits (the LLM client retries).

SECURITY
--------
Implemented:
  * Secrets live only in the git-ignored .env; nothing is hard-coded.
  * neo4j_store --diag never prints the password (length + char classes only).
  * The UI never renders raw tracebacks; errors are logged server-side.
  * No web-search tool exists, so there is no unvetted web content entering
    the context (also a limitation: no live/current info).
Recommended / not yet implemented (be honest):
  * No authentication on the FastAPI/Streamlit app.
  * No input sanitization beyond empty-string checks; prompt-injection
    resistance relies on the "use only evidence" system prompt.
  * No PII/anonymisation layer for case text.
  * LLM API keys are sent as Bearer tokens over the configured HTTPS gateway;
    a local/vulnerable gateway would be a risk.

LIMITATIONS (actual, don't hide them)
-------------------------------------
* The knowledge graph's case-law layer is sparse: only 3 of 82 judgments
  were LLM-enriched (CITES = 22 edges), so "which cases cite this section"
  answers are limited to those docs.
* Graph value on single-provision questions is coverage, not recall: the A/B
  eval (evaluation/kg_eval.py) showed baseline Recall@5 7/8 and KG-augmented
  7/8 - the KG added 48 related sections but rescued 0 queries on this gold
  set (the only miss is a PECA bail question whose judgment chunks have no
  section link).
* No OCR: 4 scanned PDFs are excluded; corpus is 15 statutes + 82 judgments
  as loaded.
* Fixed (small) corpus, no incremental/streaming ingestion.
* LLM answers are single-shot (no streaming, no self-critique loop in the
  current code - the old notes mention reflection but it is not in the tree).
* Retrieval evaluation uses a hand-built 8-query gold set; no large labelled
  eval set, no end-to-end (answer-level) evaluation harness.
* Conversation history is capped at the last ~6 turns and lives only in the
  Streamlit session (no persistence across restarts).
"""

# ==========================================================================
# 17. COMPLETE END-TO-END EXAMPLE (what happens when a user asks a question)
# ==========================================================================
SECTION_17 = r"""
17. COMPLETE END-TO-END EXAMPLE
===============================
Concrete example (this exact flow was verified live against the pipeline):

Q1 = "What is the punishment for qatl-i-amd under the Pakistan Penal Code?"

1. USER TYPES IT in ui.py's chat input.
2. ui.py appends {"role":"user","content":Q1} to st.session_state["messages"]
   and renders the user bubble. history = [] (first turn).
3. ui._run_backend(Q1, []) -> api.answer(Q1, use_llm=True, multi_hop=True,
   history=None) under a "Thinking..." spinner.
4. api.answer: history empty so no contextualization; search_q = Q1.
5. LangGraph _graph.invoke(fresh_state(Q1, max_hops=2)):
   - initialize -> router(retrieve) -> retrieve(hop 1):
       BgeM3QueryEncoder(Q1) -> QdrantHybridStore.search -> RRF over
       legal_hybrid + judgements -> ~20 chunks (PPC 302/316/320/324, ...)
   - rerank: cross-encoder scores -> top 10, top rerank_score >= 0.4
   - check_quality: planner present (LLM up) -> "expand"
   - expand: planner proposes a refined next query -> retrieve(hop 2) ->
     more evidence -> rerank
   - hop limit reached (current_hop == max_hops=2) -> check_quality returns
     "hop_limit" -> kg_expand (kg present): chunks -> Sections (BELONGS_TO)
     -> REFERENCES neighbours -> full text pulled from Qdrant, tagged _kg ->
     finalize: one rerank of the whole pool -> final_evidence (20).
6. Back in api.answer: llm_on True -> _synthesize(evidence, kg_evidence,
   Q1, None) builds the prompt (top-8 provision texts) and calls
   llm_client.llm_chat. The answer is structured markdown that cites PPC
   Section 302 (qisas/death, ta'zir death/life, up to 25 years, honour
   cases) and mentions 304/308/312/316, each with a legal caveat.
7. api returns {mode:"multi_hop", hops:2, llm_used:True, answer:<markdown>,
   evidence:[...], kg_evidence:[...]}.
8. ui builds the assistant message (content + sources + caption
   "Deep research · 2 hops · ~19.6s"), renders it, and appends it to
   st.session_state["messages"]. The chat input is still available.

Q2 = "What about an attempt to commit it?"   (a follow-up turn)
1. ui sends history = [Q1 user turn, previous assistant answer].
2. api.answer -> _contextualize_query(Q2, history): LLM rewrites Q2 into a
   standalone question mentioning "attempt ... qatl-i-amd ... Pakistan Penal
   Code (Section 302)".
3. Same graph runs on the rewritten query. This time retrieval surfaces PPC
   Section 324 ("Attempt to commit qatl-i-amd").
4. _synthesize includes the conversation block, so the answer reads
   naturally: "...an attempt to commit qatl-i-amd is punishable under
   Section 324 (up to 10 years; >=5 years if honour-related; fine; extra
   punishment if hurt caused), not the general Section 511."
5. The full conversation (Q1 + A1 + Q2 + A2) remains visible; sources and
   captions are shown for both answers.

NEW CHAT button -> st.session_state["messages"] = [] -> welcome screen
returns. Show sources toggle -> hides/shows the "Sources · N" expanders.
"""

# ==========================================================================
# 18. IMPORTANT ARCHITECTURAL / DESIGN DECISIONS
# ==========================================================================
SECTION_18 = r"""
18. IMPORTANT ARCHITECTURAL / DESIGN DECISIONS
==============================================
For each: Decision -> Why -> Alternative considered -> Why not -> Trade-off.

1. Hybrid dense + sparse (BGE-M3) instead of dense-only
   Why: legal text needs both semantics and exact terms ("section 302",
   act names); sparse lexical weights catch exact/rare tokens dense misses.
   Alternative: pure dense, or BM25. Sparse here is BGE-M3's learned
   SPLADE-style weights, not BM25. Trade-off: two indexes per collection and
   more storage, but one shared encoder at query time.

2. Server-side RRF fusion over multiple collections (legal_hybrid +
   judgements)
   Why: statutes AND case law should compete in one pool; RRF is rank-based
   and robust to scale differences between dense/sparse.
   Alternative: client-side merge or a single collection. Single collection
   would mix domains; RRF in Qdrant = fewer round trips. Trade-off: RRF
   scores are not true similarities (fine, the reranker follows).

3. Cross-encoder re-rank after hybrid retrieve (two-stage)
   Why: RRF top-20 is noisy; a cross-encoder (query x chunk jointly) is the
   strongest local relevance signal.
   Alternative: rely on RRF alone (faster, noisier). Trade-off: extra model
   load; mitigated by RERANKER_BACKEND=fast on CPU.

4. LangGraph as an explicit multi-hop state machine instead of a free-form
   "agent"
   Why: bounded, inspectable control flow (max_hops, deterministic
   conditionals) suits legal retrieval where uncontrolled loops are costly.
   Alternative: a ReAct-style LLM tool loop. Trade-off: less flexible, but
   predictable and cheap; this project deliberately does NOT do LLM
   tool-calling (the old Milvus design did; it is gone).

5. A real Neo4j knowledge graph (5-step build) over "Graph-Lite"
   Why: durable structure + Cypher traversal + citations/case law as first
   class; the retrieval bridge is the shared chunk_id/point_id.
   Alternative: Graph-Lite adjacency JSON (simpler, no server). It existed
   earlier; the project moved to Neo4j for real graph retrieval. Trade-off:
   needs Neo4j credentials/infra; adds startup coupling (graceful when down).

6. Metadata in the graph, full text in Qdrant (--drop-section-text)
   Why: avoid duplicating ~1500 large blobs; the graph returns ids/titles,
   the vector store supplies text on demand via chunk_id.
   Trade-off: an extra get_by_chunk_id call per discovered section.

7. Structure-aware, section-atomic chunking instead of fixed windows
   Why: a section is the reasoning unit; fixed windows shred legal text.
   Small-to-big (parent + children) powers deterministic expansion.
   Trade-off: complex parser with profiles; must handle OCR/contents quirks.

8. LLM as synthesizer + optional hop planner (never a knowledge source)
   Why: hallucination control; answers must be grounded in retrieved text.
   Alternative: let the LLM answer from memory. Rejected. Trade-off: answer
   quality is bounded by retrieval quality.

9. Deterministic uuid5 point ids everywhere (Qdrant, graph nodes)
   Why: idempotent ingestion/loading; re-runs never duplicate; the graph and
   vector store join 1:1.
   Trade-off: ids are opaque hashes (lookups go through payload fields).

10. Conversation handled at the app layer (Streamlit session + api.history)
    instead of graph memory
    Why: LangGraph state resets per question by design; chat memory is a UI
    concern. Follow-ups are contextualized before retrieval.
    Trade-off: no cross-turn state inside the graph; caps on history length.

11. In-process backend for the Streamlit UI (ui.py -> api.answer) plus a
    FastAPI server for the same logic
    Why: one demo command; the FastAPI layer exists for programmatic access.
    Trade-off: model singletons are per-process, so UI and API running
    together would double memory (they are meant to be run one at a time).
"""

# ==========================================================================
# 19. FUTURE IMPROVEMENTS (clearly not-yet-implemented)
# ==========================================================================
SECTION_19 = r"""
19. FUTURE IMPROVEMENTS
=======================
These are RECOMMENDATIONS, not existing features.

Retrieval / ranking
  * Full 82-judgment LLM enrichment so CITES/case-law traversal is complete
    (currently 3/82). The extract.py --tier2 path exists and is ready.
  * Add OCR for the 4 excluded scanned PDFs; re-run ingestion.
  * A proper dense-only + sparse reweighting study, or hybrid BM25+vector.
  * Re-validate REFERENCES edges (audit found ~84% textual grounding; the
    rest are regex over-matches).

Graph
  * Global/community GraphRAG summaries (needs GDS - only possible on a
    local Docker Neo4j, not AuraDB Free).
  * Serve from local Docker Neo4j + GDS for community detection.
  * Add temporal edges (amendments effective dates) for "as amended" answers.

Reasoning / agent
  * LLM intent router + optional query decomposition for multi-part
    questions (the router node is currently a stub).
  * Add deterministic legal tools back (statute/penalty/procedure lookups)
    if a tool-agent path is wanted - they existed in the old design.
  * Streaming responses and an "interrupt/stop" control in the UI.
  * Self-critique / citation-validation pass over the draft answer
    (verify every cited section is actually in evidence).

Memory / UX
  * Persist conversations (per-user storage), multiple named chats.
  * Streaming token display instead of a single spinner.
  * Keyboard focus handling, mobile layout polish.

Evaluation & ops
  * Larger labelled eval set + end-to-end answer-level metrics (citation
    accuracy, faithfulness) - the current harness is retrieval-level only.
  * LangSmith / tracing for graph runs.
  * AuthN/AuthZ, rate limiting, and secrets manager for the served app.
  * Caching of identical queries; pre-warming scripts for the API server.

Honest framing: say "the architecture makes these natural next steps", then
pick 1-2 that fit the conversation.
"""

# ==========================================================================
# 20. HOW TO EXPLAIN THIS PROJECT IN AN INTERVIEW
# ==========================================================================
SECTION_20 = r"""
20. HOW TO EXPLAIN THIS PROJECT IN AN INTERVIEW
===============================================

--- 30-SECOND EXPLANATION ---
"This is a legal research assistant for Pakistani cyber-crime and criminal
law. A user asks a natural-language legal question in a ChatGPT-style web
app. The system does hybrid semantic search over statutes AND court
judgments, re-ranks the results with a cross-encoder, optionally expands
them through a Neo4j knowledge graph of legal structure and citations, and
then asks an LLM to write an answer that cites exact act and section numbers.
Follow-up questions keep the conversation context. Everything is grounded in
retrieved evidence to reduce hallucination."

--- 1-MINUTE EXPLANATION (architecture + tech) ---
"Three layers. Data layer: PDF statutes and judgments are chunked
structure-aware (section-atomic, small-to-big), embedded with BGE-M3 into
dense plus sparse vectors, and loaded into Qdrant in two collections -
statutes and judgments. In parallel, a five-step pipeline profiles the data,
designs an ontology of 9 node types and 9 relationship types, extracts
entities and citations (deterministic, regex, and optional LLM tiers), and
loads a knowledge graph into Neo4j. Runtime layer: a LangGraph state machine
implements bounded multi-hop retrieval - retrieve, rerank, decide, expand,
optionally walk the knowledge graph, finalize. The LLM is a synthesizer, not
a knowledge source. Interface layer: a Streamlit chat UI and an optional
FastAPI server both call the same answer() function, and conversation history
is used to contextualize follow-up questions before retrieval."

--- 3-MINUTE TECHNICAL EXPLANATION (runtime flow) ---
Walk the actual code path: ui.py keeps st.session_state messages and calls
api.answer(question, history=...). answer() contextualizes the query with the
LLM, then invokes a compiled LangGraph graph built by
build_retrieval_graph(). The state is a RetrievalState TypedDict; nodes are
initialize, router (a deterministic stub that picks 'retrieve'), retrieve
(QdrantHybridStore.search - per-collection dense+sparse prefetch fused with
RRF, merged across legal_hybrid and judgements), rerank (CrossEncoderReranker
- FlagReranker bge-reranker-v2-m3 on GPU or the MiniLM cross-encoder on CPU),
then check_quality routes to expand/fallback_retrieve/kg_expand/finalize.
Expansion either uses an optional LLM planner (next query + filters) or
deterministic parent-child traversal; the knowledge-graph node maps retrieved
chunks to Section nodes and walks REFERENCES (bounded to 1 hop / 12 nodes),
re-fetching full text from Qdrant. Cycles are bounded by max_hops=2.
finalize re-ranks the whole pool. Back in api.answer, _synthesize builds the
prompt (top-8 evidence + graph case law + conversation) and the LLM returns a
structured, cited answer which the UI renders with a Sources panel. Then
explain the design decisions in Section 18 and limitations in Section 16.

--- DEEP TECHNICAL EXPLANATION (concepts tied to the code) ---
* RAG: retrieval-augmented generation; here the context is vector chunks.
* Agentic-style RAG without tool-calling: a LangGraph state machine with
  deterministic conditional edges (route_decision, check_quality) instead of
  an LLM tool loop - controllable and cheap.
* Hybrid retrieval: BGE-M3 dense + SPLADE-style sparse, fused by reciprocal
  rank fusion server-side in Qdrant (rank-based, scale-invariant).
* Graph retrieval vs vector retrieval: vectors find similar text; the Neo4j
  graph finds legally connected provisions (BELONGS_TO -> REFERENCES) and
  citing case law (CITES) that text similarity misses.
* Multi-hop: repeated retrieve+rerank with refined queries/expansion; a hop
  counter and max_hops bound every cycle; expansion stops when no new
  evidence is added (last_expansion_added) or quality is good.
* Query decomposition: NOT implemented - the closest mechanisms are the LLM
  hop planner (next-query refinement) and the conversation contextualizer.
* State management: RetrievalState TypedDict is per-question; chat memory is
  separate and lives in Streamlit session state, passed as history.
* Evidence verification: the Sources panel exposes the exact chunks; the
  answer prompt demands act+section citations from evidence only.
* Hallucination mitigation: evidence-grounded system prompt, "do not invent",
  LLM-unreachable degradation to evidence-only, and a citation panel.

INTERVIEW TIP
-------------
Always distinguish "implemented today" from "planned/legacy". Be ready to
say: the tool-calling agent, Graph-Lite and the Milvus store are from an
older design and are NOT in the current repository; the current system is a
hybrid multi-hop Graph-RAG pipeline with a Streamlit frontend and a FastAPI
backend. Also note the 3/82 case-law enrichment gap and the 7/8 recall A/B
result - showing self-awareness about limitations is a strength.
"""

# ==========================================================================
# END OF DOCUMENTATION
# ==========================================================================

# Every section above is assigned to a module constant. Running this file just
# prints a table of contents so you can jump to the section you need.

_TOC = [
    (1, "Project purpose and overall architecture"),
    (2, "Repository / file structure and responsibilities"),
    (3, "Environment variables and configuration"),
    (4, "Application startup and initialization"),
    (5, "Data ingestion -> preprocessing -> chunking -> metadata"),
    (6, "Embeddings and vector database / retrieval"),
    (7, "Graph / Graph-Lite architecture"),
    (8, "Neo4j integration (the actual knowledge graph)"),
    (9, "LangGraph state, nodes, edges, router, conditional routing, cycles"),
    (10, "Query decomposition and multi-hop retrieval"),
    (11, "Tool calling and decision-making"),
    (12, "Prompt construction and LLM interaction"),
    (13, "Conversation memory / history"),
    (14, "Streamlit frontend -> backend -> LangGraph -> LLM -> response flow"),
    (15, "Error handling and important implementation details"),
    (16, "Performance, security and limitations"),
    (17, "Complete end-to-end example (user asks a question)"),
    (18, "Important architectural / design decisions"),
    (19, "Future improvements"),
    (20, "How to explain the project in an interview"),
]


def _print_toc() -> None:
    print("=" * 74)
    print("LEGAL MULTI-HOP GRAPH-RAG AGENT - TECHNICAL WALKTHROUGH (TOC)")
    print("Open PROJECT_WALKTHROUGH.py and jump to SECTION_N for each part.")
    print("=" * 74)
    for num, title in _TOC:
        print(f"  Section {num:<2}  {title}")


if __name__ == "__main__":
    _print_toc()



