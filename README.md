# Verus — Multi-Hop Graph-RAG Legal Research Assistant

A **hybrid multi-hop Graph-RAG system** for Pakistani cyber-crime and criminal
law. Ask a legal question in natural language and get a structured, cited
answer built from the statutes (PPC, CrPC, PECA, AML, …) and high-court
judgments — through vector search **plus** a Neo4j knowledge graph, orchestrated
by a bounded LangGraph multi-hop retriever.

> **Legal safety:** informational / advisory research only — **not** a licensed
> legal practitioner and **not** guaranteed legal advice. Consult a qualified
> lawyer for any real matter.

## What it does

- **ChatGPT-style web UI** (Streamlit) with conversation memory, follow-up
  support, a "Sources" citation panel and a persistent input box.
- **Hybrid retrieval**: BGE-M3 dense (1024-d) + sparse (SPLADE-style) vectors,
  fused by reciprocal-rank fusion **server-side in Qdrant**, over two
  collections — statutes (`legal_hybrid`) and judgments (`judgements`).
- **Cross-encoder re-ranking** for precision (bge-reranker-v2-m3 on GPU, a fast
  MiniLM cross-encoder on CPU).
- **Neo4j knowledge graph**: a hand-built legal graph (Act → Chapter/Section,
  chunk membership, section cross-references, judgment citations) whose
  retrieved "related provisions / citing case law" join the evidence pool.
- **Bounded multi-hop reasoning**: a LangGraph state machine can re-query with a
  refined question (optional LLM planner) or expand parent/child chunks, until a
  quality/hop budget is met.
- **Grounded synthesis**: the LLM answers **only from retrieved evidence** and
  must cite exact act + section; answers degrade gracefully (evidence-only) when
  the LLM is unavailable.
- **Three entry points**: Streamlit UI, FastAPI server, and a CLI.

## Architecture

```
Documents (statutes + judgments)
   │  ingestion/ chunking (structure-aware, small-to-big)
   v
chunks.json / judgement_chunks.json ──BGE-M3──> qdrant-store/*.json
   │                                            │
   v                                            v
Graph/ (5-step build)                     Qdrant collections
 profile → ontology → extract → build        legal_hybrid + judgements
   │                                           (dense+sparse, RRF fusion)
   v
Neo4j AuraDB  ◄─────────────────────────────── chunk_id == point_id (uuid5)

                         RUNTIME
   User → ui.py (Streamlit) → api.answer(question, history)
        → LangGraph multi-hop retriever
             retrieve (Qdrant hybrid) → rerank → decide
             → expand / fallback / kg_expand (Neo4j) → finalize
        → evidence + kg_evidence → LLM (llm_client.py)
        → cited markdown answer → UI + "Sources · N"
```

## Repository layout

```
ui.py                 Streamlit chat UI            api.py        FastAPI backend
main_query.py         CLI                          llm_client.py OpenAI-compatible client
retrieval/            encoder, Qdrant store, reranker, fast path
retrieval_graph/      LangGraph: state.py, nodes.py, graph.py
Graph/                KG build (profile/ontology/extract/build/neo4j_store) + query.py client
ingestion/            PDF → chunk: chunking.py, Judgement_chunker.py
qdrant-store/         embed → Qdrant: ingest_hybrid.py, ingest_judgements.py
evaluation/           recall5.py, kg_eval.py, extraction_audit.py, faithfulness_eval.py
assets/, .streamlit/  UI theme + styles
PROJECT_WALKTHROUGH.py  full A-to-Z technical walkthrough (interview prep)
```

## Requirements

- Python 3.11 (`venv` recommended)
- `pip install -r requirements.txt`
- BGE-M3 models are needed at runtime but `FlagEmbedding` is commented out in
  `requirements.txt` (large). Install it if not present:
  `pip install FlagEmbedding` (first run downloads ~2.3 GB per model to the HF
  cache). `sentence-transformers` covers the "fast" reranker.
- External services (graceful if Neo4j/LLM are down; Qdrant is required):
  - **Qdrant** on `localhost:6333` with collections `legal_hybrid` + `judgements`
  - **Neo4j** (AuraDB) with the loaded legal graph
  - an **OpenAI-compatible LLM gateway** (OpenRouter / Groq / OpenAI)

## Configuration

Copy `.env.example` to `.env` and set the values. Key variables the code reads:

| Group | Variables |
|---|---|
| LLM | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TIMEOUT` (or `OPENAI_API_KEY` fallback) |
| Qdrant | `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTIONS` (default `legal_hybrid,judgements`) |
| Models | `BGE_MODEL` (BAAI/bge-m3), `RERANKER_MODEL`, `RERANKER_BACKEND` (`auto`/`bge`/`fast`) |
| Neo4j | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` |

> `.env.example` also contains **legacy** variables (MLflow, Milvus, agent
> knobs) that no current module reads — harmless leftovers.

## Quick start

```powershell
# 1. install deps
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. copy + edit .env (LLM gateway, Qdrant, Neo4j creds)
copy .env.example .env

# 3a. Web UI  (primary demo entry point)
.venv\Scripts\python.exe -m streamlit run ui.py
#     → http://localhost:8501  (first load warms models ~1 min)

# 3b. FastAPI
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
#     → GET /health, POST /query {"query": "...", "use_llm": true, "multi_hop": true}

# 3c. CLI
.venv\Scripts\python.exe main_query.py --query "What is the punishment for qatl-i-amd under the Pakistan Penal Code?" --multi-hop
```

Try in the UI:

```
What is the punishment for qatl-i-amd under the Pakistan Penal Code?
→ 2 hops · cites PPC §302 (+ §304/308/312/316) · Sources · 12

Then ask (same chat, no reset):  What about an attempt to commit it?
→ resolves to §324 (attempt) via conversation history
```

## Data & current indexes (observed)

- `chunks.json` — 1871 statute/rules/amendment chunks (15 documents)
- `judgement_chunks.json` — 1199 judgment chunks (82 judgments)
- Qdrant `legal_hybrid` / `judgements` — same counts, dense + sparse
- Neo4j — 4881 nodes / 7116 edges (Act, Part, Chapter, Section, Chunk,
  Judgment, Court, Party, Offence × CONTAINS, PARENT_OF, BELONGS_TO,
  AMENDED_BY, REFERENCES, CITES, DECIDED_BY, INVOLVES, CONCERNS)

## Rebuilding the data pipeline (only when you change the corpus)

```powershell
# chunks
.venv\Scripts\python.exe ingestion\chunking.py --data data --out chunks.json
.venv\Scripts\python.exe ingestion\Judgement_chunker.py --data "data\Judgements" --out judgement_chunks.json

# knowledge graph (5 steps → Neo4j)
.venv\Scripts\python.exe Graph\profile.py
.venv\Scripts\python.exe Graph\ontology.py
.venv\Scripts\python.exe Graph\extract.py                # add --tier2 for LLM judgment enrichment
.venv\Scripts\python.exe Graph\build.py --drop-section-text
.venv\Scripts\python.exe Graph\neo4j_store.py --load --verify

# Qdrant ingest (needs BGE-M3 embed files under qdrant-store/)
.venv\Scripts\python.exe qdrant-store\ingest_hybrid.py --collection legal_hybrid
.venv\Scripts\python.exe qdrant-store\ingest_judgements.py
```

## Evaluation

Run against a live Qdrant (+ Neo4j where noted); first run loads models.

| Script | Measures | Latest observed |
|---|---|---|
| `evaluation/recall5.py` | Recall@5 of the fast path (8-query gold set) | 7/8 (87.5%) |
| `evaluation/kg_eval.py` | A/B: baseline vs KG-augmented recall | 7/8 both; KG adds coverage (48 related sections) |
| `evaluation/extraction_audit.py` | KG edge grounding in text | REFERENCES ~84% |
| `evaluation/faithfulness_eval.py` | Answer-level faithfulness (LLM-as-judge) | proxy; see its caveats |

## Limitations (current)

- Only 3 of 82 judgments are LLM-enriched, so `CITES` case-law coverage is
  sparse (22 edges).
- No OCR — 4 scanned PDFs are excluded; corpus is fixed at 15 statutes + 82
  judgments.
- Multi-hop is bounded (`max_hops=2`) and graph expansion is 1 hop / ≤12 nodes.
- Conversation history lives in the Streamlit session (no persistence);
  answer generation is single-shot (no streaming).
- Answer-level faithfulness is measured by an LLM-as-judge proxy, not a
  production harness.

## Documentation

- `PROJECT_WALKTHROUGH.py` — complete A-to-Z technical walkthrough, written for
  interview preparation (run it to print a table of contents).
- `evaluation/faithfulness_eval.py` — how the faithfulness proxy is computed.
