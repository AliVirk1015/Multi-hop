# Pakistan Cyber Crime Legal Reasoning Agent

An **agentic RAG system** for Pakistani statutory law — with cyber-crime as the
primary domain — built on LangGraph, Milvus Standalone (Docker), structure-aware
legal chunking, hybrid dense+sparse retrieval, deterministic legal tools, and
citation-grounded synthesis with reflection/verification.

> **Legal safety:** this system provides *informational / advisory* legal
> reasoning only. It is **not** a licensed legal practitioner and its output is
> **not** guaranteed legal advice. High-stakes matters are flagged and deferred
> to a qualified lawyer.

## Architecture

```
User → LangGraph Agent → Router (multi-intent)
        → Multi-collection retrieval (legal_cyber / legal_criminal / legal_civil / legal_judgments)
        → Metadata filtering → Graph-Lite traversal → Deterministic tools
        → Evidence aggregation → Risk assessment → Synthesis → Reflection
        → Final grounded answer (or clarification / re-retrieval)
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (copy and edit as needed)
copy .env.example .env          # Windows
# cp .env.example .env          # Linux/macOS

# 3. Start Milvus Standalone (Docker)
docker compose -f docker-compose.milvus.yml up -d

# 4. Add statute documents to data/legal_docs/ (see data/ for existing PDFs)
#    then build the vector store:
python -m retrieval.ingest

# 5. Run the agent
python main.py
```

## Embedding backends

| Backend | Notes |
|---|---|
| `tfidf` | Offline deterministic stand-in (default — no model download). |
| `sentence_transformer` | Dense only (requires `sentence-transformers`). |
| `bge_m3` | Primary spec model `BAAI/bge-m3` (dense + sparse). Requires `pip install FlagEmbedding`. |

Set the backend via `.env` / env var `EMBEDDER_BACKEND`. Reranking is controlled
by `RERANKER_BACKEND` (`none`, `cross_encoder`, or `llm`).

## Testing

```bash
python test_dry_run.py      # offline graph-wiring check (no live LLM)
python -m evaluation.evaluate  # retrieval-level evaluation (no LLM)
```

## Evaluation

See `evaluation/` for the dataset and harness. Measured retrieval-level results
(2026-08-17): overall hit rate 0.75, case-law relevance 1.0, cross-act coverage
1.0, latency ~85 ms avg. LLM-dependent metrics (hallucination rate, final-answer
citation accuracy) require the live gateway.

## Documentation

- `PLAN/README.md` — canonical specification + implementation status
- `PLAN/architecture.md` — architecture diagrams and design decisions
- `PLAN/ingestion.md` — ingestion / chunking / metadata
- `PLAN/retrieval.md` — Milvus / retrieval / reranking
- `PLAN/agent.md` — LangGraph / tools / Graph-Lite
- `PLAN/evaluation.md` — evaluation dataset, metrics, findings
- `AGENTS.md` — implementation agent directives
