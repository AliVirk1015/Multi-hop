"""
Graph RAG package for the legal multi-hop agent.

Pipeline (5 steps):
    1. Data profiling  -> Graph/profile.py  (produces schema_report.json + SCHEMA.md)
    2. Ontology design -> Graph/ontology.py (node/relationship type spec)   [planned]
    3. Extraction      -> Graph/extract.py  (Tier 0/1/2 entity+relation extraction) [planned]
    4. Graph assembly  -> Graph/build.py    (dedupe, stable IDs -> nodes/edges JSONL) [planned]
    5. Neo4j load      -> Graph/neo4j_store.py (constraints + batch UNWIND MERGE) [planned]
"""
