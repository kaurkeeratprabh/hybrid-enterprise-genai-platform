# Project– Module 2: Enterprise Knowledge Engineering & Advanced RAG

## Features
- Markdown parsing and normalization
- Configurable word-based chunking and overlap
- SHA-256 content/file hashes for duplicate detection and change tracking
- Metadata: document ID, department, version, status, effective date and access roles
- Chroma persistent vector index with local Sentence Transformers embeddings
- BM25 lexical retrieval + Reciprocal Rank Fusion hybrid retrieval
- Optional CrossEncoder reranking
- FastAPI ingestion and search endpoints
- Role/department filtering before returning retrieval results

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Copy the supplied 48 Markdown knowledge documents into data/corpus/
# Copy rag_qa_ground_truth.csv into data/evaluation/
uvicorn backend.api:app --reload
```
Open http://127.0.0.1:8000/docs, call POST /ingest, then POST /search.

## Important notes
- First run downloads embedding/reranker weights. After download, inference can run locally if models are cached.
- Do not infer access permissions from department names. Supply explicit `access_roles` metadata in Markdown frontmatter.
- The supplied QA CSV has department labels but no gold chunk IDs. Department-filtered retrieval scores are proxies, not valid semantic Recall@K/MRR. Add judged relevant document IDs for defensible metrics.
- Version policy: use stable `document_id`; increment `version`; set superseded versions to `status: archived`. This starter detects exact content duplicates; version conflict resolution is completed with corpus-specific metadata rules.


## Step 1 – Ingestion pipeline status
The ingestion module supports plain Markdown and YAML frontmatter, text normalization,
SHA-256 hashes, duplicate detection, stable document IDs, active-version selection,
configurable word-based chunks, role metadata and a JSON file manifest for incremental
change detection. No synthetic policy documents are included; place the official 48 files
inside `data/corpus/` when supplied.

### Suggested Markdown frontmatter
```yaml
---
document_id: VPN-001
title: VPN Access Policy
department: IT Operations
version: "1"
status: active
effective_date: 2026-01-01
access_roles:
  - employee
  - manager
---
```

### Run ingestion tests
```powershell
py -m pip install pytest pyyaml python-frontmatter
py -m pytest -q
```
These tests validate ingestion logic only; they do not require ChromaDB, embedding downloads
or the missing official corpus.
