# Module 2 implementation and evidence plan

## Inputs currently available
- `rag_qa_ground_truth.csv`: 240 questions; 225 answerable and 15 unanswerable; 8 expected departments.
- The two other supplied CSVs are Module 3/4 test aids, not source knowledge documents.
- The 48 Markdown policy corpus referenced by the student guide is not present in the uploaded files currently available to this workspace.

## Implemented baseline
Markdown parser/normalizer; word-based configurable chunking; content hashes; metadata extraction; duplicate marking; Chroma vector indexing; local Sentence Transformers embeddings; BM25; Reciprocal Rank Fusion; optional CrossEncoder reranking; FastAPI ingest/search endpoints; role and department metadata filters; basic ingestion tests.

## Pending evidence
The real 48 Markdown documents are required to produce the actual indexed corpus, identify policy versions/conflicts, tune chunking and report measured Recall@K/MRR. The QA CSV does not provide gold document/chunk IDs, so department match can only be reported as a proxy unless relevant IDs are annotated.
