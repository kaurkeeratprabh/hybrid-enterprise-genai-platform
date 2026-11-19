
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
import os
from backend.agent_tools import execute_tool

from backend.ingestion.pipeline import ingest_directory
from backend.retrieval.index import KnowledgeIndex
from backend.rag.service import answer_with_local_model


app = FastAPI(
    title="Enterprise Knowledge Retrieval API",
    version="1.0.0"
)

index = None
reranker = None


# ---------------- REQUEST MODELS ----------------

class IngestRequest(BaseModel):
    corpus_dir: str = "data/corpus"
    chunk_size: int = Field(default=700, ge=50, le=3000)
    overlap: int = Field(default=100, ge=0, le=2999)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=30)
    roles: list[str] = Field(
        default_factory=lambda: ["employee"]
    )
    department: str | None = None
    rerank: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    role: str = "employee"
    department: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
class ToolRequest(BaseModel):
    name: str
    arguments: dict = Field(default_factory=dict)
    role: str = "employee"

# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup():
    global index, reranker

    index = KnowledgeIndex(
        chroma_dir=os.getenv("CHROMA_DIR", "data/chroma"),
        collection_name=os.getenv(
            "COLLECTION_NAME",
            "enterprise_knowledge"
        ),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    try:
        reranker = CrossEncoder(
            os.getenv(
                "RERANKER_MODEL",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
    except Exception as e:
        print("Reranker unavailable:", e)
        reranker = None


# ---------------- HEALTH ----------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": index.collection.count() if index else 0
    }


# ---------------- INGESTION ----------------

@app.post("/ingest")
def ingest(req: IngestRequest):

    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Index is not ready"
        )

    try:
        docs, chunks = ingest_directory(
            req.corpus_dir,
            req.chunk_size,
            req.overlap,
            manifest_path="data/manifest.json"
        )

        count = index.upsert_chunks(chunks)

        return {
            "documents_seen": len(docs),
            "chunks_indexed": count,
            "duplicates": sum(
                d.get("status") == "duplicate"
                for d in docs
            ),
            "chunk_size_words": req.chunk_size,
            "overlap_words": req.overlap
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ---------------- HYBRID SEARCH ----------------

@app.post("/search")
def search(req: SearchRequest):

    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Index is not ready"
        )

    results = index.hybrid_search(
        req.query,
        top_k=req.top_k,
        allowed_roles=req.roles,
        department=req.department,
        reranker=reranker if req.rerank else None
    )

    return {
        "query": req.query,
        "results": results,
        "count": len(results)
    }


# ---------------- RAG QUESTION ANSWERING ----------------

@app.post("/ask")
def ask(req: AskRequest):

    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Index is not ready"
        )

    # Step 1: Retrieve relevant documents
    results = index.hybrid_search(
        req.question,
        top_k=req.top_k,
        allowed_roles=[req.role],
        department=req.department,
        reranker=reranker
    )

    # Step 2: Abstain if no evidence is found
    if not results:
        return {
            "question": req.question,
            "answer": "Insufficient evidence to answer this question.",
            "citations": [],
            "abstained": True
        }

    # Step 3: Generate answer using local Ollama model
    try:
        response = answer_with_local_model(
            req.question,
            results
        )

        return {
            "question": req.question,
            **response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local model generation failed: {str(e)}"
        )
@app.post("/tools/execute")
def run_tool(req: ToolRequest):

    result = execute_tool(
        req.name,
        req.arguments,
        req.role
    )

    if "error" in result:
        raise HTTPException(status_code=403, detail=result)

    return result