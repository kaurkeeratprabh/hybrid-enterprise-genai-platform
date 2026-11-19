"""Persistent vector index and BM25 lexical index."""
from __future__ import annotations
import json, pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import chromadb
from sentence_transformers import SentenceTransformer

class KnowledgeIndex:
    def __init__(self, chroma_dir="data/chroma", collection_name="enterprise_knowledge", embedding_model="sentence-transformers/all-MiniLM-L6-v2", bm25_path="data/bm25.pkl"):
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        self.encoder = SentenceTransformer(embedding_model)
        self.bm25_path = Path(bm25_path)
        self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
        self.bm25 = None
        self.lexical_docs = []
        self._load_bm25()

    def _load_bm25(self):
        if self.bm25_path.exists():
            with self.bm25_path.open("rb") as f:
                saved = pickle.load(f)
            self.lexical_docs = saved["docs"]
            self.bm25 = BM25Okapi([d["text"].lower().split() for d in self.lexical_docs]) if self.lexical_docs else None

    def upsert_chunks(self, chunks):
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True).tolist()
        self.collection.upsert(ids=ids, documents=texts, metadatas=[c.metadata for c in chunks], embeddings=embeddings)
        # Rebuild lexical corpus from the complete current vector collection.
        all_rows = self.collection.get(include=["documents", "metadatas"])
        self.lexical_docs = [{"id": i, "text": t, "metadata": m} for i, t, m in zip(all_rows["ids"], all_rows["documents"], all_rows["metadatas"])]
        self.bm25 = BM25Okapi([d["text"].lower().split() for d in self.lexical_docs]) if self.lexical_docs else None
        with self.bm25_path.open("wb") as f:
            pickle.dump({"docs": self.lexical_docs}, f)
        return len(chunks)

    def lexical_search(self, query, top_k=10, allowed_roles=None, department=None):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            d = self.lexical_docs[idx]
            if not self._allowed(d["metadata"], allowed_roles, department):
                continue
            results.append({**d, "score": float(score), "retrieval": "bm25"})
            if len(results) >= top_k: break
        return results

    def vector_search(self, query, top_k=10, allowed_roles=None, department=None):
        q = self.encoder.encode([query], normalize_embeddings=True).tolist()
        where_parts = [{"status": "active"}]
        if department:
            where_parts.append({"department": department})
        if allowed_roles is not None:
            role_fields = [{f"role_{role}": True} for role in allowed_roles if role in {"employee", "manager", "admin"}]
            if not role_fields:
                return []
            where_parts.append(role_fields[0] if len(role_fields) == 1 else {"$or": role_fields})
        where = where_parts[0] if len(where_parts) == 1 else {"$and": where_parts}
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(query_embeddings=q, n_results=min(max(top_k, 1), count), where=where, include=["documents", "metadatas", "distances"])
        out = []
        for i, text in enumerate(res["documents"][0]):
            meta = res["metadatas"][0][i]
            if not self._allowed(meta, allowed_roles, department): continue
            out.append({"id": res["ids"][0][i], "text": text, "metadata": meta, "score": 1-float(res["distances"][0][i]), "retrieval": "vector"})
            if len(out) >= top_k: break
        return out

    @staticmethod
    def _allowed(meta, allowed_roles, department):
        if department and meta.get("department") != department: return False
        if allowed_roles is not None:
            roles = json.loads(meta.get("access_roles", '["employee"]'))
            if not set(roles).intersection(set(allowed_roles)): return False
        return meta.get("status", "active") == "active"

    def hybrid_search(self, query, top_k=8, allowed_roles=None, department=None, reranker=None):
        fetch = max(top_k * 3, 12)
        dense = self.vector_search(query, fetch, allowed_roles, department)
        sparse = self.lexical_search(query, fetch, allowed_roles, department)
        # Reciprocal Rank Fusion: robustly combines incomparable score scales.
        fused = {}
        for ranking in (dense, sparse):
            for rank, item in enumerate(ranking, start=1):
                key = item["id"]
                fused.setdefault(key, {**item, "rrf_score": 0.0, "sources": []})
                fused[key]["rrf_score"] += 1.0 / (60 + rank)
                fused[key]["sources"].append(item["retrieval"])
        results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)[:fetch]
        if reranker and results:
            pairs = [(query, x["text"]) for x in results]
            scores = reranker.predict(pairs)
            for item, score in zip(results, scores): item["rerank_score"] = float(score)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
