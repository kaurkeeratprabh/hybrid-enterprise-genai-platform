"""Reusable Markdown ingestion pipeline for Project 03, Module 2."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize line endings, tabs, repeated spaces and excessive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title_from_body(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return ""


def _load_markdown(path: Path) -> tuple[dict[str, Any], str]:
    """Read YAML frontmatter when present; plain Markdown is also supported."""
    raw = path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    body = raw
    if raw.startswith("---\n"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            try:
                import yaml
                loaded = yaml.safe_load(parts[1]) or {}
                if isinstance(loaded, dict):
                    metadata = loaded
                    body = parts[2]
            except Exception as exc:
                raise ValueError(f"Invalid YAML frontmatter in {path.name}: {exc}") from exc
    return metadata, normalize_text(body)


def parse_markdown(path: Path) -> dict[str, Any]:
    meta, body = _load_markdown(path)
    raw = path.read_text(encoding="utf-8")
    roles = meta.get("access_roles", ["employee", "manager", "admin"])
    if isinstance(roles, str):
        roles = [roles]
    if not isinstance(roles, list) or not all(isinstance(x, str) for x in roles):
        raise ValueError(f"access_roles must be a list of strings in {path.name}")
    doc_id = str(meta.get("document_id") or path.stem)
    version = str(meta.get("version", "1"))
    status = str(meta.get("status", "active")).lower()
    if status not in {"active", "archived", "draft", "superseded"}:
        raise ValueError(f"Unsupported status '{status}' in {path.name}")
    return {
        "document_id": doc_id,
        "source": path.name,
        "source_path": str(path),
        "title": str(meta.get("title") or _title_from_body(body) or path.stem),
        "department": str(meta.get("department") or (path.parent.name if path.parent.name not in {"", "."} else "General")),
        "version": version,
        "status": status,
        "effective_date": str(meta.get("effective_date", "")),
        "updated_at": str(meta.get("updated_at", "")),
        "access_roles": roles,
        "content_hash": sha256(body),
        "file_hash": sha256(raw),
        "text": body,
    }


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Split normalized text into word-count chunks with a fixed overlap."""
    if chunk_size < 50:
        raise ValueError("chunk_size must be at least 50 words")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def ingest_directory(
    corpus_dir: str | Path,
    chunk_size: int = 700,
    overlap: int = 100,
    manifest_path: str | Path | None = None,
):
    """Ingest Markdown files; identify exact duplicates and retain version metadata.

    The manifest records file hashes so subsequent runs can identify unchanged,
    added and changed files. Index replacement/archival is handled by the index layer.
    """
    root = Path(corpus_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {root.resolve()}")
    paths = sorted(root.rglob("*.md"))
    docs: list[dict[str, Any]] = []
    seen_content: dict[str, str] = {}
    for path in paths:
        doc = parse_markdown(path)
        if doc["content_hash"] in seen_content:
            doc["duplicate_of"] = seen_content[doc["content_hash"]]
            doc["status"] = "duplicate"
        else:
            seen_content[doc["content_hash"]] = doc["document_id"]
        docs.append(doc)

    # For each stable document_id, only the highest active version is indexed.
    active_by_id: dict[str, dict[str, Any]] = {}
    for doc in docs:
        if doc["status"] != "active":
            continue
        previous = active_by_id.get(doc["document_id"])
        if previous is None or _version_key(doc["version"]) > _version_key(previous["version"]):
            active_by_id[doc["document_id"]] = doc

    active_ids = {id(doc) for doc in active_by_id.values()}
    chunks: list[Chunk] = []
    for doc in docs:
        if id(doc) not in active_ids:
            continue
        for i, chunk in enumerate(chunk_text(doc["text"], chunk_size, overlap)):
            chunk_id = sha256(f'{doc["document_id"]}:{doc["version"]}:{i}:{chunk}')[:32]
            metadata = {
                "document_id": doc["document_id"], "source": doc["source"],
                "title": doc["title"], "department": doc["department"],
                "version": doc["version"], "status": doc["status"],
                "effective_date": doc["effective_date"], "updated_at": doc["updated_at"],
                "content_hash": doc["content_hash"], "chunk_index": i,
                "access_roles": json.dumps(doc["access_roles"]),
            }
            for role in ("employee", "manager", "admin"):
                metadata[f"role_{role}"] = role in doc["access_roles"]
            chunks.append(Chunk(chunk_id, doc["document_id"], chunk, metadata))

    if manifest_path:
        manifest_file = Path(manifest_path)
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "files": {
                str(Path(d["source_path"]).relative_to(root)): {
                    "file_hash": d["file_hash"], "document_id": d["document_id"],
                    "version": d["version"], "status": d["status"]
                } for d in docs
            }
        }
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return docs, chunks


def compare_manifest(corpus_dir: str | Path, manifest_path: str | Path) -> dict[str, list[str]]:
    """Report added, changed, unchanged and removed Markdown files."""
    root, manifest_file = Path(corpus_dir), Path(manifest_path)
    old = json.loads(manifest_file.read_text(encoding="utf-8"))["files"] if manifest_file.exists() else {}
    current = {}
    for path in sorted(root.rglob("*.md")):
        rel = str(path.relative_to(root))
        current[rel] = sha256(path.read_text(encoding="utf-8"))
    added = sorted(set(current) - set(old))
    removed = sorted(set(old) - set(current))
    changed = sorted(k for k in set(current) & set(old) if current[k] != old[k].get("file_hash"))
    unchanged = sorted(k for k in set(current) & set(old) if current[k] == old[k].get("file_hash"))
    return {"added": added, "changed": changed, "unchanged": unchanged, "removed": removed}


def _version_key(version: str):
    """Compare numeric dotted versions naturally; fall back to string ordering."""
    try:
        return (1, tuple(int(part) for part in version.split(".")))
    except ValueError:
        return (0, version.casefold())
