from pathlib import Path
import json
import pytest

from backend.ingestion.pipeline import (
    normalize_text, chunk_text, parse_markdown, ingest_directory, compare_manifest
)


def test_normalize():
    assert normalize_text("a  b\r\n\r\n\r\nc") == "a b\n\nc"


def test_chunk_overlap_and_word_limit():
    words = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_text(words, 500, 50)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 500
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]


@pytest.mark.parametrize("size,overlap", [(49, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size, overlap)


def test_plain_markdown_metadata(tmp_path):
    p = tmp_path / "policy.md"
    p.write_text("# VPN Policy\nVPN access is reviewed every 180 days.", encoding="utf-8")
    doc = parse_markdown(p)
    assert doc["title"] == "VPN Policy"
    assert doc["document_id"] == "policy"
    assert doc["department"] == tmp_path.name


def test_frontmatter_and_version_selection(tmp_path):
    (tmp_path / "old.md").write_text(
        "---\ndocument_id: leave-policy\nversion: '1'\ndepartment: HR\naccess_roles: [employee]\n---\n# Leave\nOld policy text.",
        encoding="utf-8")
    (tmp_path / "new.md").write_text(
        "---\ndocument_id: leave-policy\nversion: '2'\ndepartment: HR\naccess_roles: [employee, manager]\n---\n# Leave\nNew policy text.",
        encoding="utf-8")
    docs, chunks = ingest_directory(tmp_path, chunk_size=100, overlap=10)
    assert len(docs) == 2
    assert len(chunks) == 1
    assert chunks[0].metadata["version"] == "2"
    assert chunks[0].metadata["role_manager"] is True


def test_manifest_and_incremental_comparison(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    f = corpus / "a.md"
    f.write_text("# A\nInitial.", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    ingest_directory(corpus, manifest_path=manifest)
    assert json.loads(manifest.read_text())["files"]["a.md"]["document_id"] == "a"
    assert compare_manifest(corpus, manifest)["unchanged"] == ["a.md"]
    f.write_text("# A\nChanged.", encoding="utf-8")
    assert compare_manifest(corpus, manifest)["changed"] == ["a.md"]
