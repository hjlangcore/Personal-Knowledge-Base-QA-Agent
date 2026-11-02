"""Document loading and splitting stay local: no model and no network."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.document_processor import load_document, split_documents


def test_load_markdown_uses_utf8(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("# 标题\n\nRefunds are processed within 7 days.\n", encoding="utf-8")

    documents = load_document(str(path))

    assert len(documents) == 1
    assert "Refunds are processed within 7 days." in documents[0].page_content
    assert documents[0].metadata["source"].endswith("notes.md")


def test_unsupported_suffix_is_rejected(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("plain text", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(str(path))


def test_split_documents_respects_chunk_settings(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "20")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    page = "第一句。" * 12
    documents = [Document(page_content=page, metadata={"source": "policy.md"})]

    chunks = split_documents(documents)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 20 for chunk in chunks)
    assert all(chunk.metadata["source"] == "policy.md" for chunk in chunks)
    assert "".join(chunk.page_content for chunk in chunks) == page
