"""Embedding batching and local Chroma maintenance. The embedding API is faked."""

import contextlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.vector_store import (
    DashScopeCompatibleEmbeddings,
    build_vector_store,
    clear_vector_store,
    delete_document,
    list_documents,
)


class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _Response:
    def __init__(self, data):
        self.data = data


class _EmbeddingsAPI:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": list(input)})
        # Return out of order so the client must sort by index.
        data = [_Item(index, [float(index), 0.5]) for index in range(len(input))]
        data.reverse()
        return _Response(data)


class _FakeOpenAI:
    embeddings_api = _EmbeddingsAPI()

    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings = _FakeOpenAI.embeddings_api


class FakeEmbeddings(Embeddings):
    """Fixed vectors so Chroma can persist without a hosted embedding model."""

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_DB_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr("src.vector_store.build_embeddings", lambda api_key=None: FakeEmbeddings())
    yield tmp_path
    with contextlib.suppress(Exception):
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()


def test_embeddings_require_an_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API Key"):
        DashScopeCompatibleEmbeddings()


def test_embed_documents_batches_and_restores_index_order(monkeypatch):
    _FakeOpenAI.embeddings_api = _EmbeddingsAPI()
    monkeypatch.setattr("src.vector_store.OpenAI", _FakeOpenAI)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "unit-embed")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")

    embeddings = DashScopeCompatibleEmbeddings()
    vectors = embeddings.embed_documents(["a", "b", "c"])

    assert _FakeOpenAI.embeddings_api.calls == [
        {"model": "unit-embed", "input": ["a", "b"]},
        {"model": "unit-embed", "input": ["c"]},
    ]
    assert vectors == [[0.0, 0.5], [1.0, 0.5], [0.0, 0.5]]
    assert embeddings.embed_query(123) == [0.0, 0.5]


def test_rebuild_replaces_same_source_and_keeps_other_files(isolated_store):
    first = [
        Document(page_content="alpha one", metadata={"source": "/docs/a.md"}),
        Document(page_content="alpha two", metadata={"source": "/docs/a.md"}),
    ]
    build_vector_store(first, api_key="test-key")

    listed = list_documents()
    assert listed == [{"source": "/docs/a.md", "name": "a.md", "chunks": 2}]

    build_vector_store(
        [Document(page_content="alpha replaced", metadata={"source": "/docs/a.md"})],
        api_key="test-key",
    )
    build_vector_store(
        [Document(page_content="beta", metadata={"source": "/docs/b.md"})],
        api_key="test-key",
    )

    listed = list_documents()
    assert listed == [
        {"source": "/docs/a.md", "name": "a.md", "chunks": 1},
        {"source": "/docs/b.md", "name": "b.md", "chunks": 1},
    ]

    assert delete_document("/docs/b.md") == 1
    assert delete_document("/docs/missing.md") == 0
    assert list_documents() == [{"source": "/docs/a.md", "name": "a.md", "chunks": 1}]

    clear_vector_store()
    assert list_documents() == []


def test_build_vector_store_rejects_empty_input(isolated_store):
    with pytest.raises(ValueError, match="没有可用的文本内容"):
        build_vector_store([], api_key="test-key")
