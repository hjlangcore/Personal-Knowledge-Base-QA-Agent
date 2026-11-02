"""Settings reads environment variables and does not need an API call."""

import pytest

from src.config import Settings


def _clear_settings(monkeypatch):
    for name in (
        "DASHSCOPE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CHAT_MODEL",
        "EMBEDDING_MODEL",
        "VECTOR_DB_DIR",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_K",
        "EMBEDDING_BATCH_SIZE",
        "ENABLE_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_env_is_unset(monkeypatch):
    _clear_settings(monkeypatch)

    assert Settings.get_api_key() == ""
    assert Settings.get_openai_base_url() == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert Settings.get_chat_model() == "qwen3.6-plus"
    assert Settings.get_embedding_model() == "text-embedding-v4"
    assert Settings.get_vector_db_dir() == "data/chroma_db"
    assert Settings.get_chunk_size() == 500
    assert Settings.get_chunk_overlap() == 100
    assert Settings.get_retrieval_k() == 4
    assert Settings.get_embedding_batch_size() == 10
    assert Settings.get_enable_thinking() is True


def test_dashscope_key_wins_over_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")

    assert Settings.get_api_key() == "dashscope-key"

    monkeypatch.delenv("DASHSCOPE_API_KEY")
    assert Settings.get_api_key() == "openai-key"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("  off  ", False),
    ],
)
def test_enable_thinking_parses_common_flags(monkeypatch, raw, expected):
    monkeypatch.setenv("ENABLE_THINKING", raw)
    assert Settings.get_enable_thinking() is expected


def test_embedding_batch_size_is_at_least_one(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "0")
    assert Settings.get_embedding_batch_size() == 1

    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "-4")
    assert Settings.get_embedding_batch_size() == 1

    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "6")
    assert Settings.get_embedding_batch_size() == 6
