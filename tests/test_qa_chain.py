"""QA and chat streaming with a stand-in model client. No paid API is called."""

from types import SimpleNamespace

import pytest
from langchain_core.prompts import PromptTemplate

from src.qa_chain import (
    NO_ANSWER_TEXT,
    PROMPT_TEMPLATE,
    ask_question,
    build_qwen_client,
    create_qa_chain,
    stream_chat,
    stream_question,
    _completion_kwargs,
)


class _Completions:
    def __init__(self, responder):
        self.calls = []
        self._responder = responder

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responder(kwargs)


class _Client:
    def __init__(self, responder):
        self.chat = SimpleNamespace(completions=_Completions(responder))


def _message(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _delta(content=None, reasoning_content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, reasoning_content=reasoning_content)
            )
        ]
    )


def _chain(responder, docs):
    client = _Client(responder)
    return {
        "client": client,
        "retriever": SimpleNamespace(invoke=lambda question: docs),
        "prompt": PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        ),
    }, client


def test_build_qwen_client_requires_a_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API Key"):
        build_qwen_client()


def test_build_qwen_client_uses_explicit_key_and_base_url(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key, base_url):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    monkeypatch.setattr("src.qa_chain.OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")

    build_qwen_client(api_key="session-key")

    assert captured == {"api_key": "session-key", "base_url": "https://example.test/v1"}


def test_completion_kwargs_omit_thinking_when_disabled(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "unit-test-model")
    monkeypatch.setenv("ENABLE_THINKING", "false")

    assert _completion_kwargs(stream=False) == {
        "model": "unit-test-model",
        "stream": False,
    }

    monkeypatch.setenv("ENABLE_THINKING", "true")
    assert _completion_kwargs(stream=True)["extra_body"] == {"enable_thinking": True}


def test_ask_question_skips_the_model_when_nothing_is_retrieved():
    chain, client = _chain(lambda kwargs: _message("should not be used"), [])

    result = ask_question(chain, "Where is the refund policy?")

    assert result == {"result": NO_ANSWER_TEXT, "source_documents": []}
    assert client.chat.completions.calls == []


def test_ask_question_grounds_the_prompt_and_returns_sources():
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Refunds take 7 days.", metadata={"source": "policy.md", "page": 0})
    ]
    chain, client = _chain(lambda kwargs: _message("Refunds take 7 days."), docs)

    result = ask_question(chain, "How long do refunds take?")

    assert result["result"] == "Refunds take 7 days."
    assert result["source_documents"] == docs
    prompt = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "Refunds take 7 days." in prompt
    assert "How long do refunds take?" in prompt
    assert "未找到相关内容" in prompt
    assert client.chat.completions.calls[0]["stream"] is False


def test_ask_question_uses_fallback_when_model_content_is_empty():
    from langchain_core.documents import Document

    docs = [Document(page_content="Some context.", metadata={"source": "a.md"})]
    chain, _client = _chain(lambda kwargs: _message(""), docs)

    result = ask_question(chain, "Anything?")

    assert result["result"] == NO_ANSWER_TEXT


def test_stream_question_yields_reasoning_then_answer():
    from langchain_core.documents import Document

    docs = [Document(page_content="Office hours are 9 to 5.", metadata={"source": "hours.md"})]

    def responder(_kwargs):
        return iter(
            [
                SimpleNamespace(choices=[]),
                SimpleNamespace(choices=[SimpleNamespace(delta=None)]),
                _delta(reasoning_content="Look at hours."),
                _delta(content="Office hours are 9 to 5."),
            ]
        )

    chain, client = _chain(responder, docs)
    steps = list(stream_question(chain, "When are you open?"))

    assert [step["reasoning"] for step in steps] == ["Look at hours.", "Look at hours."]
    assert steps[-1]["answer"] == "Office hours are 9 to 5."
    assert steps[-1]["source_documents"] == docs
    assert client.chat.completions.calls[0]["stream"] is True


def test_stream_question_returns_no_answer_without_calling_the_model():
    chain, client = _chain(lambda kwargs: iter([_delta(content="nope")]), [])

    steps = list(stream_question(chain, "Unknown topic"))

    assert steps == [{"reasoning": "", "answer": NO_ANSWER_TEXT, "source_documents": []}]
    assert client.chat.completions.calls == []


def test_stream_chat_accumulates_deltas(monkeypatch):
    monkeypatch.setattr("src.qa_chain.build_qwen_client", lambda api_key=None: _Client(
        lambda kwargs: iter([_delta(content="Hello"), _delta(content=" there")])
    ))

    steps = list(stream_chat([{"role": "user", "content": "Hi"}], api_key="session-key"))

    assert [step["answer"] for step in steps] == ["Hello", "Hello there"]
    assert steps[-1]["reasoning"] == ""


def test_create_qa_chain_exposes_prompt_client_and_retriever(monkeypatch):
    monkeypatch.setattr("src.qa_chain.build_qwen_client", lambda api_key=None: "client")
    monkeypatch.setattr("src.qa_chain.load_retriever", lambda api_key=None: "retriever")

    chain = create_qa_chain(api_key="session-key")

    rendered = chain["prompt"].format(context="CTX", question="Q?")
    assert chain["client"] == "client"
    assert chain["retriever"] == "retriever"
    assert "CTX" in rendered
    assert "Q?" in rendered
