"""QA 链构建与流式交互逻辑

提供创建问答链、检索、以及流式输出的核心组件。对外暴露的接口
包括 create_qa_chain、ask_question 和 stream_question 等。"""
from openai import OpenAI
from langchain_core.prompts import PromptTemplate

from src.config import Settings
from src.vector_store import load_retriever


PROMPT_TEMPLATE = """你是一个专业的文档助手。请仅基于以下上下文回答用户的问题：
----------------
{context}
----------------
问题：{question}
要求：
1. 答案必须来自上下文；
2. 若上下文无相关信息，请回答“未找到相关内容”；
3. 回答尽量简洁、准确。
"""

NO_ANSWER_TEXT = "未找到相关内容"


def build_qwen_client() -> OpenAI:
    """构建用于对话/文本生成的 OpenAI 客户端。"""
    api_key = Settings.get_api_key()
    if not api_key:
        raise ValueError(
            "未检测到 API Key。请设置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）。"
        )
    return OpenAI(
        api_key=api_key,
        base_url=Settings.get_openai_base_url(),
    )


def create_qa_chain():
    """创建用于问答的链对象，包含 client、retriever、prompt。"""
    client = build_qwen_client()
    retriever = load_retriever()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return {
        "client": client,
        "retriever": retriever,
        "prompt": prompt,
    }


def _completion_kwargs(stream: bool):
    """构造对话补全参数；仅在开启思考模式时传递 enable_thinking。"""
    kwargs = {
        "model": Settings.get_chat_model(),
        "stream": stream,
    }
    if Settings.get_enable_thinking():
        kwargs["extra_body"] = {"enable_thinking": True}
    return kwargs


def ask_question(chain_parts, question: str):
    """基于给定的 chain_parts 执行检索并生成答案。"""
    retriever = chain_parts["retriever"]
    docs = retriever.invoke(question)
    if not docs:
        return {"result": NO_ANSWER_TEXT, "source_documents": []}
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = _create_completion_text(chain_parts, context, question)
    return {"result": answer, "source_documents": docs}


def _create_completion_text(chain_parts, context: str, question: str) -> str:
    """内部：调用模型完成并返回文本答案。"""
    client = chain_parts["client"]
    prompt = chain_parts["prompt"]
    final_prompt = prompt.format(context=context, question=question)
    messages = [{"role": "user", "content": final_prompt}]
    completion = client.chat.completions.create(
        messages=messages,
        **_completion_kwargs(stream=False),
    )
    return completion.choices[0].message.content or NO_ANSWER_TEXT


def stream_question(chain_parts, question: str):
    """流式输出：逐步生成答案及思考过程（如模型返回）"""
    client = chain_parts["client"]
    prompt = chain_parts["prompt"]
    retriever = chain_parts["retriever"]

    docs = retriever.invoke(question)
    if not docs:
        yield {
            "reasoning": "",
            "answer": NO_ANSWER_TEXT,
            "source_documents": [],
        }
        return

    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = prompt.format(context=context, question=question)
    messages = [{"role": "user", "content": final_prompt}]

    completion = client.chat.completions.create(
        messages=messages,
        **_completion_kwargs(stream=True),
    )

    reasoning_text = ""
    answer_text = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_text += reasoning
            yield {
                "reasoning": reasoning_text,
                "answer": answer_text,
                "source_documents": docs,
            }
        if delta.content:
            answer_text += delta.content
            yield {
                "reasoning": reasoning_text,
                "answer": answer_text,
                "source_documents": docs,
            }


def stream_chat(messages):
    """在对话模式下对外暴露的流式输出接口。"""
    client = build_qwen_client()
    completion = client.chat.completions.create(
        messages=messages,
        **_completion_kwargs(stream=True),
    )

    reasoning_text = ""
    answer_text = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_text += reasoning
            yield {"reasoning": reasoning_text, "answer": answer_text}
        if delta.content:
            answer_text += delta.content
            yield {"reasoning": reasoning_text, "answer": answer_text}
