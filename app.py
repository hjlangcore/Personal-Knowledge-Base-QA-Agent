"""Streamlit Web 应用主程序

提供个人知识库问答 Agent 的 Web 界面，支持两种模式：
1. 知识库问答：上传文档构建向量库，基于检索增强生成（RAG）回答问题
2. AI 对话：纯对话模式，支持多轮对话和思考过程展示
"""
import os
from pathlib import Path

import streamlit as st

from src.config import Settings
from src.document_processor import load_document, split_documents
from src.qa_chain import create_qa_chain, stream_chat, stream_question
from src.vector_store import build_vector_store, clear_vector_store


def ensure_dirs():
    """确保必要的目录存在，包括上传目录和向量数据库目录。"""
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    Path(Settings.get_vector_db_dir()).mkdir(parents=True, exist_ok=True)


def save_upload(uploaded_file) -> str:
    """保存上传的文件到本地目录，返回文件路径。"""
    ensure_dirs()
    filename = Path(uploaded_file.name).name
    if not filename:
        raise ValueError("上传的文件名无效。")
    target = Path("data/uploads") / filename
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def build_knowledge_base(uploaded_file):
    """从上传的文件构建知识库向量数据库，返回生成的文本块数量。"""
    file_path = save_upload(uploaded_file)
    documents = load_document(file_path)
    chunks = split_documents(documents)
    build_vector_store(chunks)
    return len(chunks)


def clear_knowledge_base():
    """清空向量数据库中的全部内容（保留目录，避免 Chroma 客户端缓存失效）。"""
    clear_vector_store()


def check_env():
    """检查 API Key 是否已配置，未配置则显示错误信息。"""
    if not Settings.get_api_key():
        st.error("未检测到 DASHSCOPE_API_KEY（或 OPENAI_API_KEY），请先配置环境变量后重试。")
        return False
    return True


def render_chat_mode(show_reasoning: bool):
    """渲染 AI 对话模式。"""
    st.subheader("AI 对话")
    if st.button("清空对话历史"):
        st.session_state.chat_history = []
        st.success("已清空对话历史。")

    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    user_message = st.chat_input("请输入你想和 AI 对话的内容")
    if not user_message:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    reasoning_placeholder = st.empty()
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        messages = [{"role": "system", "content": "你是一个专业、友好的 AI 助手。"}]
        messages.extend(st.session_state.chat_history)
        final_answer = ""
        try:
            for step in stream_chat(messages):
                if show_reasoning and step["reasoning"]:
                    with reasoning_placeholder.expander("思考过程（模型返回）", expanded=False):
                        st.code(step["reasoning"])
                final_answer = step["answer"]
                answer_placeholder.write(final_answer or "生成中...")
        except Exception as e:
            st.error(f"AI 对话失败：{e}")
            return
        if not final_answer:
            final_answer = "未生成有效回复"
            answer_placeholder.write(final_answer)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": final_answer}
    )


def main():
    """主函数：初始化 Streamlit 应用界面并处理用户交互。"""
    st.set_page_config(page_title="个人知识库问答 Agent", page_icon="🤖", layout="wide")
    st.title("个人知识库问答 Agent")
    st.caption("上传文档后构建向量库，再输入问题进行检索增强问答。")

    with st.sidebar:
        st.subheader("模型配置")
        mode = st.radio("功能模式", ["知识库问答", "AI 对话"], index=0)
        api_key_input = st.text_input(
            "DashScope API Key（可选）",
            type="password",
            help="填写后仅作用于当前进程，可覆盖环境变量。",
        )
        show_reasoning = st.toggle("显示思考过程", value=False)
        if api_key_input:
            os.environ["DASHSCOPE_API_KEY"] = api_key_input
        st.divider()
        if st.button("清空知识库"):
            try:
                clear_knowledge_base()
                st.success("已清空向量库。")
            except Exception as e:
                st.error(f"清空知识库失败：{e}")

    if not check_env():
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if mode == "AI 对话":
        render_chat_mode(show_reasoning)
        return

    uploaded_file = st.file_uploader("上传文档", type=["pdf", "docx", "md"])
    col_build, col_rebuild = st.columns(2)
    if col_build.button("构建知识库", type="primary"):
        if not uploaded_file:
            st.warning("请先上传文档。")
        else:
            try:
                with st.spinner("正在解析、分块并构建向量库..."):
                    chunk_count = build_knowledge_base(uploaded_file)
                st.success(f"知识库构建完成，共生成 {chunk_count} 个文本块。")
            except Exception as e:
                st.error(f"构建知识库失败：{e}")

    if col_rebuild.button("清空并重建"):
        if not uploaded_file:
            st.warning("请先上传文档。")
        else:
            try:
                with st.spinner("正在清空并重建知识库..."):
                    clear_knowledge_base()
                    chunk_count = build_knowledge_base(uploaded_file)
                st.success(f"知识库已重建，共生成 {chunk_count} 个文本块。")
            except Exception as e:
                st.error(f"重建知识库失败：{e}")

    question = st.text_input("输入问题")
    if st.button("提问"):
        if not question.strip():
            st.warning("请输入问题。")
            return
        try:
            chain = create_qa_chain()
            reasoning_placeholder = st.empty()
            answer_placeholder = st.empty()
            latest_result = {"reasoning": "", "answer": "", "source_documents": []}

            with st.spinner("正在检索并流式生成答案..."):
                for step in stream_question(chain, question):
                    latest_result = step
                    if show_reasoning and latest_result["reasoning"]:
                        with reasoning_placeholder.expander("思考过程（模型返回）", expanded=False):
                            st.code(latest_result["reasoning"])
                    answer_placeholder.subheader("答案")
                    answer_placeholder.write(latest_result["answer"] or "生成中...")
        except Exception as e:
            st.error(f"问答失败：{e}")
            return

        sources = latest_result.get("source_documents") or []
        if not latest_result["answer"]:
            answer_placeholder.subheader("答案")
            answer_placeholder.write("未找到相关内容")

        if sources:
            with st.expander("参考片段"):
                for idx, doc in enumerate(sources, start=1):
                    source = doc.metadata.get("source", "")
                    source_name = Path(source).name if source else "未知来源"
                    page = doc.metadata.get("page")
                    page_info = f" | 页码：{page + 1}" if isinstance(page, int) else ""
                    st.markdown(f"**片段 {idx} - 来源：{source_name}{page_info}**")
                    st.write(doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))
                    st.divider()


if __name__ == "__main__":
    main()
