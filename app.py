"""Streamlit Web 应用主程序

提供个人知识库问答 Agent 的 Web 界面，支持两种模式：
1. 知识库问答：上传文档构建向量库，基于检索增强生成（RAG）回答问题
2. AI 对话：纯对话模式，支持多轮对话和思考过程展示
"""
from pathlib import Path

import streamlit as st

from src.config import Settings
from src.document_processor import load_document, split_documents
from src.qa_chain import create_qa_chain, stream_chat, stream_question
from src.vector_store import (
    build_vector_store,
    clear_vector_store,
    delete_document,
    list_documents,
)

UPLOAD_DIR = Path("data/uploads")


def ensure_dirs():
    """确保必要的目录存在，包括上传目录和向量数据库目录。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Path(Settings.get_vector_db_dir()).mkdir(parents=True, exist_ok=True)


def session_api_key() -> str:
    """获取当前会话使用的 API Key：侧边栏输入优先，其次环境变量。

    侧边栏的 Key 只保存在 session_state 中，不写入进程环境变量，
    避免多个会话（多人共用部署）互相覆盖。
    """
    return (st.session_state.get("api_key_input") or "").strip() or Settings.get_api_key()


def flash(message: str):
    """记录一条提示，在下次渲染时显示（用于 rerun 之后的成功提示）。"""
    st.session_state["_flash"] = message


def render_flash():
    """显示并清除暂存的提示。"""
    message = st.session_state.pop("_flash", None)
    if message:
        st.success(message)


def save_upload(uploaded_file) -> str:
    """保存上传的文件到本地目录，返回文件路径。"""
    ensure_dirs()
    filename = Path(uploaded_file.name).name
    if not filename:
        raise ValueError("上传的文件名无效。")
    target = UPLOAD_DIR / filename
    target.write_bytes(uploaded_file.getbuffer())
    return str(target)


def build_knowledge_base(uploaded_file, api_key: str | None = None):
    """从上传的文件构建知识库向量数据库，返回生成的文本块数量。"""
    file_path = save_upload(uploaded_file)
    documents = load_document(file_path)
    chunks = split_documents(documents)
    build_vector_store(chunks, api_key=api_key)
    return len(chunks)


def clear_knowledge_base():
    """清空向量数据库中的全部内容（保留目录，避免 Chroma 客户端缓存失效）。"""
    clear_vector_store()


def clear_upload_cache() -> int:
    """删除 data/uploads 下的缓存文件，返回删除的文件数量。"""
    if not UPLOAD_DIR.exists():
        return 0
    removed = 0
    for item in UPLOAD_DIR.iterdir():
        if item.is_file():
            item.unlink()
            removed += 1
    return removed


def format_size(num_bytes: int) -> str:
    """将字节数格式化为易读的大小。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def render_document_manager():
    """展示知识库中的文档列表，并支持按文件删除。"""
    documents = list_documents()
    with st.expander(f"知识库文档管理（{len(documents)} 个文件）", expanded=False):
        if not documents:
            st.caption("知识库为空，上传文档并构建后会在这里显示。")
            return

        for index, document in enumerate(documents):
            col_name, col_chunks, col_action = st.columns([6, 2, 2])
            col_name.markdown(f"**{document['name']}**")
            col_chunks.caption(f"{document['chunks']} 个文本块")
            if col_action.button(
                "删除",
                key=f"delete_doc_{index}_{document['source']}",
                help="从向量库中删除该文件的全部文本块（不会删除原始上传缓存）",
            ):
                removed = delete_document(document["source"])
                flash(f"已从知识库删除「{document['name']}」，共 {removed} 个文本块。")
                st.rerun()


def render_upload_cache():
    """展示上传缓存信息，并提供清理入口。"""
    files = [item for item in UPLOAD_DIR.iterdir() if item.is_file()] if UPLOAD_DIR.exists() else []
    total = sum(item.stat().st_size for item in files)
    st.caption(f"上传缓存：{len(files)} 个文件，共 {format_size(total)}")
    if st.button("清理上传缓存", disabled=not files):
        removed = clear_upload_cache()
        flash(f"已清理 {removed} 个上传缓存文件。")
        st.rerun()


def check_env() -> bool:
    """检查 API Key 是否已配置，未配置则显示错误信息。"""
    if not session_api_key():
        st.error("未检测到 DASHSCOPE_API_KEY（或 OPENAI_API_KEY），请先配置环境变量或在侧边栏填写后重试。")
        return False
    return True


def render_chat_mode(api_key: str, show_reasoning: bool):
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
            for step in stream_chat(messages, api_key=api_key):
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


def render_qa_mode(api_key: str, show_reasoning: bool):
    """渲染知识库问答模式。"""
    uploaded_file = st.file_uploader("上传文档", type=["pdf", "docx", "md"])
    col_build, col_rebuild = st.columns(2)
    if col_build.button("构建知识库", type="primary"):
        if not uploaded_file:
            st.warning("请先上传文档。")
        else:
            try:
                with st.spinner("正在解析、分块并构建向量库..."):
                    chunk_count = build_knowledge_base(uploaded_file, api_key=api_key)
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
                    chunk_count = build_knowledge_base(uploaded_file, api_key=api_key)
                st.success(f"知识库已重建，共生成 {chunk_count} 个文本块。")
            except Exception as e:
                st.error(f"重建知识库失败：{e}")

    render_document_manager()

    question = st.text_input("输入问题")
    if st.button("提问"):
        if not question.strip():
            st.warning("请输入问题。")
            return
        try:
            chain = create_qa_chain(api_key=api_key)
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


def main():
    """主函数：初始化 Streamlit 应用界面并处理用户交互。"""
    st.set_page_config(page_title="个人知识库问答 Agent", page_icon="🤖", layout="wide")
    st.title("个人知识库问答 Agent")
    st.caption("上传文档后构建向量库，再输入问题进行检索增强问答。")

    with st.sidebar:
        st.subheader("模型配置")
        mode = st.radio("功能模式", ["知识库问答", "AI 对话"], index=0)
        st.text_input(
            "DashScope API Key（可选）",
            type="password",
            key="api_key_input",
            help="仅对当前浏览器会话生效，不会写入环境变量，也不会影响其他用户。",
        )
        show_reasoning = st.toggle("显示思考过程", value=False)
        st.divider()
        st.caption("知识库维护")
        if st.button("清空知识库"):
            try:
                clear_knowledge_base()
                flash("已清空向量库。")
                st.rerun()
            except Exception as e:
                st.error(f"清空知识库失败：{e}")
        render_upload_cache()

    render_flash()

    if not check_env():
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    api_key = session_api_key()

    if mode == "AI 对话":
        render_chat_mode(api_key, show_reasoning)
        return

    render_qa_mode(api_key, show_reasoning)


if __name__ == "__main__":
    main()
