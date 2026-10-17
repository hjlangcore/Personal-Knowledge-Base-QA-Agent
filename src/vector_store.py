"""向量存储与嵌入相关的实现，兼容 DashScope/OpenAI API。"""
from pathlib import Path
import shutil

import chromadb
from openai import OpenAI
from langchain_chroma import Chroma

from src.config import Settings


class DashScopeCompatibleEmbeddings:
    """Use OpenAI-compatible embeddings API with DashScope endpoint."""

    def __init__(self):
        api_key = Settings.get_api_key()
        if not api_key:
            raise ValueError(
                "未检测到 API Key。请设置 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）。"
            )
        self.client = OpenAI(
            api_key=api_key,
            base_url=Settings.get_openai_base_url(),
        )
        self.model = Settings.get_embedding_model()
        self.batch_size = Settings.get_embedding_batch_size()

    def _embed(self, texts):
        """批量请求嵌入，并按返回的 index 还原顺序。"""
        resp = self.client.embeddings.create(model=self.model, input=texts)
        data = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in data]

    def embed_documents(self, texts):
        """将一组文本转换为向量嵌入（按批次请求，减少调用次数）。"""
        contents = [text if isinstance(text, str) else str(text) for text in texts]
        embeddings = []
        for start in range(0, len(contents), self.batch_size):
            batch = contents[start : start + self.batch_size]
            embeddings.extend(self._embed(batch))
        return embeddings

    def embed_query(self, text):
        """将查询文本转换为向量嵌入。"""
        content = text if isinstance(text, str) else str(text)
        return self._embed([content])[0]


def build_embeddings():
    """构建嵌入生成器实例。"""
    return DashScopeCompatibleEmbeddings()


def build_vector_store(chunks):
    """基于文本块构建向量数据库。

    - 同一来源（source）的旧文本块会先被删除，避免重复构建导致重复数据；
    - 不同来源的文本块会保留，便于多次上传不同文档累积知识库。
    """
    if not chunks:
        raise ValueError("文档解析后没有可用的文本内容，请检查文件是否为空或为扫描件。")

    embeddings = build_embeddings()
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=Settings.get_vector_db_dir(),
    )

    sources = {
        doc.metadata.get("source")
        for doc in chunks
        if doc.metadata.get("source")
    }
    for source in sources:
        db.delete(where={"source": source})

    db.add_documents(chunks)
    return db


def clear_vector_store():
    """清空向量库中的所有集合。

    直接删除目录会让 Chroma 的常驻客户端持有失效的文件句柄，
    之后重建会报 "attempt to write a readonly database"，因此改为
    通过 collection API 删除数据，保留目录本身。
    """
    db_dir = Path(Settings.get_vector_db_dir())
    db_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=str(db_dir))
        for collection in list(client.list_collections()):
            name = getattr(collection, "name", collection)
            client.delete_collection(name)
    except Exception:
        # 兜底：清理 Chroma 进程内客户端缓存后重建目录，兼容旧版本/异常状态。
        try:
            from chromadb.api.shared_system_client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        shutil.rmtree(db_dir, ignore_errors=True)
        db_dir.mkdir(parents=True, exist_ok=True)


def load_retriever():
    """加载向量检索器用于相似性检索。"""
    embeddings = build_embeddings()
    db = Chroma(
        embedding_function=embeddings,
        persist_directory=Settings.get_vector_db_dir(),
    )
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Settings.get_retrieval_k()},
    )
