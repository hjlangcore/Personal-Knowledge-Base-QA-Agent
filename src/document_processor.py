"""文档处理：加载原始文档并将其分割成文本块以供向量化处理。

添加了模块级注释，描述该模块的职责以及每个函数的行为。"""
from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings


def load_document(file_path: str) -> list[Document]:
    """根据文件后缀加载对应的文档对象列表。"""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()
    if suffix == ".docx":
        return Docx2txtLoader(file_path).load()
    if suffix == ".md":
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}")


def split_documents(documents: list[Document]) -> list[Document]:
    """将文档列表按设置的块大小和重叠进行分块，返回文本块列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Settings.get_chunk_size(),
        chunk_overlap=Settings.get_chunk_overlap(),
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    return splitter.split_documents(documents)
