import os

"""配置读取封装

提供从环境变量提取应用运行所需的各种配置信息的静态方法。无需实例化即可访问。"""


class Settings:
    """应用配置访问入口。"""
    @staticmethod
    def get_api_key() -> str:
        """获取 API Key。优先 DASHSCOPE_API_KEY，其次 OPENAI_API_KEY。"""
        return os.getenv("DASHSCOPE_API_KEY", os.getenv("OPENAI_API_KEY", ""))

    @staticmethod
    def get_openai_base_url() -> str:
        """获取 OpenAI/DashScope 的基础 URL。"""
        return os.getenv(
            "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    @staticmethod
    def get_embedding_model() -> str:
        """获取嵌入模型名称。"""
        return os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    @staticmethod
    def get_chat_model() -> str:
        """获取对话/聊天模型名称。"""
        return os.getenv("CHAT_MODEL", "qwen3.6-plus")

    @staticmethod
    def get_vector_db_dir() -> str:
        """获取向量数据库的持久化目录。"""
        return os.getenv("VECTOR_DB_DIR", "data/chroma_db")

    @staticmethod
    def get_chunk_size() -> int:
        """获取文本分块大小（字符数）。"""
        return int(os.getenv("CHUNK_SIZE", "500"))

    @staticmethod
    def get_chunk_overlap() -> int:
        """获取文本分块重叠大小（字符数）。"""
        return int(os.getenv("CHUNK_OVERLAP", "100"))

    @staticmethod
    def get_retrieval_k() -> int:
        """获取向量检索返回的文档数量上限（k）。"""
        return int(os.getenv("RETRIEVAL_K", "4"))

    @staticmethod
    def get_embedding_batch_size() -> int:
        """获取单次嵌入请求的最大文本数（DashScope text-embedding-v4 上限为 10）。"""
        return max(1, int(os.getenv("EMBEDDING_BATCH_SIZE", "10")))

    @staticmethod
    def get_enable_thinking() -> bool:
        """是否开启模型思考模式（部分模型不支持该参数）。"""
        value = os.getenv("ENABLE_THINKING", "true").strip().lower()
        return value in {"1", "true", "yes", "on"}
