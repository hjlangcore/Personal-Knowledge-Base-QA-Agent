# 个人知识库问答 Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)](https://github.com/your-username/your-repo/stargazers)

基于 RAG（Retrieval-Augmented Generation）的智能问答项目，支持上传私有文档构建本地向量库进行检索增强问答，也支持通用 AI 多轮对话。


## 功能特性

- 支持文档上传：`PDF`、`DOCX`、`MD`
- 文档切分：`RecursiveCharacterTextSplitter`
- 向量检索：`Chroma` 本地持久化
- 大模型调用：千问模型（OpenAI 兼容调用）
- 实时流式回答：支持展示/隐藏思考过程
- 双模式：`知识库问答` + `AI 对话`
- 知识库运维：支持清空、重建、清空并重建
- 对话能力：支持多轮历史与清空对话
- 结果可解释：展示参考片段来源与页码（若可用）

## 技术栈

- `Python 3.10+`
- `Streamlit`
- `LangChain` + `langchain-chroma`
- `ChromaDB`
- `OpenAI Python SDK`（用于 DashScope 兼容接口）

## 项目结构

```text
.
├── app.py                    # Streamlit 入口
├── README.md                 # 中文说明（本文件）
├── README_EN.md              # 英文说明
├── requirements.txt
└── src
    ├── config.py             # 配置读取（环境变量）
    ├── document_processor.py # 文档加载与分块
    ├── vector_store.py       # 向量库与 embedding
    └── qa_chain.py           # 检索问答 + 通用对话（含流式）
```

## 快速开始

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 配置环境变量

推荐使用 DashScope Key：

```powershell
$env:DASHSCOPE_API_KEY="你的DashScopeKey"
```

也兼容：

```powershell
$env:OPENAI_API_KEY="你的DashScopeKey"
```

可选配置：

- `OPENAI_BASE_URL`（默认：`https://dashscope.aliyuncs.com/compatible-mode/v1`）
- `CHAT_MODEL`（默认：`qwen3.6-plus`）
- `EMBEDDING_MODEL`（默认：`text-embedding-v4`）
- `VECTOR_DB_DIR`（默认：`data/chroma_db`）
- `EMBEDDING_BATCH_SIZE`（默认：`10`；DashScope `text-embedding-v4` 单次上限为 10）
- `ENABLE_THINKING`（默认：`true`；模型不支持 `enable_thinking` 时可设为 `false`）

### 3) 启动项目

```bash
streamlit run app.py
```

## 使用说明

1. 在侧边栏配置 API Key（可选；会覆盖当前进程环境变量）。
2. 在侧边栏选择功能模式：`知识库问答` 或 `AI 对话`。
3. 使用知识库问答时：上传文档并构建知识库，然后输入问题提问。
   - 重复构建同一文件会先删除该文件的旧文本块，避免重复数据；上传不同文件会累积到同一个知识库。
4. 使用 AI 对话时：直接输入消息进行多轮对话，可清空历史。
5. 可选开启“显示思考过程”（默认关闭，折叠展示）。
6. 通过“清空知识库”或“清空并重建”管理向量库。

## 常见问题

- **401 未授权**
  - 检查 `DASHSCOPE_API_KEY`/`OPENAI_API_KEY` 是否正确。
- **提问时报向量检索错误**
  - 先执行“清空并重建”，再重新提问。
- **回答质量不稳定**
  - 调整 `CHUNK_SIZE`、`CHUNK_OVERLAP` 与 `RETRIEVAL_K`。
- **AI 对话看不到上下文**
  - 仅知识库问答模式会使用向量检索；AI 对话模式为通用聊天。

## 安全提示

- 不要在代码中硬编码 API Key。
- 若 Key 已泄露，请立即在平台侧失效并重新生成。

## 后续规划

- 检索重排（Rerank）
- 问答记录导出
- 更细粒度的错误分类提示
- 自动化测试与 CI
