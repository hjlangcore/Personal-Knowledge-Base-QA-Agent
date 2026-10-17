# Personal Knowledge Base QA Agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)](https://github.com/your-username/your-repo/stargazers)

A RAG (Retrieval-Augmented Generation) assistant for both private-document QA and general AI chat.  
It supports document upload, local vector index building, retrieval-augmented answering, and multi-turn chat with Qwen via the DashScope OpenAI-compatible API.


## Features

- Upload support: `PDF`, `DOCX`, `MD`
- Text chunking with `RecursiveCharacterTextSplitter`
- Local vector search with persistent `Chroma`
- Qwen model integration via OpenAI-compatible client
- Streaming answers with optional reasoning display
- Dual modes: `Knowledge Base QA` + `AI Chat`
- Vector DB operations: clear, rebuild, clear-and-rebuild
- Multi-turn chat history with clear-history action
- Source snippets with file and page metadata (when available)

## Tech Stack

- `Python 3.10+`
- `Streamlit`
- `LangChain` + `langchain-chroma`
- `ChromaDB`
- `OpenAI Python SDK` (used against DashScope compatible endpoint)

## Project Structure

```text
.
├── app.py                    # Streamlit entry
├── README.md                 # Chinese README
├── README_EN.md              # English README (this file)
├── requirements.txt
└── src
    ├── config.py             # Env-based configuration
    ├── document_processor.py # Load and split documents
    ├── vector_store.py       # Embedding and vector store logic
    └── qa_chain.py           # Retrieval QA + AI chat (streaming)
```

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment variables

Recommended:

```powershell
$env:DASHSCOPE_API_KEY="your_dashscope_key"
```

Also supported:

```powershell
$env:OPENAI_API_KEY="your_dashscope_key"
```

Optional:

- `OPENAI_BASE_URL` (default: `https://dashscope.aliyuncs.com/compatible-mode/v1`)
- `CHAT_MODEL` (default: `qwen3.6-plus`)
- `EMBEDDING_MODEL` (default: `text-embedding-v4`)
- `VECTOR_DB_DIR` (default: `data/chroma_db`)
- `EMBEDDING_BATCH_SIZE` (default: `10`; DashScope `text-embedding-v4` allows at most 10 per request)
- `ENABLE_THINKING` (default: `true`; set to `false` if the model does not support `enable_thinking`)

### 3) Run the app

```bash
streamlit run app.py
```

## Usage

1. Set API key in sidebar (optional; overrides env var in current process).
2. Choose mode in sidebar: `Knowledge Base QA` or `AI Chat`.
3. In Knowledge Base QA mode, upload documents, build the index, then ask questions.
   - Rebuilding the same file replaces its previous chunks; building different files accumulates them in one knowledge base.
4. In AI Chat mode, send messages directly for multi-turn conversation.
5. Toggle reasoning display if needed (off by default, collapsed view).
6. Use **Clear Knowledge Base** / **Clear and Rebuild** / **Clear Chat History** when needed.

## Troubleshooting

- **401 Unauthorized**
  - Check whether `DASHSCOPE_API_KEY` or `OPENAI_API_KEY` is correctly set.
- **Vector retrieval errors**
  - Clear and rebuild the vector database.
- **Unstable answer quality**
  - Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `RETRIEVAL_K`.
- **AI Chat does not use uploaded docs**
  - This is expected. Only Knowledge Base QA mode uses retrieval from the vector DB.

## Security Notes

- Never hardcode API keys in source code.
- If a key is exposed, revoke it immediately and create a new one.

## Roadmap

- Reranking for retrieval
- QA history export
- Finer-grained error messages
- Automated tests and CI
