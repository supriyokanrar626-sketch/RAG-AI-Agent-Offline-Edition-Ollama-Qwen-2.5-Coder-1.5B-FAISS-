# RAG AI Agent — 100% Offline with Ollama Qwen Coder

> Chat with any PDF completely offline. No API key, no internet after setup.
> **Embeddings:** `all-MiniLM-L6-v2` (local) + **Vector DB:** FAISS + **LLM:** `qwen2.5-coder:1.5b` via Ollama

[Offline](https://img.shields.io/badge/Mode-100%25%20Offline-success) [Ollama](https://img.shields.io/badge/LLM-Ollama%20Qwen%202.5%20Coder%201.5B-black) [FAISS](https://img.shields.io/badge/Vector-FAISS-00BFFF) [No Key](https://img.shields.io/badge/API%20Key-Not%20Needed-brightgreen)

A private RAG dashboard where everything runs on your laptop — your PDF never leaves your device.

## ✨ Features

- **100% Offline** — No API key, no rate limit, no internet needed after setup
- **Private** — PDF stays on your machine
- **Lightweight** — Qwen 1.5B runs even on 4GB RAM
- **Fast Search** — FAISS + 384d local embeddings
- **Grounded Answers** — Answers with [Page X] citations

##  🍂 

<img width="1920" height="1080" alt="Screenshot (103)" src="https://github.com/user-attachments/assets/5662be46-4ce9-417d-9d08-b6d6c08ff4aa" />

## 🏗️ How It Works

```
PDF -> PyMuPDF -> Chunk (1000/200) -> all-MiniLM-L6-v2 (384d) -> FAISS (faiss_index.bin)
Question -> same embedding -> FAISS Top-4 -> Context -> Ollama Qwen Coder -> Answer
```

**2 Local Models:**

1.  **Embedding:** `sentence-transformers/all-MiniLM-L6-v2` — 80MB, for vector search
2.  **LLM:** `qwen2.5-coder:1.5b` — via Ollama, for answer generation

Sentence Transformer is still needed for embeddings. Qwen is only for generating answers.

## 📁 Project Structure

```
Rag/
├── main.py              # FastAPI backend
├── rag_engine.py        # FAISS + Ollama logic
├── requirements.txt     # Python deps (~900MB, torch is biggest)
├── index.html           # Dashboard UI
├── app.js               # Frontend logic
├── faiss_index.bin      # Generated vectors (gitignored)
└── chunks.json          # Generated metadata (gitignored)
```

## 🚀 Setup

### 1. Ollama

```bash
ollama list
ollama pull qwen2.5-coder:1.5b  # skip if already downloaded
ollama run qwen2.5-coder:1.5b   # keep running in separate terminal
```

### 2. Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt  # ~900MB total

uvicorn main:app --port 8000 --host 0.0.0.0 --reload
```

First run downloads `all-MiniLM-L6-v2` (~80MB) automatically.

- API Docs: http://localhost:8000/docs
- Status: http://localhost:8000/status
- Frontend: Open `index.html` with Live Server

Upload PDF → Ask → Get answers with page numbers.

## 🔌 API

- `POST /upload-pdf` — Upload PDF, builds FAISS index
- `POST /ask` — `{question}` → `{answer, sources}`
- `GET /status` — Check indexed chunks

## 🙈 .gitignore

```
__pycache__/
.venv/
faiss_index.bin
chunks.json
uploads/
*.pdf
```

## 📜 Prompt

```
You are a RAG assistant. Answer ONLY from provided context.
If not found, say "Information not found in document."
Mention page numbers like [Page 2].
```

## 📄 License

MIT
