"""
FastAPI Backend - OFFLINE RAG with Ollama Qwen Coder
No API key needed
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List

import rag_engine

app = FastAPI(title="RAG AI AGENT - Offline Qwen", version="3.0")

# CORS - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AskRequest(BaseModel):
    question: str
    history: List[dict] = []

@app.get("/")
def root():
    return {"message": "RAG AI Agent - Offline (Qwen 2.5 Coder 1.5B) running", "model": rag_engine.OLLAMA_MODEL}

@app.get("/status")
def status():
    loaded = rag_engine.faiss_index is not None
    chunks = len(rag_engine.chunks_metadata) if loaded else 0
    return {
        "status": "ready" if loaded else "no_index",
        "model": rag_engine.OLLAMA_MODEL,
        "embedding_model": rag_engine.EMBEDDING_MODEL_NAME,
        "chunks_indexed": chunks,
        "faiss_loaded": loaded
    }

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[API] File saved: {file_path}")
        num_chunks = rag_engine.build_index_from_file(file_path)

        return {
            "filename": file.filename,
            "chunks_indexed": num_chunks,
            "size": os.path.getsize(file_path),
            "message": f"PDF indexed with {num_chunks} chunks (offline FAISS)"
        }
    except Exception as e:
        print(f"[API] Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question empty")

    if rag_engine.faiss_index is None:
        if not rag_engine.load_faiss_index():
            raise HTTPException(status_code=400, detail="No PDF indexed. Upload PDF first.")

    try:
        # 1. Search FAISS
        relevant_chunks = rag_engine.search_similar_chunks(req.question, top_k=4)
        if not relevant_chunks:
            return {"answer": "No relevant context found in document.", "sources": []}

        # 2. Generate with Ollama Qwen
        answer = rag_engine.generate_answer_with_ollama(req.question, relevant_chunks)

        sources = [{"page": c["page_number"], "snippet": c["text"][:200]} for c in relevant_chunks]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"[API] Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
