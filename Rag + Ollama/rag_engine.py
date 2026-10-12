"""
RAG Engine - OFFLINE VERSION
Embeddings: sentence-transformers/all-MiniLM-L6-v2 (local, 384d)
Vector DB: FAISS IndexFlatL2 -> faiss_index.bin
LLM: Ollama qwen2.5-coder:1.5b (local, no API key)
"""

import os
import json
import fitz  # PyMuPDF
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import ollama

# ================= CONFIG =================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dim, 80MB, offline
OLLAMA_MODEL = "qwen2.5-coder:1.5b"  # tomar model

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_JSON_PATH = "chunks.json"

# Load embedding model once
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

# Global FAISS index in RAM
faiss_index = None
chunks_metadata: List[Dict] = []

def recursive_split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple recursive splitter with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:  # ignore tiny chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """Extract text per page"""
    doc = fitz.open(pdf_path)
    pages_data = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        if text.strip():
            pages_data.append({"page_number": i+1, "text": text})
    return pages_data

def build_index_from_file(pdf_path: str):
    global faiss_index, chunks_metadata

    print(f"[RAG] Extracting PDF: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    
    all_chunks = []
    for p in pages:
        split_chunks = recursive_split_text(p["text"])
        for c in split_chunks:
            all_chunks.append({
                "text": c,
                "page_number": p["page_number"],
                "source_file": os.path.basename(pdf_path)
            })
    
    print(f"[RAG] Total chunks: {len(all_chunks)}")
    if not all_chunks:
        raise ValueError("No text found in PDF")

    # Embedding - LOCAL, no API
    texts = [c["text"] for c in all_chunks]
    print(f"[RAG] Creating embeddings (384d) for {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # FAISS Index
    dim = embeddings.shape[1]
    print(f"[RAG] Building FAISS index, dim={dim}")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    faiss_index = index
    chunks_metadata = all_chunks

    print(f"[RAG] Saved: {FAISS_INDEX_PATH} + {CHUNKS_JSON_PATH}")
    return len(all_chunks)

def load_faiss_index():
    global faiss_index, chunks_metadata
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(CHUNKS_JSON_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                chunks_metadata = json.load(f)
            print(f"[RAG] Loaded index: {len(chunks_metadata)} chunks")
            return True
        except Exception as e:
            print(f"[RAG] Load failed: {e}")
            return False
    return False

def search_similar_chunks(question: str, top_k: int = 4) -> List[Dict]:
    if faiss_index is None or not chunks_metadata:
        if not load_faiss_index():
            return []

    print(f"[RAG] Searching for: {question}")
    q_emb = embedding_model.encode([question]).astype('float32')
    distances, indices = faiss_index.search(q_emb, top_k)

    results = []
    for idx in indices[0]:
        if idx < len(chunks_metadata) and idx != -1:
            results.append(chunks_metadata[idx])
    print(f"[RAG] Found {len(results)} relevant chunks")
    return results

def generate_answer_with_ollama(question: str, context_chunks: List[Dict]) -> str:
    """Use local Ollama Qwen Coder to generate answer"""
    if not context_chunks:
        return "No relevant context found. Please upload a PDF first."

    context_text = "\n\n".join([f"[Page {c['page_number']}] {c['text']}" for c in context_chunks])

    prompt = f"""You are a helpful RAG AI Assistant. Answer the user's question ONLY using the provided DOCUMENT CONTEXT.

DOCUMENT CONTEXT:
{context_text}

INSTRUCTIONS:
1. Answer ONLY from context. If answer not in context, say "Information not found in document."
2. Be concise and clear.
3. Mention page numbers like [Page 2] when you use info.
4. If user asks what is in document, summarize.

USER QUESTION: {question}

ANSWER:"""

    try:
        print(f"[OLLAMA] Querying {OLLAMA_MODEL} ...")
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a RAG assistant. Use only provided context."},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.2,
                "num_predict": 500,
            }
        )
        answer = response['message']['content']
        print(f"[OLLAMA] Answer generated: {len(answer)} chars")
        return answer
    except Exception as e:
        print(f"[OLLAMA] Error: {e}")
        return f"Error calling Ollama: {e}. Make sure Ollama is running and model {OLLAMA_MODEL} is pulled. Run: ollama run {OLLAMA_MODEL}"

# Try load on import
load_faiss_index()