const API_BASE = "http://localhost:8000";

// DOM Elements
const uploadArea = document.getElementById('upload-area');
const pdfInput = document.getElementById('pdf-file-input');
const statusTag = document.getElementById('pdf-status-tag');
const fileNameEl = document.getElementById('uploaded-filename');
const fileMetaEl = document.getElementById('uploaded-meta');
const completeBadge = document.getElementById('upload-complete-badge');
const uploadContent = document.getElementById('upload-content');
const uploadProcessing = document.getElementById('upload-processing');
const progressFill = document.getElementById('upload-progress-fill');
const pctLabel = document.getElementById('upload-pct-label');
const stepText = document.getElementById('step-text');
const scanLine = document.getElementById('scan-beam-line');
const extractedBox = document.getElementById('extracted-text-box');
const tokensEl = document.getElementById('extracted-stat-tokens');
const ragBadge = document.getElementById('rag-context-badge');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatBox = document.getElementById('chat-messages-box');
const clearBtn = document.getElementById('chat-clear-btn');
const attachBtn = document.getElementById('chat-attach-btn');

let isUploading = false;

console.log("RAG Offline - Qwen2.5 Coder 1.5B Mode");

async function checkStatus(){
    try{
        const res = await fetch(`${API_BASE}/status`);
        const data = await res.json();
        console.log("Status:", data);
        if(data.faiss_loaded){
            statusTag.textContent = `Indexed ${data.chunks_indexed} chunks`;
            statusTag.className = "font-mono text-[11px] text-emerald-400 bg-emerald-950/40 border border-emerald-500/30 px-2 py-0.5 rounded";
            ragBadge.textContent = `Context: ${data.chunks_indexed} chunks (${data.model})`;
            ragBadge.classList.remove('hidden');
            extractedBox.innerHTML = `FAISS loaded: ${data.chunks_indexed} chunks from previous session.<br>Model: ${data.model}`;
            tokensEl.textContent = `Chunks: ${data.chunks_indexed}`;
        }
    }catch(e){
        console.error("Backend not running", e);
        addBotMessage("⚠️ Backend not running. Start: uvicorn main:app --port 8000");
    }
}
checkStatus();

// Upload events
uploadArea.addEventListener('click', ()=> { if(!isUploading) pdfInput.click(); });
attachBtn?.addEventListener('click', ()=> { if(!isUploading) pdfInput.click(); });

uploadArea.addEventListener('dragover', e=> { e.preventDefault(); uploadArea.classList.add('border-primary','bg-primary/10'); });
uploadArea.addEventListener('dragleave', e=> { e.preventDefault(); uploadArea.classList.remove('border-primary','bg-primary/10'); });
uploadArea.addEventListener('drop', e=> { e.preventDefault(); uploadArea.classList.remove('border-primary','bg-primary/10'); const f = e.dataTransfer.files[0]; if(f) handleUpload(f); });
pdfInput.addEventListener('change', e=> { const f = e.target.files[0]; if(f) handleUpload(f); });

async function handleUpload(file){
    if(!file.name.endsWith('.pdf')){ alert("Only PDF"); return; }
    if(file.size > 20*1024*1024){ alert("Max 20MB"); return; }

    isUploading = true;
    uploadArea.classList.add('pulse-border');
    scanLine?.classList.remove('hidden');
    uploadContent.classList.add('opacity-0','pointer-events-none');
    uploadProcessing.classList.remove('opacity-0','pointer-events-none');
    completeBadge?.classList.add('hidden');
    statusTag.textContent = "Embedding locally...";
    stepText.textContent = "Creating embeddings (all-MiniLM-L6-v2)...";
    progressFill.style.width = "40%"; pctLabel.textContent = "40%";

    const fd = new FormData();
    fd.append('file', file);

    try{
        progressFill.style.width = "70%"; pctLabel.textContent = "70%"; stepText.textContent = "Building FAISS index...";
        const res = await fetch(`${API_BASE}/upload-pdf`, {method:"POST", body: fd});
        const data = await res.json();
        if(!res.ok) throw new Error(data.detail || "Upload failed");

        progressFill.style.width = "100%"; pctLabel.textContent = "100%";
        isUploading = false;
        uploadArea.classList.remove('pulse-border');
        scanLine?.classList.add('hidden');
        uploadProcessing.classList.add('opacity-0','pointer-events-none');
        completeBadge?.classList.remove('hidden');
        fileNameEl.textContent = data.filename;
        fileMetaEl.textContent = `${data.chunks_indexed} chunks • ${(data.size/1024/1024).toFixed(2)} MB (Offline)`;
        statusTag.textContent = `Vectorized ${data.chunks_indexed}`;
        statusTag.className = "font-mono text-[11px] text-emerald-400 bg-emerald-950/40 border border-emerald-500/30 px-2 py-0.5 rounded";
        ragBadge.textContent = `Context: ${data.filename} (${data.chunks_indexed}) Qwen Coder`;
        ragBadge.classList.remove('hidden');
        extractedBox.innerHTML = `<span class="text-emerald-400">✅ Indexed offline</span><br>File: ${data.filename}<br>Chunks: ${data.chunks_indexed}<br>Embedding: all-MiniLM-L6-v2 (384d)<br>LLM: qwen2.5-coder:1.5b (Ollama)`;
        tokensEl.textContent = `Chunks: ${data.chunks_indexed}`;
        addBotMessage(`✅ **${data.filename}** indexed offline with ${data.chunks_indexed} chunks.\nEmbedding: all-MiniLM-L6-v2 (local)\nLLM: qwen2.5-coder:1.5b (Ollama)\nNow ask anything!`);

    }catch(err){
        isUploading = false;
        uploadArea.classList.remove('pulse-border');
        uploadProcessing.classList.add('opacity-0','pointer-events-none');
        uploadContent.classList.remove('opacity-0','pointer-events-none');
        scanLine?.classList.add('hidden');
        console.error(err);
        addBotMessage(`❌ Upload failed: ${err.message}`);
        alert(err.message);
    }
}

// Chat
chatForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const q = chatInput.value.trim();
    if(!q) return;
    addUserMessage(q);
    chatInput.value = "";
    chatInput.disabled = true;

    const thinking = addBotMessage("Thinking with Qwen Coder... 🔍 FAISS search + local LLM...", false, true);

    try{
        const res = await fetch(`${API_BASE}/ask`, {
            method:"POST",
            headers:{"Content-Type":"application/json"},
            body: JSON.stringify({question: q, history: []})
        });
        const data = await res.json();
        if(!res.ok) throw new Error(data.detail || "Ask failed");
        thinking.remove();
        addBotMessage(data.answer, true, false, data.sources || []);
    }catch(err){
        thinking.remove();
        addBotMessage(`❌ Error: ${err.message}\nCheck Ollama running: ollama run qwen2.5-coder:1.5b`);
    }finally{
        chatInput.disabled = false;
        chatInput.focus();
    }
});

clearBtn?.addEventListener('click', ()=>{
    chatBox.innerHTML = `<div class="flex gap-md max-w-[85%]"><div class="w-8 h-8 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0"><span class="material-symbols-outlined text-primary text-sm">smart_toy</span></div><div class="bg-[#222226] border border-[#2A2A2E] rounded-xl rounded-tl-sm p-md shadow-sm ai-bubble"><p class="font-body-lg text-body-lg">Chat cleared. Ask about your offline indexed PDF.</p></div></div>`;
});

function addUserMessage(msg){
    const d = document.createElement('div');
    d.className = 'flex gap-md max-w-[85%] self-end flex-row-reverse';
    d.innerHTML = `<div class="w-8 h-8 rounded-full bg-surface-container-high border border-outline-variant/30 flex items-center justify-center shrink-0"><span class="material-symbols-outlined text-on-surface-variant text-sm">person</span></div><div class="bg-primary-container text-on-primary-container rounded-xl rounded-tr-sm p-md shadow-sm"><p class="font-body-lg text-body-lg text-white">${escapeHtml(msg)}</p></div>`;
    chatBox.appendChild(d); chatBox.scrollTop = chatBox.scrollHeight;
}
function addBotMessage(text, hasCitations=false, isThinking=false, sources=[]){
    const d = document.createElement('div');
    d.className = 'flex gap-md max-w-[85%]';
    let cit = '';
    if(sources && sources.length>0){
        cit = `<div class="flex flex-wrap gap-2 mt-3">${sources.map(s=>`<span class="bg-[#131316] rounded px-2 py-1 font-mono text-xs text-secondary/80 border border-outline-variant/20">Page ${s.page||s.page_number||'?'} - ${escapeHtml((s.snippet||'').substring(0,60))}</span>`).join('')}</div>`;
    }
    const display = isThinking ? `<span class="animate-pulse">${escapeHtml(text)}</span>` : parseMarkdown(text);
    d.innerHTML = `<div class="w-8 h-8 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center shrink-0"><span class="material-symbols-outlined text-primary text-sm">smart_toy</span></div><div class="bg-[#222226] border border-[#2A2A2E] rounded-xl rounded-tl-sm p-md shadow-sm ai-bubble"><div class="font-body-lg text-body-lg text-on-surface leading-relaxed">${display}</div>${cit}</div>`;
    chatBox.appendChild(d); chatBox.scrollTop = chatBox.scrollHeight;
    return d;
}
function escapeHtml(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }
function parseMarkdown(t){ if(!t) return ""; t = escapeHtml(t); t = t.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>"); t = t.replace(/\n/g,"<br>"); return t; }
