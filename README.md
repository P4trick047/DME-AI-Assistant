# 🏥 DME Billing AI Assistant

> A locally-running AI assistant for DME medical billing — powered by LLaMA 3, LangChain RAG, and ChromaDB. **100% free. No API keys. No cloud costs. All data stays on your machine.**

[![CI](https://github.com/YOUR_USERNAME/dme-ai-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/dme-ai-assistant/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LLaMA3](https://img.shields.io/badge/LLM-LLaMA%203-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## 📸 What It Does

| Feature | Description |
|---------|-------------|
| 💬 **Chat Interface** | Streamlit web UI with streaming responses |
| 📚 **RAG Pipeline** | Upload PDFs/Excel/CSV — AI answers from YOUR documents |
| 🔍 **HCPCS Lookup** | Instant code reference with coverage and CMN flags |
| ✅ **Claim Validator** | Catch NPI errors, missing fields, coding issues before submission |
| 🚨 **Denial Analyzer** | CO-4, CO-50, CO-197 and more — with step-by-step remediation |
| 🔬 **Document Analyzer** | Extract structured fields from claims and CMN forms |
| 🧠 **Memory** | Conversation history persists across sessions |

---

## 🏗️ Architecture

```
User Browser
     ↓
Streamlit Web App  (app.py)
     ↓
LangChain Agent   (src/rag_chain.py + src/agent.py)
     ↓
ChromaDB          (src/vector_store.py)  ←  Your billing documents
     ↓
Local LLM         (LLaMA 3 or Mistral via Ollama)
     ↓
AI Response
```

---

## ⚡ Quick Start (5 minutes)

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh   # Mac/Linux
# Windows: https://ollama.ai/download
```

### 2. Pull Models

```bash
ollama pull llama3            # Main model (~4.7 GB)
ollama pull mistral           # Faster alternative (~4.1 GB)
ollama pull nomic-embed-text  # Required for RAG embeddings (~274 MB)
```

### 3. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/dme-ai-assistant.git
cd dme-ai-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

### 4. Run

```bash
# Terminal 1 — keep Ollama running
ollama serve

# Terminal 2 — start the app
streamlit run app.py
```

Open **http://localhost:8501** in your browser. 🎉

---

## 📁 Project Structure

```
dme-ai-assistant/
├── app.py                      # Main Streamlit application
├── requirements.txt
├── .env.example                # Config template (copy to .env)
│
├── src/                        # Core modules
│   ├── document_loader.py      # PDF, Excel, CSV, OCR loading
│   ├── vector_store.py         # ChromaDB management
│   ├── rag_chain.py            # LangChain RAG pipeline
│   ├── agent.py                # LangChain agent with tools
│   ├── billing_tools.py        # HCPCS lookup, claim validator, denial analyzer
│   ├── document_analyzer.py    # Claim field extraction, denial risk scoring
│   └── memory_manager.py       # Persistent conversation memory
│
├── config/
│   ├── settings.py             # Central configuration
│   └── prompts/                # System prompt templates
│       ├── billing_expert.txt
│       └── denial_specialist.txt
│
├── data/                       # Runtime data (gitignored)
│   ├── docs/                   # ← Put your DME documents here
│   ├── chroma_db/              # Vector store (auto-created)
│   └── conversations/          # Chat history (auto-created)
│
├── tests/
│   ├── test_llm.py             # Week 1: Verify Ollama works
│   ├── test_rag.py             # Week 2: Verify RAG pipeline
│   └── test_tools.py           # Week 4: Verify billing tools
│
└── scripts/
    ├── ingest_documents.py     # Batch-index documents
    └── rebuild_index.py        # Wipe and rebuild vector store
```

---

## 📄 Loading Your Documents

### Option A — Via the Web UI
1. Open the app in your browser
2. Use the **Upload Documents** section in the sidebar
3. Drag and drop your PDFs, Excel, or CSV files
4. Click **Index Documents**

### Option B — Batch Load from Folder
```bash
# Put documents in data/docs/ (subfolders supported)
cp /path/to/your/billing/docs/*.pdf data/docs/

# Index everything
python scripts/ingest_documents.py

# Rebuild from scratch (if you changed documents)
python scripts/rebuild_index.py
```

### Recommended Documents to Load

| Document | Source | Priority |
|----------|--------|----------|
| Medicare DMEPOS Fee Schedule | cms.gov/medicare/payment | 🔴 Critical |
| LCD for CPAP (L33718) | cms.gov/medicare-coverage-database | 🔴 Critical |
| LCD for Power Wheelchairs (L33702) | cms.gov/medicare-coverage-database | 🔴 Critical |
| HCPCS Code Reference | cms.gov/medicare/coding-billing | 🔴 Critical |
| Your company billing policies | Internal | 🟡 High |
| Payer-specific coverage policies | Payer portals | 🟡 High |

---

## 🧪 Running Tests

```bash
# Week 1 — Test Ollama connection
python tests/test_llm.py

# Week 2 — Test RAG pipeline (requires Ollama)
python tests/test_rag.py

# Week 4 — Test billing tools (no Ollama needed)
python tests/test_tools.py
```

---

## 🛠️ Configuration

Edit `.env` to customize the app:

```env
DEFAULT_MODEL=llama3          # or mistral
EMBEDDING_MODEL=nomic-embed-text
CHUNK_SIZE=1000               # Larger = more context per chunk
RETRIEVAL_K=4                 # Number of chunks retrieved per query
LLM_MAX_TOKENS=1024
```

---

## 💡 Example Questions to Ask

```
"What are the documentation requirements for CPAP billing?"
"What modifier should I use after the compliance period?"
"Explain HCPCS code E0470"
"How do I appeal a CO-50 denial?"
"What is needed on a CMN for home oxygen?"
"What is the 13-month rental rule?"
"What diagnosis codes cover CPAP under Medicare?"
```

---

## 🚀 30-Day Development Roadmap

| Week | Focus | Outcome |
|------|-------|---------|
| Week 1 | Environment Setup | Ollama + LangChain working locally |
| Week 2 | RAG System | Documents indexed and searchable |
| Week 3 | Chat Interface | Full Streamlit app running |
| Week 4 | Advanced Features | Tools, agents, memory, file upload |

---

## 🔧 Troubleshooting

**Ollama not responding:**
```bash
curl http://localhost:11434/api/tags   # Should return JSON
ollama serve                           # Start if not running
```

**Out of memory:**
```bash
ollama pull phi3:mini    # Smaller model (3.8B, ~2 GB)
# Then change DEFAULT_MODEL=phi3:mini in .env
```

**ChromaDB errors on Windows:**
```bash
pip install chromadb==0.4.22
```

**Slow responses:**
- Switch to `mistral` (faster than llama3)
- Reduce `RETRIEVAL_K=2` in `.env`
- Reduce `LLM_MAX_TOKENS=512` in `.env`

---

## 🌐 SaaS Upgrade Path

```
Phase 1 (Month 2):  Docker + cloud VM (AWS/GCP free tier)
Phase 2 (Month 3):  Multi-user auth + per-user document isolation
Phase 3 (Month 4):  FastAPI backend + React frontend
Phase 4 (Month 5):  Stripe billing + multi-tenant SaaS
```

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## ⚠️ Compliance Note

This tool is for billing workflow assistance only. It does not constitute legal, medical, or compliance advice. Always verify billing decisions with your compliance officer and the applicable payer policies. Patient data (PHI) should be handled in accordance with HIPAA requirements.
