# ============================================================
# config/settings.py
# Central configuration — reads from .env or Streamlit secrets
# ============================================================

import os
from pathlib import Path

# Load .env file if it exists (local dev)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Also read Streamlit secrets if running on Streamlit Cloud
try:
    import streamlit as st
    _secrets = st.secrets
except Exception:
    _secrets = {}

def _get(key: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets, then default."""
    val = os.getenv(key, "")
    if not val and _secrets:
        try:
            val = _secrets.get(key, default)
        except Exception:
            val = default
    return val or default

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
UPLOADS_DIR = DATA_DIR / "uploads"

for d in [DATA_DIR, DOCS_DIR, CHROMA_DIR, CONVERSATIONS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Groq (free cloud LLM) ────────────────────────────────────
GROQ_API_KEY = _get("GROQ_API_KEY", "")
USE_GROQ = bool(GROQ_API_KEY)

# Groq-hosted models (free tier)
# llama3-8b-8192      — fast, great for general use
# llama3-70b-8192     — slower, highest quality
# mixtral-8x7b-32768  — large context window
# gemma2-9b-it        — lightweight alternative
GROQ_MODEL = _get("GROQ_MODEL", "llama3-8b-8192")

# ── Ollama (local fallback) ──────────────────────────────────
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = _get("DEFAULT_MODEL", "llama3")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "nomic-embed-text")
AVAILABLE_MODELS = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]

# ── Vector Store ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = _get("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
CHROMA_COLLECTION_NAME = _get("CHROMA_COLLECTION_NAME", "dme_billing_docs")

# ── Document Processing ──────────────────────────────────────
CHUNK_SIZE = int(_get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "200"))
RETRIEVAL_K = int(_get("RETRIEVAL_K", "4"))

# ── LLM ─────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
LLM_CONTEXT_WINDOW = 4096

# ── App ──────────────────────────────────────────────────────
APP_TITLE = _get("APP_TITLE", "DME Billing AI Assistant")
