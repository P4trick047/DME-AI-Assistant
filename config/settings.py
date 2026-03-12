# ============================================================
# config/settings.py — Central configuration
# Reads from .env (local dev) OR Streamlit secrets (cloud)
# ============================================================

import os
from pathlib import Path

# Load .env when running locally
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _secret(key: str, default: str = "") -> str:
    """Read from env vars first, then Streamlit secrets, then default."""
    val = os.getenv(key, "")
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key, default)
        except Exception:
            val = default
    return val or default


# ── Paths ─────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent.parent
DATA_DIR          = BASE_DIR / "data"
DOCS_DIR          = DATA_DIR / "docs"
CHROMA_DIR        = DATA_DIR / "chroma_db"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
UPLOADS_DIR       = DATA_DIR / "uploads"

for _d in [DATA_DIR, DOCS_DIR, CHROMA_DIR, CONVERSATIONS_DIR, UPLOADS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Groq Cloud LLM (FREE — get key at console.groq.com) ───────
#
# ACTIVE models as of March 2026 (from Groq deprecation docs):
#   llama-3.1-8b-instant      → fast, great for billing Q&A
#   llama-3.3-70b-versatile   → highest quality, best for analysis
#
# DECOMMISSIONED (will cause 400 errors — do NOT use):
#   ✗  llama3-8b-8192          (shutdown Aug 30 2025)
#   ✗  llama3-70b-8192         (shutdown Aug 30 2025)
#   ✗  mixtral-8x7b-32768      (shutdown Mar 20 2025)
#   ✗  gemma2-9b-it            (shutdown Oct 8  2025)
#   ✗  llama3-groq-8b-8192-tool-use-preview  (shutdown Jan 6 2025)
#
GROQ_API_KEY = _secret("GROQ_API_KEY", "")
USE_GROQ     = bool(GROQ_API_KEY)
GROQ_MODEL   = _secret("GROQ_MODEL", "llama-3.1-8b-instant")

AVAILABLE_MODELS = [
    "llama-3.1-8b-instant",      # Fast — recommended for most billing Q&A
    "llama-3.3-70b-versatile",   # Best quality — for complex analysis
]

# ── Ollama (local fallback when GROQ_API_KEY is not set) ───────
OLLAMA_BASE_URL = _secret("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL   = _secret("DEFAULT_MODEL", "llama3")
EMBEDDING_MODEL = _secret("EMBEDDING_MODEL", "nomic-embed-text")

# ── Vector Store ───────────────────────────────────────────────
CHROMA_PERSIST_DIR     = _secret("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
CHROMA_COLLECTION_NAME = _secret("CHROMA_COLLECTION_NAME", "dme_billing_docs")

# ── Document Processing ────────────────────────────────────────
CHUNK_SIZE    = int(_secret("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_secret("CHUNK_OVERLAP", "200"))
RETRIEVAL_K   = int(_secret("RETRIEVAL_K", "4"))

# ── LLM Params ────────────────────────────────────────────────
LLM_TEMPERATURE    = 0.1
LLM_MAX_TOKENS     = 1024
LLM_CONTEXT_WINDOW = 4096

# ── App ────────────────────────────────────────────────────────
APP_TITLE = _secret("APP_TITLE", "DME Billing AI Assistant")
