# ============================================================
# config/settings.py
# Central configuration — reads from .env file
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma_db"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
UPLOADS_DIR = DATA_DIR / "uploads"

# Create dirs if they don't exist
for d in [DATA_DIR, DOCS_DIR, CHROMA_DIR, CONVERSATIONS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Ollama ───────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
AVAILABLE_MODELS = ["llama3", "mistral", "phi3:mini", "tinyllama"]

# ── Vector Store ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(CHROMA_DIR))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "dme_billing_docs")

# ── Document Processing ──────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 4))
MAX_UPLOAD_SIZE_MB = 50

# ── LLM ─────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
LLM_CONTEXT_WINDOW = 4096

# ── App ──────────────────────────────────────────────────────
APP_TITLE = os.getenv("APP_TITLE", "DME Billing AI Assistant")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "change-me-in-production")

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(DATA_DIR / "app.log"))
