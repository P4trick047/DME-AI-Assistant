# ============================================================
# app.py  —  DME Billing AI Assistant
# Main Streamlit web application
# Run with:  streamlit run app.py
# ============================================================

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from config.settings import (
    USE_GROQ,
    GROQ_API_KEY,
    GROQ_MODEL,
    APP_TITLE,
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DOCS_DIR,
)
from src.document_analyzer import DMEDocumentAnalyzer
from src.document_loader import DMEDocumentLoader
from src.memory_manager import ConversationMemoryManager
from src.rag_chain import DMERAGChain
from src.vector_store import DMEVectorStore

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/app.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown(
    """
<style>
.header-box {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    padding: 22px 28px;
    border-radius: 12px;
    color: white;
    margin-bottom: 18px;
}
.header-box h1 { margin: 0 0 6px 0; font-size: 1.8rem; }
.header-box p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }
.source-card {
    background: #f0f7ff;
    border-left: 4px solid #2d6a9f;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 4px 0;
    font-size: 0.84rem;
}
.risk-high   { color: #c0392b; font-weight: bold; }
.risk-medium { color: #e67e22; font-weight: bold; }
.risk-low    { color: #27ae60; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE INIT
# ============================================================
def init_state():
    defaults = {
        "messages": [],
        "rag_chain": None,
        "vector_store": None,
        "docs_loaded": False,
        "selected_model": DEFAULT_MODEL,
        "pending_question": None,
        "analyzer": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner="Loading vector store...")
def load_vector_store():
    return DMEVectorStore()


@st.cache_resource(show_spinner="Initialising AI model...")
def build_rag_chain(_vs: DMEVectorStore, model: str):
    return DMERAGChain(vector_store=_vs, model_name=model)


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Model picker
        model = st.selectbox(
            "LLM Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.selected_model)
            if st.session_state.selected_model in AVAILABLE_MODELS
            else 0,
            help="llama3 — best quality | mistral — faster",
        )
        if model != st.session_state.selected_model:
            st.session_state.selected_model = model
            st.session_state.rag_chain = None  # force reload
            st.success(f"Switched to {model}")

        st.markdown("---")
        st.markdown("## 📤 Upload Documents")
        st.caption("PDFs, Excel, CSV, or text files")

        uploaded = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=["pdf", "xlsx", "xls", "csv", "txt", "md"],
        )

        if uploaded and st.button("📥 Index Documents", type="primary", use_container_width=True):
            _process_uploads(uploaded)

        st.markdown("---")
        st.markdown("## 📚 Default Docs")
        st.caption(f"Loads everything from `data/docs/`")
        if st.button("Load Default Docs", use_container_width=True):
            _load_default_docs()

        st.markdown("---")
        st.markdown("## 📊 Stats")
        vs = st.session_state.vector_store
        count = vs.get_document_count() if vs else 0
        col1, col2 = st.columns(2)
        col1.metric("Chunks", count)
        col2.metric("Messages", len(st.session_state.messages))

        st.markdown("---")
        c1, c2 = st.columns(2)
        if c1.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_memory()
            st.rerun()

        if c2.button("♻️ Reset DB", use_container_width=True):
            vs = st.session_state.vector_store
            if vs:
                vs.delete_collection()
                st.session_state.docs_loaded = False
                st.session_state.rag_chain = None
            st.rerun()

        st.markdown("---")
        st.markdown("## 🔖 Quick Ask")
        quick = [
            "What is a CMN form?",
            "Explain modifier KX",
            "CPAP billing requirements",
            "How to appeal a CO-50 denial?",
            "What is HCPCS E0601?",
            "ABN requirements for Medicare",
        ]
        for q in quick:
            if st.button(q, use_container_width=True, key=f"q_{q}"):
                st.session_state.pending_question = q
                st.rerun()


# ============================================================
# DOCUMENT PROCESSING
# ============================================================
def _process_uploads(files):
    loader = DMEDocumentLoader(chunk_size=800, chunk_overlap=150)
    all_docs = []
    bar = st.sidebar.progress(0, text="Processing…")

    for i, f in enumerate(files):
        suffix = Path(f.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        try:
            ext = suffix.lower()
            if ext == ".pdf":
                docs = loader.load_pdf(tmp_path)
            elif ext in {".xlsx", ".xls"}:
                docs = loader.load_excel(tmp_path)
            elif ext == ".csv":
                docs = loader.load_csv(tmp_path)
            else:
                docs = loader.load_text(tmp_path)

            for d in docs:
                d.metadata["filename"] = f.name
            all_docs.extend(docs)
        except Exception as e:
            st.sidebar.error(f"❌ {f.name}: {e}")
        finally:
            os.unlink(tmp_path)

        bar.progress((i + 1) / len(files))

    bar.empty()

    if all_docs:
        chunks = loader.split_documents(all_docs)
        vs = _get_or_create_vs()
        vs.add_documents(chunks)
        st.session_state.docs_loaded = True
        st.session_state.rag_chain = None  # rebuild with new docs
        st.sidebar.success(f"✅ {len(files)} files → {len(chunks)} chunks indexed!")


def _load_default_docs():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*.*")):
        st.sidebar.warning("⚠️ No files found in data/docs/ — add your documents there first")
        return

    loader = DMEDocumentLoader()
    with st.spinner("Loading documents…"):
        docs = loader.load_directory(str(DOCS_DIR))
        if not docs:
            st.sidebar.warning("No supported documents found.")
            return
        chunks = loader.split_documents(docs)
        vs = _get_or_create_vs()
        vs.add_documents(chunks)
        st.session_state.docs_loaded = True
        st.session_state.rag_chain = None
        st.sidebar.success(f"✅ Loaded {len(docs)} docs → {len(chunks)} chunks")


def _get_or_create_vs() -> DMEVectorStore:
    if st.session_state.vector_store is None:
        st.session_state.vector_store = load_vector_store()
    return st.session_state.vector_store


def _get_rag() -> DMERAGChain:
    if st.session_state.rag_chain is None:
        vs = _get_or_create_vs()
        st.session_state.rag_chain = build_rag_chain(vs, st.session_state.selected_model)
    return st.session_state.rag_chain


# ============================================================
# CHAT
# ============================================================
def _format_sources(sources: list) -> str:
    seen, lines = set(), []
    for doc in sources:
        fn = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", "")
        sheet = doc.metadata.get("sheet", "")
        key = f"{fn}-{page}-{sheet}"
        if key not in seen:
            seen.add(key)
            loc = f" (page {page})" if page else (f" (sheet: {sheet})" if sheet else "")
            lines.append(f"📄 {fn}{loc}")
    return "\n".join(lines)


def handle_user_input(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Thinking…"):
            try:
                rag = _get_rag()
                result = rag.ask(user_input)
                answer = result["answer"]
                sources_text = _format_sources(result.get("sources", []))
            except Exception as e:
                answer = f"⚠️ Error: {e}\n\nEnsure Ollama is running: `ollama serve`"
                sources_text = ""

        # Streaming-style display
        placeholder = st.empty()
        shown = ""
        for char in answer:
            shown += char
            placeholder.markdown(shown + "▌")
            time.sleep(0.004)
        placeholder.markdown(answer)

        if sources_text:
            with st.expander("📚 Sources used", expanded=False):
                st.markdown(
                    f'<div class="source-card">{sources_text}</div>',
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_text}
    )


# ============================================================
# ANALYSIS TAB
# ============================================================
def render_analysis_tab():
    st.markdown("### 🔬 Claim & Document Analysis")
    st.caption("Paste claim text or CMN content to extract structured data and assess denial risk.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Extract Claim Fields")
        claim_text = st.text_area(
            "Paste claim or document text here",
            height=220,
            placeholder="Paste a claim, CMN, or billing document here...",
        )

        analysis_type = st.radio(
            "Analysis type",
            ["Claim Fields", "CMN Fields", "Denial Risk"],
            horizontal=True,
        )

        if st.button("🔍 Analyze", type="primary"):
            if not claim_text.strip():
                st.warning("Please paste some text first.")
            else:
                with st.spinner("Analyzing…"):
                    if st.session_state.analyzer is None:
                        st.session_state.analyzer = DMEDocumentAnalyzer(
                            model_name=st.session_state.selected_model
                        )
                    analyzer = st.session_state.analyzer

                    if analysis_type == "Claim Fields":
                        result = analyzer.analyze_claim(claim_text)
                    elif analysis_type == "CMN Fields":
                        result = analyzer.extract_cmn_fields(claim_text)
                    else:
                        extracted = analyzer.analyze_claim(claim_text)
                        result = analyzer.predict_denial_risk(extracted)

                    with col2:
                        st.markdown("#### Analysis Result")
                        if analysis_type == "Denial Risk" and "risk_level" in result:
                            level = result["risk_level"]
                            css = f"risk-{level.lower()}"
                            st.markdown(
                                f'Risk Level: <span class="{css}">{level} ({result["risk_score"]}/100)</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(f"**Recommendation:** {result['recommendation']}")
                            if result["risk_factors"]:
                                st.markdown("**Risk Factors:**")
                                for f in result["risk_factors"]:
                                    st.markdown(f"- {f}")
                        else:
                            st.json(result)


# ============================================================
# MAIN
# ============================================================
def main():
    init_state()
    render_sidebar()

    # Header
    backend = "☁️ Groq (cloud)" if USE_GROQ else "💻 Ollama (local)"
    st.markdown(
        f"""
    <div class="header-box">
        <h1>🏥 {APP_TITLE}</h1>
        <p>Powered by {backend} + RAG — Ask about HCPCS codes, CMNs, denials &amp; billing rules</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Status strip
    c1, c2, c3, c4 = st.columns(4)
    vs = st.session_state.vector_store
    c1.metric("Model", GROQ_MODEL if USE_GROQ else st.session_state.selected_model)
    c2.metric("Knowledge Base", "✅ Loaded" if st.session_state.docs_loaded else "⚠️ Empty")
    c3.metric("Doc Chunks", vs.get_document_count() if vs else 0)
    c4.metric("Messages", len(st.session_state.messages))

    # API key warning
    if not USE_GROQ:
        st.warning(
            "⚠️ **No Groq API key found.** "
            "The app needs a free Groq API key to work on Streamlit Cloud.  \n"
            "1. Get a free key at [console.groq.com](https://console.groq.com)  \n"
            "2. In Streamlit Cloud → your app → **Settings → Secrets** → add: `GROQ_API_KEY = \"gsk_...\"` \n"
            "3. Redeploy the app.",
            icon="🔑",
        )

    st.markdown("---")

    # Tabs
    tab_chat, tab_analysis = st.tabs(["💬 Chat", "🔬 Analyze"])

    with tab_chat:
        if not st.session_state.docs_loaded:
            st.info(
                "👋 **Welcome!**  \n"
                "Upload your DME billing documents in the sidebar (PDF, Excel, CSV, TXT)  \n"
                "or click **Load Default Docs** if you have files in `data/docs/`.  \n\n"
                "The AI can still answer general billing questions without documents, "
                "but adding your specific policies and fee schedules dramatically improves accuracy."
            )

        # Display history
        for msg in st.session_state.messages:
            avatar = "🏥" if msg["role"] == "assistant" else "👤"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📚 Sources", expanded=False):
                        st.markdown(
                            f'<div class="source-card">{msg["sources"]}</div>',
                            unsafe_allow_html=True,
                        )

        # Handle quick-question button presses
        if st.session_state.pending_question:
            q = st.session_state.pending_question
            st.session_state.pending_question = None
            handle_user_input(q)

        # Chat input
        user_input = st.chat_input(
            "Ask about HCPCS codes, CMN requirements, denial reasons, billing rules…"
        )
        if user_input:
            handle_user_input(user_input)

    with tab_analysis:
        render_analysis_tab()


if __name__ == "__main__":
    main()
