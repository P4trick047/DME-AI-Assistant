# ============================================================
# src/rag_chain.py
# RAG pipeline — works with Groq (cloud/free) OR Ollama (local)
# Auto-detects which to use based on GROQ_API_KEY in .env
# ============================================================

import logging
import os
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.settings import (
    DEFAULT_MODEL,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    GROQ_API_KEY,
    USE_GROQ,
    GROQ_MODEL,
)
from src.vector_store import DMEVectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert DME (Durable Medical Equipment) medical billing \
assistant with deep knowledge of:

- Medicare and Medicaid DME billing rules and regulations
- HCPCS Level II coding (E codes, K codes, A codes, modifiers)
- Local Coverage Determinations (LCDs) and National Coverage Determinations (NCDs)
- Certificate of Medical Necessity (CMN) forms: CMS-484, CMS-849, CMS-854, CMS-10125
- Prior authorization processes and requirements
- Claim submission requirements and common errors
- Denial management, appeal strategies, and timelines
- ABN (Advance Beneficiary Notice) requirements
- Medicare's capped rental program (13-month rule)
- DMEPOS supplier standards and accreditation

Rules:
1. Be specific — cite billing codes, modifier requirements, or policy references when available.
2. If information comes from the provided context, mention the source filename.
3. If you are unsure, say so — accuracy is critical in medical billing.
4. Format billing codes clearly (e.g., HCPCS: E0601, Modifier: KX).
5. For compliance or legal questions, recommend consulting a compliance officer.

Retrieved context from billing documentation:
{context}"""


def _format_docs(docs) -> str:
    if not docs:
        return "No relevant documents found in the knowledge base."
    parts = []
    for doc in docs:
        fn = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "")
        loc = f" (page {page})" if page else ""
        parts.append(f"[Source: {fn}{loc}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_llm():
    """Return Groq LLM if API key is set, otherwise fall back to Ollama."""
    if USE_GROQ and GROQ_API_KEY:
        from langchain_groq import ChatGroq
        logger.info(f"Using Groq cloud LLM: {GROQ_MODEL}")
        return ChatGroq(
            model=GROQ_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            groq_api_key=GROQ_API_KEY,
        )
    else:
        from langchain_ollama import OllamaLLM
        model = DEFAULT_MODEL
        logger.info(f"Using local Ollama LLM: {model}")
        return OllamaLLM(
            model=model,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
            num_ctx=LLM_CONTEXT_WINDOW,
        )


class DMERAGChain:
    """
    RAG pipeline — auto-switches between Groq (cloud) and Ollama (local).
    Set GROQ_API_KEY in .env or Streamlit secrets to use Groq.
    Leave it empty to use local Ollama.
    """

    def __init__(
        self,
        vector_store: DMEVectorStore,
        model_name: str = DEFAULT_MODEL,
        retrieval_k: int = 4,
    ):
        self.vector_store = vector_store
        self.model_name = GROQ_MODEL if USE_GROQ else model_name
        self.retrieval_k = retrieval_k
        self.chat_history: List = []

        self.llm = _build_llm()
        self._build_chain()
        logger.info(f"RAG chain ready — {'Groq' if USE_GROQ else 'Ollama'}: {self.model_name}")

    def _build_chain(self) -> None:
        self.retriever = self.vector_store.get_retriever(k=self.retrieval_k)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        self.chain = (
            {
                "context": lambda x: _format_docs(self.retriever.invoke(x["question"])),
                "chat_history": lambda x: x["chat_history"],
                "question": lambda x: x["question"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> Dict[str, Any]:
        try:
            sources = self.retriever.invoke(question)
        except Exception:
            sources = []

        try:
            answer = self.chain.invoke({
                "question": question,
                "chat_history": self.chat_history,
            })
        except Exception as e:
            err = str(e)
            logger.error(f"RAG chain error: {err}", exc_info=True)
            if "groq" in err.lower() or "api_key" in err.lower():
                return {
                    "answer": (
                        "⚠️ Groq API error. Check your GROQ_API_KEY in Streamlit secrets.\n"
                        f"Details: {err}"
                    ),
                    "sources": [], "model": self.model_name,
                }
            if "connection" in err.lower() or "ollama" in err.lower():
                return {
                    "answer": "⚠️ Cannot connect to Ollama. Run `ollama serve` locally.",
                    "sources": [], "model": self.model_name,
                }
            return {"answer": f"⚠️ Error: {err}", "sources": [], "model": self.model_name}

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return {"answer": answer, "sources": sources, "model": self.model_name}

    def clear_memory(self) -> None:
        self.chat_history = []

    def switch_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.llm = _build_llm()
        self._build_chain()

    def get_chat_history(self) -> List:
        return self.chat_history
