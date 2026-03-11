# ============================================================
# src/rag_chain.py
# RAG pipeline using modern LangChain LCEL (v0.2+)
# Replaces deprecated ConversationalRetrievalChain
# ============================================================

import logging
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM

from config.settings import (
    DEFAULT_MODEL,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
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


class DMERAGChain:
    """
    RAG pipeline using LangChain LCEL (v0.2+ compatible).
    Uses in-memory chat history list instead of deprecated ConversationBufferMemory.
    """

    def __init__(
        self,
        vector_store: DMEVectorStore,
        model_name: str = DEFAULT_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        retrieval_k: int = 4,
    ):
        self.vector_store = vector_store
        self.model_name = model_name
        self.retrieval_k = retrieval_k
        self.chat_history: List = []

        logger.info(f"Initialising LLM: {model_name}")
        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=LLM_CONTEXT_WINDOW,
        )
        self._build_chain()
        logger.info(f"RAG chain ready — model: {model_name}")

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
        except ConnectionError:
            return {
                "answer": "⚠️ Cannot connect to Ollama. Run `ollama serve` and try again.",
                "sources": [],
                "model": self.model_name,
            }
        except Exception as e:
            logger.error(f"RAG chain error: {e}", exc_info=True)
            return {
                "answer": f"⚠️ Error: {str(e)}",
                "sources": [],
                "model": self.model_name,
            }

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return {"answer": answer, "sources": sources, "model": self.model_name}

    def clear_memory(self) -> None:
        self.chat_history = []
        logger.info("Conversation memory cleared")

    def switch_model(self, model_name: str) -> None:
        self.model_name = model_name
        self.llm = OllamaLLM(
            model=model_name,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
            num_ctx=LLM_CONTEXT_WINDOW,
        )
        self._build_chain()
        logger.info(f"Switched to model: {model_name}")

    def get_chat_history(self) -> List:
        return self.chat_history
