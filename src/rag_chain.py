# ============================================================
# src/rag_chain.py
# Core RAG pipeline: Question → Retrieve → Augment → Generate
# ============================================================

import logging
from typing import Dict, List, Any

from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from src.vector_store import DMEVectorStore
from config.settings import DEFAULT_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_CONTEXT_WINDOW

logger = logging.getLogger(__name__)

# ============================================================
# SYSTEM PROMPT
# This defines the AI's expertise, tone, and behavior.
# Customise this for your specific billing operation.
# ============================================================

DME_SYSTEM_PROMPT = """You are an expert DME (Durable Medical Equipment) medical billing \
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

When answering questions:
1. Be specific — cite billing codes, modifier requirements, or policy references when available
2. If information comes from the provided documents, mention the source filename
3. If you are unsure, say so clearly — accuracy is critical in medical billing
4. Format billing codes clearly (e.g., HCPCS: E0601, Modifier: KX)
5. For compliance or legal questions, recommend consulting a compliance officer
6. Structure longer answers with clear headings when helpful

Use the following context retrieved from billing documentation:
{context}

Chat history:
{chat_history}

Question: {question}

Answer:"""


class DMERAGChain:
    """
    The main RAG pipeline.

    Flow:
      User question
        → vector search for relevant doc chunks
        → inject chunks + question into LLM prompt
        → LLM generates a grounded answer
        → answer + source docs returned
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

        logger.info(f"Initializing LLM: {model_name}")

        self.llm = OllamaLLM(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=LLM_CONTEXT_WINDOW,
        )

        # ConversationBufferMemory stores the last N message pairs
        # so the AI can reference earlier parts of the conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question",
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=DME_SYSTEM_PROMPT,
        )

        self._build_chain(retrieval_k)
        logger.info(f"RAG chain ready — model: {model_name}")

    def _build_chain(self, k: int) -> None:
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.get_retriever(k=k),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            return_source_documents=True,
            verbose=False,
        )

    # ── Public API ───────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question. Returns answer + source documents used.

        Returns:
            {
                "answer":  "The model's response...",
                "sources": [List[Document]],
                "model":   "llama3"
            }
        """
        try:
            response = self.chain.invoke({"question": question})
            return {
                "answer": response["answer"],
                "sources": response.get("source_documents", []),
                "model": self.model_name,
            }
        except ConnectionError:
            return {
                "answer": (
                    "⚠️ Cannot connect to Ollama. "
                    "Please run `ollama serve` in a terminal and try again."
                ),
                "sources": [],
                "model": self.model_name,
            }
        except Exception as e:
            logger.error(f"RAG chain error: {e}", exc_info=True)
            return {
                "answer": f"⚠️ An error occurred: {str(e)}",
                "sources": [],
                "model": self.model_name,
            }

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def switch_model(self, model_name: str) -> None:
        """Hot-swap the underlying LLM (no restart needed)."""
        self.model_name = model_name
        self.llm = OllamaLLM(
            model=model_name,
            temperature=LLM_TEMPERATURE,
            num_predict=LLM_MAX_TOKENS,
            num_ctx=LLM_CONTEXT_WINDOW,
        )
        self._build_chain(k=4)
        logger.info(f"Switched to model: {model_name}")

    def get_chat_history(self) -> List:
        return self.memory.chat_memory.messages
