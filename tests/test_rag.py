# ============================================================
# tests/test_rag.py
# Week 2 verification — test the full RAG pipeline end-to-end
# Run: python tests/test_rag.py
# ============================================================

import sys
sys.path.insert(0, ".")

from pathlib import Path
from src.document_loader import DMEDocumentLoader
from src.vector_store import DMEVectorStore
from src.rag_chain import DMERAGChain


SAMPLE_DIR = Path("data/test_sample_docs")


def create_sample_docs():
    """Create minimal DME billing sample documents for testing."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    (SAMPLE_DIR / "cpap_policy.txt").write_text("""
DME BILLING GUIDE — CPAP (E0601)

HCPCS Code E0601: Continuous Positive Airway Pressure Device
Medicare Coverage: Covered for obstructive sleep apnea (OSA) with AHI >= 15.

Required Documentation:
- Written physician order
- Sleep study results (polysomnography or HST)
- Certificate of Medical Necessity (CMN) Form CMS-484
- Face-to-face encounter within 6 months prior to sleep study

Modifiers:
- RR: Rental equipment
- KX: After month 3 compliance confirmed (>= 4 hrs/night, >= 70% of nights)
- GA: ABN on file

Common Denials:
- Missing or unsigned CMN
- Compliance data not downloaded at month 3 visit
- Sleep study AHI below 15 (without other qualifying criteria)
""")

    (SAMPLE_DIR / "hcpcs_reference.txt").write_text("""
HCPCS DME CODE QUICK REFERENCE

E0601 - CPAP Device
E0470 - BiPAP without backup rate  
E0471 - BiPAP with backup rate
K0001 - Standard Manual Wheelchair
K0005 - Ultralightweight Wheelchair (< 17 lbs)
E0143 - Walker, folding wheeled
E1390 - Oxygen Concentrator
A4253 - Blood Glucose Test Strips per 50

Modifier Reference:
RR - Rental equipment
KX - Medical necessity requirements documented
GA - ABN issued to beneficiary
GZ - Item expected to be denied (no ABN)
NU - New equipment purchased
""")
    print(f"✅ Sample docs created in {SAMPLE_DIR}")


def test_document_loading():
    print("=" * 55)
    print("🧪 Test 1: Document loading")
    print("=" * 55)
    loader = DMEDocumentLoader(chunk_size=400, chunk_overlap=80)
    docs = loader.load_directory(str(SAMPLE_DIR))
    assert len(docs) > 0, "No documents loaded!"
    print(f"Loaded {len(docs)} documents")

    chunks = loader.split_documents(docs)
    assert len(chunks) >= len(docs), "Chunks should be >= source docs"
    print(f"Split into {len(chunks)} chunks")
    print("✅ Document loading — PASS\n")
    return chunks


def test_vector_store(chunks):
    print("=" * 55)
    print("🧪 Test 2: Vector store indexing and retrieval")
    print("=" * 55)
    vs = DMEVectorStore(
        persist_directory="./data/test_chroma",
        collection_name="test_collection",
    )
    vs.delete_collection()  # fresh start
    vs.add_documents(chunks)
    assert vs.get_document_count() > 0

    results = vs.similarity_search("CPAP compliance requirements", k=2)
    assert len(results) > 0, "No results returned!"
    print(f"Query returned {len(results)} relevant chunks")
    print(f"Top result preview: {results[0].page_content[:100]}...")
    print("✅ Vector store — PASS\n")
    return vs


def test_rag_chain(vs):
    print("=" * 55)
    print("🧪 Test 3: Full RAG chain")
    print("=" * 55)
    rag = DMERAGChain(vector_store=vs, model_name="llama3")

    questions = [
        "What modifier is needed after the CPAP compliance period?",
        "What is the HCPCS code for a standard manual wheelchair?",
        "What documentation is required for CPAP billing?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        result = rag.ask(q)
        print(f"💬 {result['answer'][:200]}...")
        assert len(result["answer"]) > 10, "Answer too short!"

    print("\n✅ RAG chain — PASS\n")


if __name__ == "__main__":
    print("🚀 Running RAG Pipeline Tests\n")
    create_sample_docs()
    chunks = test_document_loading()
    vs = test_vector_store(chunks)
    test_rag_chain(vs)
    print("🎉 All RAG tests passed!")
