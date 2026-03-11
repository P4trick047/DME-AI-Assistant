# ============================================================
# tests/test_llm.py
# Week 1 verification — confirm Ollama + LangChain is working
# Run: python tests/test_llm.py
# ============================================================

import sys
sys.path.insert(0, ".")

from langchain_ollama import OllamaLLM


def test_llama3():
    print("=" * 55)
    print("🧪 Test 1: LLaMA 3 basic response")
    print("=" * 55)
    llm = OllamaLLM(model="llama3", temperature=0.1, num_predict=200)
    prompt = (
        "You are a DME billing assistant. "
        "In 2 sentences: What is a Certificate of Medical Necessity (CMN)?"
    )
    response = llm.invoke(prompt)
    print(f"Response:\n{response}")
    assert len(response) > 20, "Response too short"
    print("✅ LLaMA 3 — PASS\n")


def test_mistral():
    print("=" * 55)
    print("🧪 Test 2: Mistral basic response")
    print("=" * 55)
    llm = OllamaLLM(model="mistral", temperature=0.1, num_predict=200)
    response = llm.invoke("What is HCPCS code E0601? One sentence answer.")
    print(f"Response:\n{response}")
    assert len(response) > 10, "Response too short"
    print("✅ Mistral — PASS\n")


def test_ollama_connection():
    print("=" * 55)
    print("🧪 Test 3: Ollama connection check")
    print("=" * 55)
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"Available models: {models}")
        print("✅ Ollama is running — PASS\n")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("Run `ollama serve` in a separate terminal\n")


if __name__ == "__main__":
    test_ollama_connection()
    test_llama3()
    test_mistral()
    print("🎉 All LLM tests passed!")
