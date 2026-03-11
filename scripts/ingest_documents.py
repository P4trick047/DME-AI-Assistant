#!/usr/bin/env python
# ============================================================
# scripts/ingest_documents.py
# Batch-index all documents from data/docs/ into ChromaDB
# Run: python scripts/ingest_documents.py
#      python scripts/ingest_documents.py --reset   (rebuild from scratch)
# ============================================================

import sys
import argparse
sys.path.insert(0, ".")

from config.settings import DOCS_DIR
from src.document_loader import DMEDocumentLoader
from src.vector_store import DMEVectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest DME billing documents into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Wipe vector store before ingesting")
    parser.add_argument("--dir", type=str, default=str(DOCS_DIR), help="Document directory path")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (characters)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (characters)")
    args = parser.parse_args()

    print(f"📁 Source directory: {args.dir}")

    vs = DMEVectorStore()

    if args.reset:
        print("🗑️ Resetting vector store...")
        vs.delete_collection()

    loader = DMEDocumentLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("📄 Loading documents...")
    documents = loader.load_directory(args.dir)

    if not documents:
        print("⚠️ No documents found. Add files to data/docs/ and try again.")
        sys.exit(1)

    print("✂️ Splitting into chunks...")
    chunks = loader.split_documents(documents)

    print("💾 Indexing into ChromaDB...")
    vs.add_documents(chunks)

    print(f"\n✅ Done! Vector store now has {vs.get_document_count()} chunks.")


if __name__ == "__main__":
    main()
