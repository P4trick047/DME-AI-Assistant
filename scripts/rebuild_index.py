#!/usr/bin/env python
# ============================================================
# scripts/rebuild_index.py
# Wipes and rebuilds the entire vector store
# Use when you've updated documents and want a clean re-index
# Run: python scripts/rebuild_index.py
# ============================================================

import sys
sys.path.insert(0, ".")

from scripts.ingest_documents import main
import sys

sys.argv = ["rebuild_index.py", "--reset"]
main()
