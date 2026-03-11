#!/bin/bash
# ============================================================
# setup_github.sh
# One-command script to push this project to GitHub
#
# USAGE:
#   chmod +x setup_github.sh
#   ./setup_github.sh https://github.com/YOUR_USERNAME/dme-ai-assistant.git
# ============================================================

set -e  # Exit on any error

REPO_URL=$1

if [ -z "$REPO_URL" ]; then
  echo ""
  echo "❌ ERROR: Please provide your GitHub repository URL"
  echo ""
  echo "Usage:  ./setup_github.sh https://github.com/YOUR_USERNAME/dme-ai-assistant.git"
  echo ""
  echo "Steps to get the URL:"
  echo "  1. Go to https://github.com/new"
  echo "  2. Repository name: dme-ai-assistant"
  echo "  3. Set to Public or Private"
  echo "  4. Do NOT check 'Add a README file'"
  echo "  5. Click 'Create repository'"
  echo "  6. Copy the HTTPS URL shown on the next page"
  echo ""
  exit 1
fi

echo ""
echo "🚀 Setting up Git and pushing to GitHub..."
echo "   Repository: $REPO_URL"
echo ""

# Init git if not already done
if [ ! -d ".git" ]; then
  git init
  echo "✅ Git initialized"
fi

# Set main as default branch
git checkout -b main 2>/dev/null || git checkout main

# Stage all files
git add .

# Create initial commit
git commit -m "🏥 Initial commit — DME Billing AI Assistant

- Streamlit chat interface with RAG pipeline
- LLaMA 3 + Mistral via Ollama (100% local)
- ChromaDB vector store for document retrieval
- PDF, Excel, CSV, OCR document loading
- HCPCS lookup, claim validator, denial analyzer tools
- Persistent conversation memory
- 30-day development roadmap included"

# Add remote origin
git remote remove origin 2>/dev/null || true
git remote add origin "$REPO_URL"

# Push to GitHub
git push -u origin main

echo ""
echo "✅ ✅ ✅  PROJECT PUSHED TO GITHUB SUCCESSFULLY!"
echo ""
echo "   🔗 View your repo at: ${REPO_URL%.git}"
echo ""
echo "📌 Next steps:"
echo "   1. Clone on your machine:  git clone $REPO_URL"
echo "   2. Install Ollama:         https://ollama.ai/download"
echo "   3. Pull models:            ollama pull llama3 && ollama pull nomic-embed-text"
echo "   4. Install deps:           pip install -r requirements.txt"
echo "   5. Run app:                streamlit run app.py"
echo ""
