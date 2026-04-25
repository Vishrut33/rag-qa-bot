# RAG QA Bot

Ask questions from any PDF using local LLM — no API keys, no internet.

## Stack
- LangChain + ChromaDB + Mistral (via Ollama)
- Sentence Transformers (all-MiniLM-L6-v2)
- Runs 100% locally

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Add your PDF as `document.pdf`
3. Run: `python pdf_rag.py`

## How it works
PDF → split into chunks → embed → store in ChromaDB → 
question → retrieve similar chunks → Mistral answers
