# RAG QA Bot 🤖

An intelligent document Q&A system powered by RAG (Retrieval-Augmented Generation).
Upload any PDF and ask questions — get accurate answers instantly.

🚀 **Live Demo**: http://13.201.74.28:8501

## What it does

Upload a PDF document and ask natural language questions about it.
The system retrieves the most relevant sections and generates precise answers.

## Tech Stack

| Component | Technology |
|---|---|
| LLM | LLaMA3 via Groq API |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Framework | LangChain |
| UI | Streamlit |
| Deployment | AWS EC2 |

## Architecture 

PDF Upload
→ Text Extraction (PyPDF)
→ Chunking (RecursiveCharacterTextSplitter)
→ Embeddings (Sentence-Transformers)
→ Vector Storage (ChromaDB)
→ Similarity Search
→ LLM Generation (LLaMA3 via Groq)
→ Answer

## Run Locally

**Prerequisites**
- Python 3.10+
- Groq API key (free at console.groq.com)

**Setup**
```bash
git clone https://github.com/Vishrut33/rag-qa-bot.git
cd rag-qa-bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Configure**
```bash
# Add your Groq API key in app.py
api_key="your_groq_api_key_here"
```

**Run**
```bash
streamlit run app.py
```

## Features

- 📄 PDF upload and processing
- 🔍 Semantic search across document chunks
- 🤖 LLaMA3 powered answer generation
- ⚡ Sub-second retrieval
- 🔒 Password protected demo

## Project Structure

rag-qa-bot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Dependencies
└── README.md
## Author

**Vishrut Ghotge**
AI/MLOps Engineer | Bangalore

[LinkedIn](linkedin.com/in/vishrutg) • [GitHub](https://github.com/Vishrut33)
