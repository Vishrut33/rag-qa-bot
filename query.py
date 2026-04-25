from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Step 1 — Load document
loader = TextLoader("data.txt")
documents = loader.load()

# Step 2 — Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

# Step 3 — Embed and store in ChromaDB
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Stored in ChromaDB")

# Step 4 — Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Step 5 — Create RAG chain
llm = OllamaLLM(model="mistral")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using 
only the context provided below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
)

# Step 6 — Ask questions
questions = [
    "What is MLOps?",
    "What tools are used in MLOps?",
    "What is the difference between DevOps and MLOps?"
]

for q in questions:
    print(f"\nQ: {q}")
    response = chain.invoke(q)
    print(f"A: {response}")