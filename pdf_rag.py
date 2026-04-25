from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

loader = PyPDFLoader("demo3.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
print(f"Total chunks: {len(chunks)}")

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma.db"
)
print("Stored in ChromaDB")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = OllamaLLM(model="mistral",temperature = 0.9)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using 
only the context provided below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}

Answer:
""")

chain = (
    {
        "context": retriever ,
        "question": lambda x: x
    }
    | prompt
    | llm
)

questions = [
    "What is title of the pdf ?",
    "What is this pdf about ?",
    "What are his key skills ?"
]

for q in questions:
    print(f"\nQ: {q}")
    response = chain.invoke(q)
    print(f"A: {response}")