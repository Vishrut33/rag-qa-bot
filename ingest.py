from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("demo.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./chroma_db")
vectordb.persist()

print(f'Stored {len(chunks)} chunks in ChromaDB')