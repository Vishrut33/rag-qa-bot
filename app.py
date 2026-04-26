import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import tempfile
import os

st.title("RAG QA Bot")
st.caption("Upload a PDF and ask questions from it - runs 100% locally")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name
    with st.spinner("Processinf file......"):
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        vectorstore= Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
        )
        retriever = vectorstore.as_retriever(
            search_kwargs={"k":3}
        )
    st.success(f"PDF processed — {len(chunks)} chunks created")

    llm = OllamaLLM(model="mistral")
    prompt = ChatPromptTemplate.from_template("""
    Answer using only the context below.
    If answer not in context, say "I am sorry. I dont know".
    Context: {context}
    Question: {question}
    Answer:""")

    chain = (
        {
            "context": retriever,
            "question": lambda x:x
        }
        | prompt
        | llm
    )

    question = st.text_input("Ask a question")

    if question:
        with st.spinner("Thinking..."):
            response = chain.invoke(question)
        st.markdown(f"Answer: {response}")
    os.unlink(temp_path)