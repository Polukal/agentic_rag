import os
import logging
import streamlit as st
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Clean terminal output
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)

# Constants
DATA_DIR = "data"
INDEX_DIR = "index"

# Load and split PDFs
def load_documents_from_folder(folder_path):
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder_path, filename)) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        docs.append(Document(page_content=text, metadata={"source": filename, "page": i + 1}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

# Vectorstore management
@st.cache_resource
def get_vectorstore():
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return FAISS.load_local(INDEX_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        return None

def update_vectorstore(docs):
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    vectorstore.save_local(INDEX_DIR)
    return vectorstore

# RAG Chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0.2, model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True)

# App UI
st.set_page_config(page_title="📚 Agentic RAG Assistant", layout="wide")
st.title("📖 Agentic RAG Assistant")
st.markdown("This app allows you to ask questions to an AI assistant based on the PDF files you upload.")

# File upload section
uploaded_files = st.file_uploader("📤 Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join(DATA_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success("📄 Files uploaded successfully. Vectorstore is being updated...")

    # Re-process and update vectorstore
    documents = load_documents_from_folder(DATA_DIR)
    vectorstore = update_vectorstore(documents)
else:
    vectorstore = get_vectorstore()

# Question section
if vectorstore:
    rag_chain = build_rag_chain(vectorstore)

    query = st.text_area("📝 Enter your question", height=200)
    if st.button("🧠 Cevapla") and query.strip():
        with st.spinner("Cevap aranıyor..."):
            result = rag_chain.invoke({"query": query})
            st.markdown("### ✍️ Answer")
            st.write(result["result"])

            with st.expander("📄 Sources"):
                for doc in result["source_documents"]:
                    st.markdown(f"- `{doc.metadata['source']}` (sayfa {doc.metadata.get('page', '?')})")
else:
    st.warning("Please upload at least one PDF file to begin.")