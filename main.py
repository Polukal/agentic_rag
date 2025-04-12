import os
import logging
import pdfplumber

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA


# -- Clean terminal output
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)


# -- Load PDFs using pdfplumber (better for Turkish)
def load_and_split_pdfs(pdf_dir):
    docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            full_path = os.path.join(pdf_dir, filename)
            with pdfplumber.open(full_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1},
                            )
                        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


# -- Vectorstore loading/creation with FAISS
INDEX_DIR = "index"


def create_or_load_vectorstore(docs):
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("Loading existing vectorstore...")
        return FAISS.load_local(
            folder_path=INDEX_DIR,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True,
        )
    else:
        print("Creating new vectorstore...")
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
        vectorstore.save_local(INDEX_DIR)
        return vectorstore


# -- RAG chain setup (PDF-based Q&A)
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
        temperature=0.2,
        model="gpt-4",  # or "gpt-3.5-turbo"
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
    )


# -- Direct RAG-only interaction loop
def run_rag_only(rag_chain):
    print("\n🔎 RAG system is ready. You can now ask questions.")
    while True:
        query = input("\n📘 Type your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        response = rag_chain.invoke({"query": query})
        print("\n✍️ Answer:\n", response["result"])
        print("\n📄 Sources:")
        for doc in response["source_documents"]:
            print(f"- {doc.metadata['source']} (sayfa {doc.metadata.get('page', '?')})")

# -- Main script entry
if __name__ == "__main__":
    data_dir = "data"
    print("📚 Loading and splitting PDF files...")
    docs = load_and_split_pdfs(data_dir)

    print("📦 Preparing vector database...")
    vectorstore = create_or_load_vectorstore(docs)

    print("🧠 Starting the RAG (PDF-based question answering) system...")
    rag_chain = build_rag_chain(vectorstore)

    run_rag_only(rag_chain)
