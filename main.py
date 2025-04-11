from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # or use HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
import os

# Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Or use your local model setup
PDF_DIR = "data"
INDEX_DIR = "index"

def load_and_split_pdfs(pdf_dir):
    docs = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, filename))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def create_or_load_vectorstore(docs):
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("Loading existing vectorstore...")
        return FAISS.load_local(INDEX_DIR, OpenAIEmbeddings())
    else:
        print("Creating new vectorstore...")
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
        vectorstore.save_local(INDEX_DIR)
        return vectorstore

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # or any model
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def run_agent(rag_chain):
    tools = [
        Tool(
            name="PDFSearch",
            func=rag_chain.run,
            description="Useful for answering questions about the PDFs"
        )
    ]
    agent = initialize_agent(
        tools, 
        llm=ChatOpenAI(temperature=0), 
        agent="zero-shot-react-description",
        verbose=True
    )

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        response = agent.run(query)
        print("\nAnswer:", response)

if __name__ == "__main__":
    docs = load_and_split_pdfs(PDF_DIR)
    vectorstore = create_or_load_vectorstore(docs)
    rag_chain = build_rag_chain(vectorstore)
    run_agent(rag_chain)
