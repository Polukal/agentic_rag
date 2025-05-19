import os
import logging
import pdfplumber

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -- Log file
LOG_FILE = "agentic_log.txt"

def log_print(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# -- Silence noise
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.WARNING)

# -- Load PDFs and chunk
def load_and_split_pdfs(pdf_dir):
    docs = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            full_path = os.path.join(pdf_dir, filename)
            with pdfplumber.open(full_path) as pdf:
                log_print(f"📝 Loaded document: {filename} with {len(pdf.pages)} pages.")
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        docs.append(Document(
                            page_content=text,
                            metadata={"source": filename, "page": i + 1}
                        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    if not split_docs:
        log_print("⚠️ No valid text chunks extracted from PDFs.")
    else:
        for i, chunk in enumerate(split_docs[:10]):
            log_print(f"🔹 Chunk {i+1}: {chunk.page_content[:300]}...")

    return split_docs

# -- FAISS vectorstore
INDEX_DIR = "index"

def create_or_load_vectorstore(docs, force_rebuild=False):
    if not docs:
        log_print("⚠️ No valid text extracted from PDFs. Skipping vectorstore creation.")
        return None

    index_path = os.path.join(INDEX_DIR, "index.faiss")

    if os.path.exists(index_path) and not force_rebuild:
        log_print("Loading existing vectorstore...")
        return FAISS.load_local(INDEX_DIR, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        log_print("Creating new vectorstore from uploaded documents...")
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
        vectorstore.save_local(INDEX_DIR)
        return vectorstore

# -- Create tool-using agent
def create_agent(vectorstore):
    tools = []

    if vectorstore:
        def rag_tool_func(query: str) -> str:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            rag_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(temperature=0.2),
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=False
            )
            return rag_chain.run(query)

        rag_tool = Tool(
            name="VectorSearch",
            func=rag_tool_func,
            description=(
                "Use this tool to answer any question based on the uploaded documents. "
                "This includes resumes, CVs, reports, etc. ALWAYS try using this tool first if documents exist."
            )
        )

        tools.append(rag_tool)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful AI assistant. "
            "If PDF documents were uploaded, ALWAYS use tools to search for the answer from those documents. "
            "If no document is uploaded, respond based on your general knowledge."
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# -- Initialize system
def initialize_rag_app(rebuild=False):
    data_dir = "data"
    log_print("📚 Loading and splitting PDF files...")
    docs = load_and_split_pdfs(data_dir)

    log_print("📦 Preparing vector database...")
    vectorstore = create_or_load_vectorstore(docs, force_rebuild=rebuild)

    log_print("🧠 RAG system initialized.")
    return vectorstore
