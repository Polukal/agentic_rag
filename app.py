import os
import streamlit as st
from main import initialize_rag_app, create_agent, log_print
from langchain.callbacks import get_openai_callback
import io
import sys

# Clear the agentic log on every page refresh
with open("agentic_log.txt", "w", encoding="utf-8") as f:
    f.write("")

LOG_FILE = "agentic_log.txt"
DATA_DIR = "data"

st.set_page_config(page_title="🤖 Agentic RAG", layout="wide")
st.title("🧠 Agentic RAG Assistant")
st.markdown(
    "Upload PDFs and ask questions. The assistant will think through and answer based on your documents or its own knowledge."
)

# -- Handle upload
rebuild_needed = False
uploaded_files = st.file_uploader(
    "📤 Upload your PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success("✅ Files uploaded successfully.")
    log_print(
        f"📄 User uploaded {len(uploaded_files)} file(s). Vectorstore will be rebuilt."
    )
    rebuild_needed = True

# -- Init backend
vectorstore = initialize_rag_app(rebuild=rebuild_needed)

if not vectorstore:
    st.warning("📂 Please upload at least one valid PDF to begin.")
else:
    agent = create_agent(vectorstore)

    # -- Question input only if vectorstore exists
    query = st.text_area(
        "📝 Ask your question", placeholder="Type your question here...", height=200
    )

    if st.button("🧠 Let the Agent Think") and query.strip():
        # Reset log at every run
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("")

        with st.spinner("🤔 Thinking..."):
            log_print(f"\n🧠 New User Query: {query}")

            buffer = io.StringIO()
            sys.stdout = buffer
            with get_openai_callback() as cb:
                result = agent.invoke({"input": query})
            sys.stdout = sys.__stdout__

            answer = result["output"] if "output" in result else str(result)
            agent_trace = buffer.getvalue()

            log_print(agent_trace)
            log_print(f"✍️ Final Answer: {answer}")
            log_print(
                f"🧾 Tokens used: {cb.total_tokens} | Prompt: {cb.prompt_tokens} | Completion: {cb.completion_tokens}"
            )

            st.markdown("### ✍️ Final Answer")
            st.write(answer)

# -- Agentic log viewer
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = f.read()
    with st.expander("📜 Agent's Thinking Log", expanded=False):
        st.code(logs, language="text")
