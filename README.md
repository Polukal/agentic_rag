# Agentic RAG System

This repository contains a simple yet powerful agentic RAG (Retrieval-Augmented Generation) system that allows you to query PDF documents using natural language. The system combines document retrieval with a language model agent to provide accurate answers based on the content of your PDFs.

## Features

- PDF document loading and chunking
- Vector embeddings for efficient semantic search
- Persistent vector storage for quick reloading
- LLM-powered question answering
- Agent-based reasoning for complex queries

## Requirements

- Python 3.8+
- OpenAI API key (or alternative for local models)
- Required packages (see Installation)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Polukal/agentic_rag.git
   cd agentic-rag
   ```

2. Install the required packages:
   ```bash
   pip install langchain openai faiss-cpu pypdf tiktoken
   ```

3. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage

1. Place your PDF files in the `data/` directory.

2. Create directories if they don't exist:
   ```bash
   mkdir -p data index
   ```

3. Run the main script:
   ```bash
   python main.py
   ```

4. Ask questions about your PDFs when prompted. Type 'exit' to quit.

## How It Works

1. **Document Loading**: The system loads PDF documents from the specified directory.
2. **Text Splitting**: Documents are split into manageable chunks with overlap for context preservation.
3. **Vectorization**: Text chunks are converted to vector embeddings using OpenAI's embedding model.
4. **Indexing**: Vectors are indexed using FAISS for efficient retrieval.
5. **Retrieval**: When a query is submitted, semantically similar chunks are retrieved.
6. **Agent Reasoning**: An LLM agent uses the retrieved information to formulate an answer.

## Customization

- Adjust chunk size and overlap in `RecursiveCharacterTextSplitter`
- Change the number of retrieved documents with `search_kwargs={"k": 4}`
- Use different language models by changing the `ChatOpenAI` model parameter
- Implement HuggingFaceEmbeddings for local embedding models

## Project Structure

```
agentic_rag/
│
├── main.py             # Main application code
├── data/               # Directory for PDF files
│   ├── your_large_1.pdf
│   └── your_large_2.pdf
├── index/              # Directory for vector store persistence
│   └── (FAISS index files)
```

## Limitations

- Works best with text-based PDFs (not scanned documents)
- Performance depends on the quality of the PDF parsing
- May require API key management for production use
- Vector store can grow large with many documents

## Future Improvements

- Support for more document types (DOCX, TXT, etc.)
- Document preprocessing and cleaning
- Web UI for easier interaction
- Multi-modal support for images within PDFs
- Batch processing for large document collections
