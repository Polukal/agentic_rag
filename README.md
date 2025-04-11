# 🧠 Agentic RAG System

This is a specialized **Retrieval-Augmented Generation (RAG)** system designed to interpret and answer **Turkish literature questions**, especially **multiple-choice style (MCQ)** questions based on PDF documents.

The system enables users to:
- Upload their own Turkish literature PDFs (e.g., poetry, essays, practice questions)
- Ask complex, contextual questions
- Get accurate, document-grounded answers via GPT-4
- Run everything through a user-friendly **Streamlit web interface**

🔗 **Live demo** (when deployed): [https://omer-agentic-rag.streamlit.app](https://omer-agentic-rag.streamlit.app)

---

## ✨ Features

- 🧾 Upload your own PDF documents
- 💬 Ask questions about literature, poetry, or authors
- 📘 Answered using GPT-4, grounded in vector search
- 🧠 Based on LangChain + FAISS
- ✅ Uses `pdfplumber` for better Turkish PDF parsing
- 💻 Simple UI powered by Streamlit

---

## 🚀 Deployment on Streamlit Cloud (Free)

1. Fork or clone this repo:  
   📁 [https://github.com/Polukal/agentic_rag](https://github.com/Polukal/agentic_rag)

2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)

3. Click **"New app"**, select your repo

4. Set **main file** to `app.py`

5. Add your OpenAI API key under **Advanced Settings**:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: `sk-...`

6. Click **Deploy**

You'll get a public URL like:
https://your-name-your-app.streamlit.app

---

## 💻 Local Installation

### 1. Clone the project

```bash
git clone https://github.com/Polukal/agentic_rag.git
cd agentic_rag
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a .env file:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 4. Launch the app locally

```bash
python entry.py
```

Your browser will open http://localhost:8501

---

## 🧠 How It Works

1. **PDF Upload**: PDFs are uploaded via the Streamlit UI
2. **Text Extraction**: pdfplumber parses the documents
3. **Chunking**: Text is split into overlapping sections
4. **Embedding**: Chunks are converted into vector embeddings
5. **Storage**: Embeddings are stored in a FAISS index
6. **RAG**: Questions trigger retrieval + GPT-4 synthesis

---

## 🧪 Example Use Case

You can paste questions like this:

```
Fazıl Hüsnü Dağlarca, şiiriyle ilgili yaptığı tanımlamalarda kendisini "yarısı şiir olan bir yaratık" olarak nitelendirmiştir. Bu benzetmeyle ne anlatmak istemektedir?

A) Şiiri yalnızca çocukluk anılarına dayandırdığını  
B) Şiirin kendi kişiliğiyle özdeşleştiğini ve yaşamının ayrılmaz bir parçası olduğunu  
C) Şiiri sadece duygusal bir uğraş olarak gördüğünü  
D) Şiir yazmayı öğretmenlerinden öğrendiğini  
E) Şiiri Tanrı'dan bağımsız düşünemediğini
```

The app will answer based on PDF context and highlight the correct option.

---

## 📂 Project Structure

```
agentic_rag/
├── app.py          # ✅ Streamlit UI
├── entry.py        # 🔁 App launcher (local only)
├── main.py         # PDF/vector logic (optional)
├── data/           # Uploaded PDF files
├── index/          # FAISS vectorstore
├── requirements.txt
├── .env            # API key for local use
└── README.md
```

---

## 📌 Notes

- Only text-based PDFs are supported (not scanned images)
- Works best for literary analysis, poetry interpretation, MCQ reasoning
- Vectorstore is automatically updated on PDF upload

---

## 🛠️ Powered By

- LangChain
- OpenAI
- Streamlit
- FAISS
- pdfplumber

---

## 📬 Contact

Made with ❤️ by @Polukal

Pull requests welcome!