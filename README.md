# Ask the Docs – Mini RAG App (Open Source)

## 🧠 What It Does

Upload a `.pdf` or `.txt` file and ask questions about its content. It retrieves relevant chunks using FAISS and answers using HuggingFace's Flan-T5 model.

## 💡 Tech Stack

- Streamlit (UI)
- HuggingFace Transformers (LLM)
- SentenceTransformers (Embeddings)
- FAISS (Vector DB)
- LangChain (RAG Chain)
- PyPDF2 (PDF Parsing)

## 🚀 How to Run

```bash
git clone https://github.com/your-username/ask-the-docs
cd ask-the-docs
pip install -r requirements.txt
streamlit run app.py
```

## 🤖 Model Used

- LLM: `google/flan-t5-base`
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`

## ✅ No API key or OpenAI needed!