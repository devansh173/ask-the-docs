# 🧠 Ask the Docs

**Ask the Docs** is a Retrieval-Augmented Generation (RAG) application built using **Streamlit**, allowing users to ask natural language questions over a set of documents. It leverages embeddings and a vector database to provide accurate, context-aware answers. This app is deployed on an AWS EC2 instance.

---

## 📸 Screenshots

> Replace these with actual screenshots and adjust the paths accordingly.

### 🔍 Home Page  
(![Screenshot 2025-05-21 164341](https://github.com/user-attachments/assets/aac26bbf-2265-49cc-a30f-413a2b507a04)


### 📤 Upload Documents  
(![Screenshot 2025-05-21 164929](https://github.com/user-attachments/assets/35189f5b-4b70-4776-afe5-fdeddb8acd0d)


### 💬 Ask Questions  
(![Screenshot 2025-05-21 165927](https://github.com/user-attachments/assets/dcc9c28c-f407-4618-a86c-958a13eed714)


---

## Features - Ask the Docs App

### 🔍 Document Ingestion
- **Upload PDF documents**
  - Users can upload one or more PDF files to the app.
- **Automatic parsing**
  - PDFs are parsed into readable text using `PyPDFLoader`.

### 🤖 Intelligent Question Answering (RAG)
- **Retrieval-Augmented Generation**
  - Combines context retrieval with LLM response generation.
- **Semantic search**
  - User queries are semantically matched against document content using FAISS and Sentence Transformers.

### 🧠 LLM Integration
- **Context-aware answers**
  - Uses pretrained LLMs (e.g., FLAN-T5) to generate accurate responses based on the retrieved chunks.
- **Optimized chunking**
  - Documents are split into manageable chunks for better retrieval and reduced model input length.

### 🌐 Streamlit Web Interface
- **Minimal and intuitive UI**
  - Clean layout with file upload, input box, and answer display.
- **Responsive**
  - Usable from desktop and mobile browsers.

### 💾 Persistent App Deployment
- **Hosted on EC2**
  - Runs on an AWS EC2 instance for 24/7 availability.
- **Elastic IP support**
  - Ensures the app URL remains static and accessible.


---

## Tech Stack - Ask the Docs App

### 🧠 Core Technology
- **Retrieval-Augmented Generation (RAG)**
  - Combines document retrieval and LLM-based generation

### 🌐 Frontend
- **Streamlit**
  - For interactive UI and web app deployment

### 🧪 Backend / Logic
- **LangChain**
  - Document loading, embedding, vector storage, and chaining
- **Hugging Face Transformers**
  - Pretrained models for embeddings and text generation
  - `google/flan-t5-base`

### 📚 Document Handling
- **PyPDFLoader (LangChain)**
  - Parses and loads PDF documents

### 🧠 Embeddings & Vector DB
- **Sentence Transformers**
  - Example: `all-MiniLM-L6-v2`
- **FAISS**
  - For fast vector similarity search

### 🐳 Deployment
- **Amazon EC2 (Ubuntu)**
  - Cloud-hosted server for running the app
- **Elastic IP**
  - To maintain a fixed public IP address
- **tmux / nohup**
  - To keep the app running after SSH logout


---

## 🧑‍💻 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ask-the-docs.git
cd ask-the-docs
```
2. Create Virtual Environment & Activate
```bash

python3 -m venv venv
source venv/bin/activate
```
3. Install Requirements
```bash
pip install -r requirements.txt
```
4. Run the App Locally
```bash

streamlit run app.py
```
To access the app on a public server, use:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```








