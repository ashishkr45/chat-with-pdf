# 📄 Ragified PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using **LangChain**, **FAISS**, and **Chainlit**. Upload any PDF (e.g., your resume), and ask natural language questions to extract relevant info using **Google Gemini** LLM.

![Demo Screenshot](assets/demo-screenshot.png)

---

## 🚀 Features

- ✅ Upload and process any PDF document
- ✅ Text chunking & embeddings using HuggingFace
- ✅ Semantic search via FAISS vector store
- ✅ Chatbot answers questions based on the document
- ✅ Built with Chainlit UI for real-time interaction

---

## 🧱 Tech Stack

- [Python](https://www.python.org/)
- [LangChain](https://docs.langchain.com/)
- [Chainlit](https://docs.chainlit.io/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Gemini (via langchain-google-genai)](https://python.langchain.com/docs/integrations/llms/google_generative_ai/)
- [HuggingFace Embeddings](https://huggingface.co/sentence-transformers)

---

## 📂 Project Structure

ragifiedPDF/
│
├── app.py # Main Chainlit app
├── .env # API keys
├── requirements.txt # Dependencies
└── README.md # This file


---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/ragified-pdf-chatbot.git
cd ragified-pdf-chatbot
```

### 2. Create Virtual Environment

```bash
python3 -m venv pdfBot
source pdfBot/bin/activate  # or `.\pdfBot\Scripts\activate` on Windows
```

### 3. Install 

```bash
pip install -r requirements.txt
```

### 4. Create .env File

```bash 
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 5. Run the App

```bash
chainlit run app.py --host 0.0.0.0 --port 8000
```

## 💡 How it Works
- Upload a PDF → Extracts text via PyPDFLoader

- Chunks the text using RecursiveCharacterTextSplitter

- Converts chunks to vectors using HuggingFace Embeddings

- Stores vectors in a FAISS database

- On user query:

	- Performs semantic similarity search

	- Sends matched chunks to Gemini LLM

	- Displays AI-generated answer