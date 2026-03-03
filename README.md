# 🧠 Personal RAG Assistant: PDF & Text Intelligence

A minimalist, high-performance **Retrieval-Augmented Generation (RAG)** engine built from scratch. This project allows users to chat with their local documents (PDFs and TXT) using local embeddings and the ultra-fast Llama 3.3 model via Groq.

## 🚀 Key Features
- **Local Vector Search:** Uses `sentence-transformers` for privacy and `NumPy` for efficient similarity calculations.
- **Smart Chunking:** Implements recursive text splitting with overlap to preserve semantic context.
- **Multi-Format Support:** Native parsing for `.pdf` and `.txt` files.
- **Interactive Web UI:** Modern chat interface powered by **Streamlit**.
- **Edge Inference:** Powered by **Groq Cloud** (Llama-3.3-70b-versatile) for near-instant responses.

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **LLM:** Llama 3.3 (via Groq API)
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector Math:** NumPy (Cosine Similarity/Dot Product)
- **Frontend:** Streamlit

## 📐 How it Works (The RAG Pipeline)

1. **Ingestion:** Documents are loaded from the `/data` folder.
2. **Chunking:** Text is split into 1000-character segments with a 200-character overlap.
3. **Embedding:** Each chunk is converted into a high-dimensional vector.
4. **Retrieval:** When a user asks a question, we calculate the dot product between the query vector and all document vectors:
   $$A \cdot B = \sum_{i=1}^{n} A_i B_i$$
5. **Augmentation:** The most relevant chunk is injected into a specialized prompt.
6. **Generation:** The LLM generates a response based *only* on the provided context.

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/adrianizmi/Simple-RAG.git](https://github.com/adrianizmi/Simple-RAG.git)
   cd Simple-RAG
   mkdir data
   python -m venv venv
# Windows:
   .\venv\Scripts\activate
# Mac/Linux:
   source venv/bin/activate

   pip install -r requirements.txt

> [!IMPORTANT]
> **API Key Required:** To run this project, you need a Groq API Key. 
> You can get one for free at [console.groq.com](https://console.groq.com/).