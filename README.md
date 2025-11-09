# Jarvis AI

Jarvis AI is a local AI assistant built with LangChain, Ollama, and FAISS. It uses Retrieval-Augmented Generation (RAG) to answer questions based on your local data files (PDFs, text, etc.).  
It runs entirely on your machine, ensuring data privacy while providing intelligent, context-aware responses.

---

## Features

- Local, privacy-focused AI assistant
- Retrieval-Augmented Generation using FAISS
- Optional Pinecone vector database support
- Configurable Ollama model integration (e.g., smollm:360m, qwen3-vl:4b)
- Gradio-based chat interface
- Streamlit version available for extended UI customization
- .env configuration for flexible setup

---

## Project Structure

jarvis-ai/
├─ src/
│ ├─ app.py # Main Gradio application
│ ├─ ingest.py # Document loading and text chunking
│ ├─ jarvis_chain.py # Chain logic and model configuration
│ ├─ vectorstore_faiss.py # FAISS vector database logic
│ ├─ vectorstore_pinecone.py # Pinecone vector database logic
├─ data/ # Folder for your source documents
├─ faiss_index/ # Auto-generated FAISS index
├─ .env # Environment configuration (not committed)
├─ .gitignore
├─ requirements.txt
└─ README.md


---

## Installation

1. Clone the Repository

    ```bash
    git clone https://github.com/pradeepmalar/jarvis-ai.git
    cd jarvis-ai


2. Create and Activate a Virtual Environment

    python -m venv venv
    venv\Scripts\activate      # On Windows
    ``` or ```
    source venv/bin/activate   # On Linux/Mac

3. Install Dependencies

    pip install -r requirements.txt

4. Create a .env File

    Create a .env file in the project root and add:

    USE_PINECONE=false
    SHOW_CONTEXT=false
    DATA_DIR=data
    FAISS_DIR=faiss_index
    PORT=7860

    # Ollama configuration
    OLLAMA_MODEL=smollm:360m
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_TEMPERATURE=0.2
    OLLAMA_TIMEOUT=120

    # Pinecone configuration (optional)
    PINECONE_API_KEY=your_real_pinecone_api_key
    PINECONE_ENVIRONMENT=us-east-1
    PINECONE_INDEX_NAME=jarvis-index

    # Embeddings
    EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2


Usage

Run the Gradio App
python src/app.py


Then open your browser at:

http://localhost:7860