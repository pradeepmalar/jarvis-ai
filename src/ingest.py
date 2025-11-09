# src/ingest.py
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(folder="data"):
    docs = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.lower().endswith(".txt") or fname.lower().endswith(".md"):
            loader = TextLoader(path, encoding="utf8")
            docs.extend(loader.load())
        elif fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
    return docs

def chunk_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Chunks: {len(chunks)}")