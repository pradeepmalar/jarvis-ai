# src/vectorstore_faiss.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_faiss(chunks, persist_dir="faiss_index"):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    faiss_path = os.path.join(persist_dir, "index.pkl")
    with open(faiss_path, "wb") as f:
        pickle.dump(vectorstore, f)
    print("Saved FAISS index to", faiss_path)
    return vectorstore

def load_faiss(persist_dir="faiss_index"):
    import pickle, os
    p = os.path.join(persist_dir, "index.pkl")
    if not os.path.exists(p):
        raise FileNotFoundError("FAISS index not found. Run ingestion first.")
    with open(p, "rb") as f:
        vs = pickle.load(f)
    return vs
