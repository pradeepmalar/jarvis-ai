import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TensorFlow info logs

from dotenv import load_dotenv
load_dotenv()

from ingest import load_documents, chunk_documents
from vectorstore_faiss import create_faiss, load_faiss
from vectorstore_pinecone import create_pinecone, connect_pinecone
from jarvis_chain import get_ollama_llm, create_conversational_chain

import gradio as gr


# ===============================
# 🔧 Environment Configuration
# ===============================
USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() == "true"
DATA_DIR = os.getenv("DATA_DIR", "data")
FAISS_DIR = os.getenv("FAISS_DIR", "faiss_index")
PORT = int(os.getenv("PORT", "7860"))

# ===============================
# 🧠 Vectorstore Setup
# ===============================
def prepare_vectorstore(force_reindex=False):
    """
    Automatically prepares the correct vectorstore:
    - Pinecone if USE_PINECONE=true
    - FAISS otherwise
    """
    if USE_PINECONE:
        print("🔹 Using Pinecone for vector storage...")
        if force_reindex:
            print("⏳ Rebuilding Pinecone index from documents...")
            docs = load_documents(DATA_DIR)
            chunks = chunk_documents(docs)
            return create_pinecone(chunks)
        else:
            print("📂 Connecting to existing Pinecone index...")
            return connect_pinecone()
    else:
        print("🔹 Using local FAISS vectorstore...")
        if force_reindex or not os.path.exists(FAISS_DIR):
            print("⏳ Building FAISS vectorstore from data folder...")
            docs = load_documents(DATA_DIR)
            if not docs:
                raise SystemExit(f"❌ No documents found in '{DATA_DIR}/' — please add .txt or .pdf files.")
            chunks = chunk_documents(docs)
            return create_faiss(chunks, persist_dir=FAISS_DIR)
        else:
            print("📂 Loading existing FAISS index...")
            return load_faiss(FAISS_DIR)


# ===============================
# 🚀 Initialize Components
# ===============================
print("🚀 Initializing vectorstore and LLM...")
vectorstore = prepare_vectorstore()
llm = get_ollama_llm(model_name=os.getenv("OLLAMA_MODEL", "smollm:360m"))
chain = create_conversational_chain(vectorstore, llm)
print("✅ Jarvis backend ready.")


# ===============================
# 💬 Chatbot Logic
# ===============================
def respond(message, chat_history):
    """
    Handles user messages and generates responses using the chain.
    """
    if not message:
        return chat_history, ""

    try:
        result = chain(message, chat_history)
        answer = result.get("answer") or "Sorry, I couldn’t generate an answer."
        sources = result.get("source_documents", [])
        if sources:
            source_texts = "\n\n".join(
                [f"• {getattr(d, 'page_content', '')[:300]}" for d in sources[:2]]
            )
            answer += f"\n\n📚 *Retrieved context:*\n{source_texts}"
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    chat_history = chat_history or []
    chat_history.append(("You", message))
    chat_history.append(("Jarvis", answer))
    return chat_history, ""


# ===============================
# 🖥️ Gradio Interface
# ===============================
with gr.Blocks(theme="default") as demo:
    gr.Markdown("## 🤖 Jarvis — Your Local AI Assistant (Ollama + FAISS / Pinecone)")
    chatbox = gr.Chatbot(label="Chat with Jarvis", type="messages")
    msg = gr.Textbox(placeholder="Ask Jarvis something...", show_label=False)
    state = gr.State([])

    msg.submit(respond, inputs=[msg, state], outputs=[chatbox, msg])

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=PORT)
