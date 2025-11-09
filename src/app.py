import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # silence TensorFlow logs

from dotenv import load_dotenv
load_dotenv()

from ingest import load_documents, chunk_documents
from vectorstore_faiss import create_faiss, load_faiss
from jarvis_chain import get_ollama_llm, create_conversational_chain

import gradio as gr


# ------------------------------------------------------
# 1️⃣  Configuration
# ------------------------------------------------------
USE_PINECONE = os.getenv("USE_PINECONE", "false").lower() == "true"
DATA_DIR = "data"
FAISS_DIR = "faiss_index"

# ------------------------------------------------------
# 2️⃣  Prepare vectorstore (FAISS or Pinecone)
# ------------------------------------------------------
def prepare_vectorstore(force_reindex=False):
    """
    Loads documents from ./data, creates FAISS embeddings if needed,
    and returns the vectorstore object.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    if force_reindex or not os.path.exists(FAISS_DIR):
        print("⏳ Building FAISS vectorstore from data folder...")
        docs = load_documents(DATA_DIR)
        if not docs:
            raise SystemExit(f"No files found in {DATA_DIR}/ — please add .txt or .pdf files.")
        chunks = chunk_documents(docs)
        vs = create_faiss(chunks, persist_dir=FAISS_DIR)
        print("✅ FAISS vectorstore built successfully.")
    else:
        print("📂 Loading existing FAISS index...")
        vs = load_faiss(FAISS_DIR)
        print("✅ FAISS index loaded successfully.")

    return vs


# ------------------------------------------------------
# 3️⃣  Initialize components
# ------------------------------------------------------
print("🚀 Initializing vectorstore and LLM...")
vectorstore = prepare_vectorstore()
llm = get_ollama_llm(model_name=os.getenv("OLLAMA_MODEL", "qwen3-vl:4b"))
chain = create_conversational_chain(vectorstore, llm)
print("✅ Jarvis backend ready.")


# ------------------------------------------------------
# 4️⃣  Gradio chat interface
# ------------------------------------------------------
def respond(message, chat_history):
    """
    Called every time the user submits input in Gradio.
    Uses the chain(question, chat_history) callable we built.
    """
    if not message:
        return chat_history, ""

    try:
        result = chain(message, chat_history)  # run our custom chain callable
        answer = result.get("answer") or "Sorry, I couldn’t generate an answer."
        sources = result.get("source_documents", [])
       
    except Exception as e:
        answer = f"⚠️ Error: {e}"

    chat_history = chat_history or []
    chat_history.append(("You", message))
    chat_history.append(("Jarvis", answer))
    return chat_history, ""


# ------------------------------------------------------
# 5️⃣  Launch Gradio UI
# ------------------------------------------------------
with gr.Blocks(theme="default") as demo:
    gr.Markdown("## 🤖 Jarvis — Your Local AI Assistant (Ollama + FAISS)")
    chatbox = gr.Chatbot(label="Chat with Jarvis")
    msg = gr.Textbox(placeholder="Ask Jarvis something...", show_label=False)
    state = gr.State([])

    msg.submit(respond, inputs=[msg, state], outputs=[chatbox, msg])

demo.queue()
demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
