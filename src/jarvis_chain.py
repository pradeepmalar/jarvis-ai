# src/jarvis_chain.py
import os
from typing import List, Dict, Any

# Vectorstore + embeddings (community)
from langchain_community.vectorstores import FAISS  # used elsewhere when creating vectorstore
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ollama LLM (community)
from langchain_community.llms import Ollama

# Chat history and runnables (core)
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory

# Prompt and parsing
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Utility: simple retriever via the vectorstore interface
# (we do not depend on langchain_community.retrievers.multi_vector)

# ---------------------------
# LLM factory
# ---------------------------
def get_ollama_llm(model_name: str | None = None, base_url: str | None = None, temperature: float = 0.0):
    model_name = model_name or os.getenv("OLLAMA_MODEL", "qwen3-vl:4b")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return Ollama(model=model_name, base_url=base_url, temperature=temperature)

# ---------------------------
# Helper: format chat history into a simple string
# ---------------------------
def render_chat_history(history: List[Dict[str, str]]) -> str:
    """
    history: list of {"role":"user"/"assistant", "content": "..."}
    If you store chat_history as tuples [('You','text'),('Jarvis','text')] adapt accordingly.
    """
    if not history:
        return ""
    lines = []
    for turn in history:
        # support both tuple and dict formats
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = turn.get("content", "")
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            role = "user" if turn[0].lower() in ("you", "user") else "assistant"
            content = turn[1]
        else:
            continue
        prefix = "User:" if role == "user" else "Assistant:"
        lines.append(f"{prefix} {content}")
    return "\n".join(lines)

# ---------------------------
# Create conversational "chain" (callable)
# ---------------------------
def create_conversational_chain(vectorstore, llm):
    """
    Returns a function `run(question, chat_history)`:
      - question: str
      - chat_history: list of (speaker, text) tuples OR list of dicts
    This keeps things simple and compatible with a wide range of LangChain versions.
    """

    # Build a retriever from the vectorstore (works for FAISS and Pinecone vectorstores)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Template used for the LLM — keep it simple and explicit
    template = (
        "You are Jarvis, an expert assistant.\n"
        "You will be given CONTEXT from retrieved documents and the CHAT HISTORY.\n\n"
        "CONTEXT:\n{context}\n\n"
        "CHAT HISTORY:\n{chat_history}\n\n"
        "User question: {question}\n\n"
        "Answer concisely and cite or mention when the answer is not in the context.\n"
        "Jarvis:"
    )
    prompt_template = ChatPromptTemplate.from_template(template)

    # Simple in-memory history factory (not strictly required here, but kept for parity)
    def get_history(session_id: str = "default") -> BaseChatMessageHistory:
        return InMemoryChatMessageHistory()

    # The callable the app will call
    def run(question: str, chat_history=None) -> Dict[str, Any]:
        """
        Returns:
          {
            "answer": "<text>",
            "source_documents": [Document, ...]  # original LangChain Document objects (if available)
          }
        """
        chat_history = chat_history or []
        # 1) Retrieve relevant docs from vectorstore
        docs = retriever.invoke(question)

        # 2) Build context text (concatenate top doc contents)
        # limit length if needed
        context_pieces = []
        for d in docs:
            text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
            # optionally include doc.metadata if helpful
            context_pieces.append(text.strip())
        context = "\n\n---\n\n".join(context_pieces) if context_pieces else "No context retrieved."

        # 3) Render chat history into a single string
        chat_history_text = render_chat_history(chat_history)

        # 4) Fill prompt
        filled_prompt = prompt_template.format(
            context=context,
            chat_history=chat_history_text,
            question=question
        )

        # 5) Call the LLM
        # Most LangChain LLM wrappers are callable: llm(filled_prompt) -> string
        # If your wrapper returns a dict, inspect it and extract text accordingly.
        try:
            try:
    # Try invoking the LLM normally
                llm_output = llm.invoke(filled_prompt) if hasattr(llm, "invoke") else llm(filled_prompt)
    # Handle both string and dict results
                if isinstance(llm_output, dict):
                    answer = llm_output.get("text") or llm_output.get("output") or str(llm_output)
                elif hasattr(llm_output, "content"):
        # Some wrappers return an object with .content
                    answer = llm_output.content
                else:
                    answer = str(llm_output)
    # In case the LLM responded with empty string
                if not answer.strip():
                    answer = "Hmm, I didn’t get a response from the model."
            except Exception as e:
                answer = f"⚠️ Error calling LLM: {e}"

        except TypeError:
            # fallback: some LLM wrappers require `.generate` with specific API; try common patterns
            try:
                gen = llm.generate([filled_prompt])
                # pick the text from the first generation if available
                answer = gen.generations[0][0].text if hasattr(gen, "generations") else str(gen)
            except Exception as e:
                # last resort: return an error-like result for easier debugging
                return {"answer": None, "error": f"LLM call failed: {e}", "source_documents": docs}

        # 6) Return answer + source docs for transparency
        return {"answer": answer, "source_documents": docs}

    # return the callable
    return run
