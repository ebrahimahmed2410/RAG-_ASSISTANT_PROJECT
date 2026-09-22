"""
Streamlit Web Frontend for RAG-Powered Document Assistant
Provides a chat-style UI with grounded answer rendering and source citations.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from api_client import RAGApiClient

# Load environment configuration
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a clean, academic appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .source-card {
        background-color: #F7FAFC;
        border-left: 4px solid #3182CE;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
        font-size: 0.9rem;
    }
    .badge-doc {
        background-color: #EBF8FF;
        color: #2B6CB0;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .badge-page {
        background-color: #EDF2F7;
        color: #4A5568;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .stChatMessage {
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API Client
api_url = os.getenv("API_BASE_URL", "http://localhost:8000")
client = RAGApiClient(base_url=api_url)

# Session State initialization
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    """Renders cited document sources in structured expandable cards."""
    if not sources:
        st.caption("No specific source citations returned.")
        return

    st.markdown("##### 📖 Cited Sources")
    for i, src in enumerate(sources, start=1):
        doc = src.get("document", "Unknown Document")
        page = src.get("page", "?")
        score = src.get("relevance_score")
        snippet = src.get("snippet", "")

        score_text = f" &bull; Relevance: **{score:.1%}**" if score is not None else ""

        with st.expander(f"Source {i}: {doc} (Page {page}){score_text}", expanded=False):
            st.markdown(f"**Document:** `{doc}` &nbsp;|&nbsp; **Page:** `{page}`")
            if snippet:
                st.markdown(f"**Retrieved Passage:**\n> *{snippet}*")


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/education.png", width=110)
    st.title("System Status")
    st.caption(f"Backend Target: `{api_url}`")

    # Health check button & live indicators
    if st.button("🔄 Check Backend Health", use_container_width=True):
        st.session_state.health_info = client.check_health()

    health_info = st.session_state.get("health_info")
    if health_info is None:
        health_info = client.check_health()
        st.session_state.health_info = health_info

    status = health_info.get("status", "unknown")
    components = health_info.get("components", {})
    details = health_info.get("details", {})

    if status == "healthy":
        st.success("● Backend: Online & Ready", icon="✅")
    elif status == "degraded":
        st.warning("● Backend: Degraded (Check components)", icon="⚠️")
    else:
        st.error(f"● Backend: {status.title()}", icon="❌")
        if "error" in health_info:
            st.caption(health_info["error"])

    st.markdown("---")
    st.markdown("##### Component Status")
    vs_ready = components.get("vector_store_initialized", False)
    st.write(f"{'✅' if vs_ready else '❌'} **Vector Store:** {'Loaded' if vs_ready else 'Uninitialized'}")
    
    emb_ready = components.get("embedding_model_loaded", False)
    st.write(f"{'✅' if emb_ready else '❌'} **Embeddings:** {'Ready' if emb_ready else 'Not Loaded'}")

    ollama_ready = components.get("ollama_connected", False)
    st.write(f"{'✅' if ollama_ready else '⚠️'} **Ollama LLM:** {'Connected' if ollama_ready else 'Offline / Fallback'}")

    if details.get("indexed_chunk_count"):
        st.caption(f"Indexed Chunks: **{details['indexed_chunk_count']}**")

    st.markdown("---")
    st.markdown("##### 💡 Example Questions")
    sample_questions = [
        "What is the time complexity of binary search?",
        "What are the four Coffman conditions for deadlocks?",
        "Explain the ACID properties in database transactions.",
        "What is the difference between TCP and UDP?",
        "How does an AVL tree maintain its balance?"
    ]

    for sq in sample_questions:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            st.session_state.preset_query = sq

    st.markdown("---")
    if st.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Chat Interface
st.markdown('<div class="main-header">📚 RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about your Computer Science course documents and receive grounded answers with exact source citations.</div>',
    unsafe_allow_html=True
)

# Display historical messages
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

# Query input
preset = st.session_state.pop("preset_query", None)
prompt = st.chat_input("Ask a question about your documents...")

if preset and not prompt:
    prompt = preset

if prompt:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating grounded answer..."):
            result = client.query(prompt)

        if result.get("success"):
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            st.markdown(answer)
            if sources:
                render_sources(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
        else:
            error_message = result.get("error", "An unknown error occurred.")
            st.error(error_message, icon="⚠️")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ **Error:** {error_message}",
                "sources": []
            })
