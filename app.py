"""
Streamlit UI for RAG Chatbot
Run: streamlit run app.py
"""

import streamlit as st
import tempfile
import os
from rag_chatbot import RAGChatbot, DocumentLoader

st.set_page_config(page_title="RAG Chatbot | Azeezat Akinola", page_icon="🤖", layout="wide")
st.title("🤖 Document Q&A Chatbot")
st.caption("Upload documents and ask questions — answers are grounded with source citations")

if "bot" not in st.session_state:
    st.session_state.bot = RAGChatbot()
    st.session_state.ingested = False
    st.session_state.history = []

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or CSV files", accept_multiple_files=True,
        type=["pdf", "txt", "csv"],
    )

    if st.button("Ingest Documents", use_container_width=True) and uploaded_files:
        loader = DocumentLoader()
        all_docs = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for file in uploaded_files:
                path = os.path.join(tmp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                all_docs.extend(loader.load_single_file(path))

        with st.spinner("Building vector store..."):
            st.session_state.bot.ingest(all_docs)
        st.session_state.ingested = True
        st.success(f"Ingested {len(all_docs)} chunks from {len(uploaded_files)} files!")

# Chat interface
if st.session_state.ingested:
    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.bot.ask(question)
            st.write(response["answer"])
            with st.expander("View sources"):
                for src in response["sources"]:
                    st.caption(f"📄 {src['source']}")
                    st.text(src["snippet"] + "...")

        st.session_state.history.append({"role": "assistant", "content": response["answer"]})
else:
    st.info("👈 Upload documents in the sidebar and click 'Ingest Documents' to get started.")
