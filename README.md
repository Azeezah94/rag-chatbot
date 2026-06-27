# RAG Chatbot — Document Q&A Agent
**Author:** Azeezat Akinola | [LinkedIn](https://www.linkedin.com/in/azeezat-akinola-710b73113) | [Portfolio](https://Azeezah94.github.io)

A retrieval-augmented generation chatbot that answers questions grounded in your own documents (PDF, TXT, CSV), with source citations for every answer.

## How it works
1. **Ingest** — documents are chunked and embedded using sentence-transformers
2. **Store** — embeddings persist in a local Chroma vector database
3. **Retrieve** — relevant chunks are pulled for each question via semantic search
4. **Generate** — an LLM answers using only the retrieved context, citing sources

## Features
- Multi-format support: PDF, TXT, CSV
- Persistent vector store (no re-ingesting on restart)
- Source citation for every answer
- Streamlit chat UI

## Tech Stack
Python, LangChain, Chroma, HuggingFace Embeddings, Claude (Anthropic), Streamlit

## Setup
```bash
git clone https://github.com/Azeezah94/rag-chatbot
cd rag-chatbot
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"

# CLI version
python rag_chatbot.py

# Web UI version
streamlit run app.py
```

## Example Use Cases
- Chat with a research paper or thesis
- Query company policy documents
- Search across multiple PDFs at once
- Build a personal knowledge base assistant
