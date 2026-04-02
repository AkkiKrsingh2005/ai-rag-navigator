"""
📝 AI RAG Navigator: Intelligent Study Companion
Developed by: Ankit Kumar

This application implements an enterprise-grade Retrieval-Augmented Generation (RAG) 
workflow to transform PDF study materials into interactive conversations. 
Leverages: LangChain v0.3, Google Gemini 1.5 Flash, and ChromaDB.
"""

# ---------------------------------------------------------
# 1. ENVIRONMENT & BACKEND SETUP
# ---------------------------------------------------------
# SQLITE3 MONKEYPATCH: Required for ChromaDB compatibility on Streamlit Community Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI RAG Navigator | Intelligent Study", 
    layout="wide", 
    page_icon="📝"
)

# --- APP HEADER ---
st.title("📝 AI Study Companion: RAG Navigator")
st.markdown("""
    **Transform static notes into interactive AI partners.** 
    This system uses semantic retrieval to ensure high-accuracy answers based *only* on your uploaded materials.
""")

# ---------------------------------------------------------
# 2. SIDEBAR & CONFIGURATION
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Securely handle API Keys (Priority: Secrets > User Input)
    gemini_api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else None
    
    if not gemini_api_key:
        gemini_api_key = st.text_input("Enter Google Gemini API Key:", type="password")

    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    else:
        st.warning("⚠️ API Key required to activate internal LLM reasoning.")
    
    st.divider()
    st.markdown("""
        ### 🚀 Workflow Status
        1. **Ingestion**: Upload PDF documents.
        2. **Vectorization**: Documents are chunked and embedded via Google Vector API.
        3. **Inference**: Chat using context-aware retrieval.
    """)
    st.divider()
    st.caption("Built with LangChain v0.3 & Google Gemini 1.5 Flash")

# ---------------------------------------------------------
# 3. STATE MANAGEMENT
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------------------------------------------------
# 4. DOCUMENT PROCESSING PIPELINE
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Study Materials (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files and gemini_api_key:
    if st.button("🏗️ Build Knowledge Base"):
        with st.spinner("Building vector index for semantic search..."):
            all_pages = []
            for uploaded_file in uploaded_files:
                # Store stream as temporary file for PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                all_pages.extend(pages)
                os.remove(tmp_file_path)

            # Recursive Chunking for optimized context window management
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            document_splits = text_splitter.split_documents(all_pages)

            # Generate Embeddings & Vector Store
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vectorstore = Chroma.from_documents(documents=document_splits, embedding=embeddings)
                st.session_state.vectorstore = vectorstore
                st.success(f"✅ Successfully indexed {len(uploaded_files)} documents!")
            except Exception as e:
                st.error(f"Failed to generate embeddings: {str(e)}")

# ---------------------------------------------------------
# 5. CHAT INTERFACE & RAG INFERENCE
# ---------------------------------------------------------
st.divider()

# Rendering Conversation History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Handling User Query
if prompt := st.chat_input("Query your knowledge base..."):
    if not gemini_api_key:
        st.error("Authentication Error: Please provide a valid API Key in the sidebar.")
    elif not st.session_state.vectorstore:
        st.error("Data Error: Please build the knowledge base before chatting.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Initialize Inference Engines
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

        # --- A. HISTORY-AWARE RETRIEVAL ---
        # Goal: Reformulate user query into a standalone query based on conversation history
        contextualize_q_system_pt = (
            "Given a chat history and the latest user query "
            "formulate a standalone question which can be understood "
            "independently of the history. Do NOT answer the question."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_pt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # --- B. QUESTION ANSWERING CHAIN ---
        # Goal: Synthesize answer using retrieved context and conversation history
        qa_system_prompt = (
            "You are a professional AI study assistant. Answer the question using ONLY the provided context. "
            "If the context doesn't contain the answer, state that you don't know based on the provided notes. "
            "Keep answers structured and professional.\n\n"
            "Context: {context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Assemble Full RAG Chain
        doc_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, doc_chain)

        # Execute Chain & Display Response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context..."):
                result = rag_chain.invoke({
                    "input": prompt, 
                    "chat_history": st.session_state.chat_history
                })
                st.markdown(result["answer"])
                
                # Update Session History
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=result["answer"]),
                ])

st.sidebar.caption("Enterprise RAG v2.6 | Optimized for Python 3.12")
