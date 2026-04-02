import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Set up page config
st.set_page_config(page_title="Talk to Your Notes | AI-RAG", layout="wide", page_icon="📝")

# Header
st.title("📝 AI Study Companion: Talk to Your Notes")
st.markdown("""
    Upload your PDFs and chat with them! This application uses **Retrieval-Augmented Generation (RAG)** 
    to provide accurate answers based on your study materials.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    # Priority: Streamlit Secrets > Sidebar Input
    gemini_api_key = st.secrets.get("GOOGLE_API_KEY") if "GOOGLE_API_KEY" in st.secrets else None
    
    if not gemini_api_key:
        gemini_api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    else:
        st.success("API Key loaded from Secrets!")

    if gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
    else:
        st.warning("Please enter your API Key to continue.")
    
    st.divider()
    st.markdown("""
        ### How it works:
        1. **Upload** your PDF documents.
        2. **Process** documents into segments.
        3. **Chat** with AI using the context from your notes.
    """)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File uploader
uploaded_files = st.file_uploader("Choose your study materials (PDFs)", type="pdf", accept_multiple_files=True)

if uploaded_files and gemini_api_key:
    if st.button("🚀 Process Documents"):
        with st.spinner("Processing documents..."):
            all_pages = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load()
                all_pages.extend(pages)
                os.remove(tmp_file_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_pages)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            st.session_state.vectorstore = vectorstore
            st.success(f"Processed {len(uploaded_files)} documents!")

# Chat Interface
st.divider()

# Display chat history
for message in st.session_state.chat_history:
    role = "user" if message.__class__.__name__ == "HumanMessage" else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("Ask something..."):
    if not gemini_api_key:
        st.error("Missing API Key.")
    elif not st.session_state.vectorstore:
        st.error("Upload documents first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)

        # Setup Modern RAG Chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        retriever = st.session_state.vectorstore.as_retriever()

        # 1. Contextualize question (Chat History Aware)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # 2. Answer question
        system_prompt = (
            "You are an expert study assistant. Use the following retrieved context "
            "to answer the question. If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
                st.markdown(response["answer"])
                
                # Update history
                from langchain_core.messages import HumanMessage, AIMessage
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=response["answer"]),
                ])

st.sidebar.caption("v2.0 (Modern LCEL) | Built with Streamlit & LangChain")
