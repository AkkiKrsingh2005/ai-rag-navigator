import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
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
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key updated!")
    else:
        st.warning("Please enter your API Key to continue.")
    
    st.divider()
    st.markdown("""
        ### How it works:
        1. **Upload** your PDF documents.
        2. **Process** the documents into small chunks.
        3. **Embed** them in a local vector database.
        4. **Chat** with your notes using AI!
    """)

# Initialize session state for chat and vectorstore
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# File uploader
uploaded_files = st.file_uploader("Choose your study materials (PDFs)", type="pdf", accept_multiple_files=True)

if uploaded_files and api_key:
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

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(all_pages)

            # Create embeddings and vectorstore
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            st.session_state.vectorstore = vectorstore
            st.success(f"Successfully processed {len(uploaded_files)} documents into {len(splits)} segments!")

# Chat Interface
st.divider()
st.subheader("💬 Chat")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask something about your notes..."):
    if not api_key:
        st.error("Please enter your API key in the sidebar.")
    elif not st.session_state.vectorstore:
        st.error("Please upload and process some documents first.")
    else:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            
            # Setup Retrieval Chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=st.session_state.vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            # Prepare chat history for LangChain
            chat_history = [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"]
            
            with st.spinner("Thinking..."):
                result = qa_chain({"question": prompt, "chat_history": []}) 
                response = result["answer"]
                st.markdown(response)
                
                # Show sources (optional)
                with st.expander("Show Sources"):
                    for doc in result["source_documents"][:3]:
                        st.write(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})")
                        st.caption(doc.page_content[:200] + "...")

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})

# Visual polish
st.sidebar.caption("v1.0 | Built with Streamlit & LangChain")
