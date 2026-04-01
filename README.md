# 📝 Talk to Your Notes (RAG System)

An AI-powered document interrogation system that uses **Retrieval-Augmented Generation (RAG)** to let you chat with your PDFs. 

## 🚀 Features
- **PDF Ingestion**: Upload multiple study materials at once.
- **Semantic Search**: Uses Google Gemini Embeddings to find relevant text chunks.
- **Conversational AI**: Powered by Google Gemini 1.5 Flash.
- **Source Attribution**: See exactly where the AI found its information.

## 🛠️ How to Run
1. **Clone the repository**:
   ```bash
   git clone <your-repo-link>
   cd rag-notes
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   streamlit run app.py
   ```
4. **Enter your API Key**:
   - Obtain a free API key from [Google AI Studio](https://aistudio.google.com/).
   - Paste it into the sidebar of the application.

## 🧠 Technologies Used
- **Streamlit**: For the interactive web interface.
- **LangChain**: For the RAG orchestration and memory.
- **Google Gemini**: Large Language Model and Embeddings.
- **ChromaDB**: High-performance vector database for local storage.

---
Built by [Ankit Kumar](https://github.com/raxx21)
