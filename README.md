# 📝 AI RAG Navigator: Intelligent Study Companion

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-orange.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Live-App-FF4B4B.svg)](https://akkikrsingh2005-ai-rag-navigator-app-5gfm3y.streamlit.app/)

An enterprise-grade **Retrieval-Augmented Generation (RAG)** application designed to transform static study materials into interactive AI-driven conversations. Built for the modern LLM ecosystem in 2026.

## 🚀 Key Features
- **Context-Aware Retrieval**: Uses **Modern LCEL (LangChain Expression Language)** for stable and high-performance document interrogation.
- **Multimodal Document Processing**: High-accuracy PDF parsing and semantic chunking.
- **Vector Intelligence**: Seamlessly integrates with **ChromaDB** for local, privacy-centric vector storage.
- **Dual-Model Inference**: Leverages **Google Gemini 1.5 Flash** for high-speed, cost-effective reasoning.

## 🛠️ Technical Implementation
- **Framework**: LangChain v0.3 Core (History-aware retrievers).
- **Embeddings**: Google GoogleGenerativeAIEmbeddings (`embedding-001`).
- **Database**: SQLITE-optimized ChromaDB (pysqlite3 monkeypatched for cloud).
- **Interface**: Real-time Streamlit dashboard with conversational history.

## 🏃 Quick Start
1. **Explore the Live App**: [AI RAG Navigator Demo](https://akkikrsingh2005-ai-rag-navigator-app-5gfm3y.streamlit.app/)
2. **Local Setup**:
   ```bash
   git clone https://github.com/AkkiKrsingh2005/ai-rag-navigator.git
   cd ai-rag-navigator
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. **Environment**: Add your `GOOGLE_API_KEY` to the application sidebar or a `.env` file.

---
#### Developed as part of an AI/ML Internship Portfolio 🧠
Developed by **Ankit Kumar** | [Portfolio](https://rajesh-portfolio-two.vercel.app/)
