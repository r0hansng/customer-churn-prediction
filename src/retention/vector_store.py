import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Resolve project root so paths work regardless of the launch directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DEFAULT_KB_PATH = os.path.join(
    _PROJECT_ROOT, "src", "data", "knowledge_base", "telecom_retention_policies.md"
)

# Prefer Streamlit secrets (cloud) → fall back to env var / .env (local)
def _get_api_key() -> str:
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

class VectorStoreManager:
    def __init__(self, data_path: str = _DEFAULT_KB_PATH):
        self.data_path = data_path
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=_get_api_key()
        )
        self.vector_store = None

    def initialize_store(self):
        """Loads default policies and builds FAISS index."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Knowledge base file not found at {self.data_path}")
            
        loader = TextLoader(self.data_path)
        documents = loader.load()
        
        # Split documents into chunks for retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)
        
        # Initialize FAISS with OpenAI embeddings
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        return self.vector_store

    def get_retriever(self, k=2):
        """Returns a retriever interface for LangGraph."""
        if self.vector_store is None:
            self.initialize_store()
        return self.vector_store.as_retriever(search_kwargs={"k": k})

# Global instance for easier import
store_manager = VectorStoreManager()
