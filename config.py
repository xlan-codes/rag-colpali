"""
Configuration management for RAG Chatbot.
Loads environment variables and provides configuration settings.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for RAG Chatbot."""
    
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", str(BASE_DIR / "chroma_db"))
    TRANSCRIPTIONS_DIR = BASE_DIR / "transcriptions"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Chunking Parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval Parameters
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    
    # LLM Settings
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5.1")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please copy .env.example to .env and add your API key."
            )
        
        # Create necessary directories
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)
        Path(cls.CHROMA_PERSIST_DIR).mkdir(exist_ok=True)
    
    @classmethod
    def display(cls):
        """Display current configuration (masking sensitive data)."""
        print("=" * 60)
        print("RAG Chatbot Configuration")
        print("=" * 60)
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"ChromaDB Directory: {cls.CHROMA_PERSIST_DIR}")
        print(f"Transcriptions Directory: {cls.TRANSCRIPTIONS_DIR}")
        print(f"\nEmbedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Top-K Results: {cls.TOP_K_RESULTS}")
        print(f"\nLLM Model: {cls.LLM_MODEL}")
        print(f"LLM Temperature: {cls.LLM_TEMPERATURE}")
        print(f"Max Tokens: {cls.MAX_TOKENS}")
        print(f"\nOpenAI API Key: {'✓ Set' if cls.OPENAI_API_KEY else '✗ Not Set'}")
        print("=" * 60)


if __name__ == "__main__":
    Config.validate()
    Config.display()
