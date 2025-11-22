"""
Text chunking module for semantic text splitting.
Provides intelligent chunking for better retrieval performance.
"""
import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextChunker:
    """Chunks text into smaller, semantically meaningful pieces."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        
        # Initialize RecursiveCharacterTextSplitter
        # This splitter tries to split on paragraphs, then sentences, then words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                "! ",
                "? ",
                ", ",    # Clauses
                " ",     # Words
                ""       # Characters
            ],
            is_separator_regex=False
        )
        
        self.logger.info(
            f"TextChunker initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def chunk_text(self, text: str, source: str = "unknown") -> List[dict]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to chunk
            source: Source identifier (e.g., filename)
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text or not text.strip():
            self.logger.warning(f"Empty text provided for chunking from source: {source}")
            return []
        
        self.logger.info(f"Chunking text from {source} ({len(text)} characters)")
        
        # Split text into chunks
        text_chunks = self.splitter.split_text(text)
        
        # Create chunks with metadata
        chunks_with_metadata = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "source": source,
                    "chunk_id": i,
                    "total_chunks": len(text_chunks),
                    "char_count": len(chunk_text)
                }
            }
            chunks_with_metadata.append(chunk)
        
        self.logger.info(
            f"Created {len(chunks_with_metadata)} chunks from {source}"
        )
        
        return chunks_with_metadata
    
    def chunk_multiple_sources(self, sources: List[dict]) -> List[dict]:
        """
        Chunk text from multiple sources.
        
        Args:
            sources: List of dicts with 'text' and 'source' keys
                    Example: [{"text": "...", "source": "document.pdf"}, ...]
            
        Returns:
            Combined list of chunks from all sources
        """
        all_chunks = []
        
        for source_data in sources:
            text = source_data.get("text", "")
            source = source_data.get("source", "unknown")
            
            chunks = self.chunk_text(text, source)
            all_chunks.extend(chunks)
        
        self.logger.info(
            f"Total chunks created from {len(sources)} sources: {len(all_chunks)}"
        )
        
        return all_chunks


if __name__ == "__main__":
    # Test the chunker
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    sample_text = """
    This is a sample text for testing the text chunker.
    
    It has multiple paragraphs to demonstrate how the chunker works.
    The chunker should split this text into meaningful chunks.
    
    Each chunk should have some overlap with the previous chunk.
    This helps maintain context when retrieving relevant information.
    
    The chunker uses recursive splitting, trying to split on:
    - Paragraphs first
    - Then sentences
    - Then words
    - And finally characters if needed
    
    This ensures that the chunks are as semantically meaningful as possible.
    """ * 5  # Repeat to create longer text
    
    chunks = chunker.chunk_text(sample_text, source="test.txt")
    
    print(f"\nCreated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"Characters: {chunk['metadata']['char_count']}")
        print(f"Text preview: {chunk['text'][:100]}...")
