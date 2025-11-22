"""
Vector store module using ChromaDB and sentence-transformers.
Handles embedding generation and vector storage/retrieval.
"""
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector embeddings and ChromaDB storage."""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
            embedding_model_name: Name of the sentence-transformers model
        """
        self.logger = logger
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_model_name = embedding_model_name
        
        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
        # Initialize ChromaDB client
        self.logger.info(f"Initializing ChromaDB at: {persist_directory}")
        self.client = chromadb.HttpClient(
            host="localhost",
            port=8000
        )

        # self.client = chromadb.PersistentClient(
        #     path=str(self.persist_directory),
        #     settings=Settings(
        #         anonymized_telemetry=False,
        #         allow_reset=True
        #     )
        # )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self.logger.info(
            f"Vector store initialized with collection: {collection_name}"
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
        """
        if not chunks:
            self.logger.warning("No chunks provided to add to vector store")
            return
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Generate IDs for chunks
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(chunks))]
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        self.logger.info(
            f"Successfully added {len(chunks)} chunks. "
            f"Total documents in collection: {self.collection.count()}"
        )
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text
            top_k: Number of top results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Dictionary with query results including documents, metadatas, and distances
        """
        self.logger.info(f"Querying vector store: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embed_text(query_text)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = {
            "query": query_text,
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "count": len(results["documents"][0]) if results["documents"] else 0
        }
        
        self.logger.info(f"Found {formatted_results['count']} relevant documents")
        
        return formatted_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_dimension,
            "persist_directory": str(self.persist_directory)
        }
        
        return stats
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.logger.warning("Clearing all documents from collection")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.logger.info("Collection cleared")
    
    def reset(self) -> None:
        """Reset the entire ChromaDB database."""
        self.logger.warning("Resetting ChromaDB database")
        self.client.reset()
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.logger.info("Database reset complete")


if __name__ == "__main__":
    # Test the vector store
    print("\n=== Testing Vector Store ===")
    
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_directory="./test_chroma_db"
    )
    
    # Sample chunks
    sample_chunks = [
        {
            "text": "Vector databases are optimized for similarity search.",
            "metadata": {"source": "test.pdf", "chunk_id": 0}
        },
        {
            "text": "ChromaDB is an open-source embedding database.",
            "metadata": {"source": "test.pdf", "chunk_id": 1}
        },
        {
            "text": "RAG combines retrieval with language model generation.",
            "metadata": {"source": "test.pdf", "chunk_id": 2}
        }
    ]
    
    # Add documents
    vector_store.add_documents(sample_chunks)
    
    # Query
    results = vector_store.query("What is a vector database?", top_k=2)
    
    print(f"\nQuery: {results['query']}")
    print(f"Found {results['count']} results:\n")
    
    for i, (doc, metadata, distance) in enumerate(
        zip(results['documents'], results['metadatas'], results['distances'])
    ):
        print(f"Result {i+1}:")
        print(f"  Text: {doc}")
        print(f"  Source: {metadata['source']}")
        print(f"  Distance: {distance:.4f}\n")
    
    # Stats
    stats = vector_store.get_stats()
    print("\nVector Store Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
