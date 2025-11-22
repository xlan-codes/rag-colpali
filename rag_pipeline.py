"""
RAG Pipeline for question answering.
Combines retrieval from vector store with LLM generation.
"""
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: VectorStore instance for retrieval
            api_key: OpenAI API key
            model: OpenAI model name
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens in response
        """
        self.logger = logger
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        
        self.logger.info(
            f"RAG Pipeline initialized with model: {model}"
        )
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from vector store.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filter
            
        Returns:
            List of retrieved documents with metadata
        """
        self.logger.info(f"Retrieving context for query: '{query[:50]}...'")
        
        # Query vector store
        results = self.vector_store.query(
            query_text=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Format retrieved documents
        retrieved_docs = []
        for doc, metadata, distance in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            retrieved_docs.append({
                "text": doc,
                "metadata": metadata,
                "distance": distance,
                "relevance_score": 1 - distance  # Convert distance to similarity
            })
        
        self.logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc["metadata"].get("source", "unknown")
            text = doc["text"]
            relevance = doc["relevance_score"]
            
            context_parts.append(
                f"[Document {i} - Source: {source} - Relevance: {relevance:.2f}]\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = """You are a helpful AI assistant answering questions about databases for GenAI and RAG systems.
Use the provided context to answer questions accurately and comprehensively.
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite the relevant parts of the context in your answer.
Be concise but thorough."""
        
        # Construct messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Context information is below:
---
{context}
---

Based on the context above, please answer the following question:
{query}

Provide a clear, well-structured answer that references the context where appropriate."""
            }
        ]
        
        self.logger.info(f"Generating answer with {self.model}")
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            self.logger.info("Answer generated successfully")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            raise
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_context: bool = False
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filter
            return_context: Whether to return retrieved context in response
            
        Returns:
            Dictionary with answer and optional context
        """
        self.logger.info(f"Processing question: '{question[:100]}...'")
        
        # Step 1: Retrieve relevant context
        retrieved_docs = self.retrieve_context(
            query=question,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Step 2: Format context
        context = self.format_context(retrieved_docs)
        
        # Step 3: Generate answer
        answer = self.generate_answer(
            query=question,
            context=context
        )
        
        # Prepare response
        response = {
            "question": question,
            "answer": answer,
            "num_sources": len(retrieved_docs)
        }
        
        if return_context:
            response["context"] = retrieved_docs
        
        self.logger.info("RAG query completed successfully")
        
        return response
    
    def batch_query(
        self,
        questions: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            top_k: Number of documents to retrieve per question
            
        Returns:
            List of responses
        """
        self.logger.info(f"Processing {len(questions)} questions in batch")
        
        responses = []
        for i, question in enumerate(questions, 1):
            self.logger.info(f"Processing question {i}/{len(questions)}")
            response = self.query(question, top_k=top_k)
            responses.append(response)
        
        return responses


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test RAG pipeline
    print("\n=== Testing RAG Pipeline ===")
    
    # Initialize vector store (assuming it has data)
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_directory="./test_chroma_db"
    )
    
    # Initialize RAG pipeline
    rag = RAGPipeline(
        vector_store=vector_store,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    # Test query
    test_question = "What is a vector database?"
    response = rag.query(test_question, return_context=True)
    
    print(f"\nQuestion: {response['question']}")
    print(f"\nAnswer: {response['answer']}")
    print(f"\nUsed {response['num_sources']} sources")
