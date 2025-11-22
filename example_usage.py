"""
Example usage script demonstrating all features.
Run this after setting up your environment and ingesting data.
"""
import os
from pathlib import Path

# Ensure we're in the right directory
os.chdir(Path(__file__).parent)

from config import Config
from main import RAGChatbot


def main():
    """Demonstrate all RAG chatbot features."""
    
    print("\n" + "=" * 80)
    print("RAG CHATBOT - COMPLETE EXAMPLE")
    print("=" * 80)
    
    # 1. Display configuration
    print("\n1. CONFIGURATION")
    print("-" * 80)
    Config.validate()
    Config.display()
    
    # 2. Initialize chatbot
    print("\n2. INITIALIZING CHATBOT")
    print("-" * 80)
    chatbot = RAGChatbot(Config)
    print("✓ Chatbot initialized successfully")
    
    # 3. Check knowledge base
    print("\n3. KNOWLEDGE BASE STATUS")
    print("-" * 80)
    stats = chatbot.vector_store.get_stats()
    
    if stats["total_documents"] == 0:
        print("\n⚠️  Knowledge base is empty!")
        print("\nTo use this example, first ingest some documents:")
        print("\n  # Option 1: Batch ingest all files in data/")
        print("  python batch_ingest.py")
        print("\n  # Option 2: Ingest individual files")
        print("  python main.py --ingest-pdf data/your_file.pdf")
        print("  python main.py --ingest-audio data/your_audio.mp3")
        return
    
    print(f"✓ Knowledge base loaded")
    print(f"  - Total documents: {stats['total_documents']}")
    print(f"  - Embedding model: {stats['embedding_model']}")
    print(f"  - Vector dimension: {stats['embedding_dimension']}")
    
    # 4. Sample questions
    print("\n4. SAMPLE QUERIES")
    print("-" * 80)
    
    sample_questions = [
        "What are the production 'Do's' for RAG?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "Why is hybrid search better than vector-only search?",
        "What are the key features of vector databases?",
        "How does ChromaDB work?"
    ]
    
    print(f"\nAsking {len(sample_questions)} questions...\n")
    
    for i, question in enumerate(sample_questions[:3], 1):  # Ask first 3
        print(f"\n{'=' * 80}")
        print(f"QUESTION {i}: {question}")
        print('=' * 80)
        
        # Get answer
        response = chatbot.rag_pipeline.query(question, top_k=5, return_context=True)
        
        print(f"\nANSWER:\n{response['answer']}")
        print(f"\nSources used: {response['num_sources']}")
        
        # Show top source
        if response.get('context'):
            top_source = response['context'][0]
            print(f"\nTop source: {top_source['metadata']['source']}")
            print(f"Relevance score: {top_source['relevance_score']:.3f}")
        
        print("\n" + "-" * 80)
    
    # 5. Interactive mode hint
    print("\n5. NEXT STEPS")
    print("-" * 80)
    print("\nFor more questions, use interactive mode:")
    print("  python main.py --chat")
    print("\nOr ask individual questions:")
    print("  python main.py --question 'Your question here'")
    print("\nTo see retrieved context:")
    print("  python main.py --question 'Your question' --show-context")
    
    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
