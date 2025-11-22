"""
Test script for RAG Chatbot.
Tests the complete pipeline with sample questions.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from main import RAGChatbot


def test_questions():
    """Test the chatbot with predefined questions."""
    
    # Sample questions from the requirements
    questions = [
        "What are the production 'Do's' for RAG?",
        "What is the difference between standard retrieval and the ColPali approach?",
        "Why is hybrid search better than vector-only search?"
    ]
    
    print("\n" + "=" * 80)
    print("TESTING RAG CHATBOT")
    print("=" * 80)
    
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize chatbot
        print("\nInitializing chatbot...")
        chatbot = RAGChatbot(Config)
        
        # Check if knowledge base has data
        stats = chatbot.vector_store.get_stats()
        if stats["total_documents"] == 0:
            print("\n⚠️  WARNING: Knowledge base is empty!")
            print("Please ingest documents first using:")
            print("  python main.py --ingest-pdf data/your_presentation.pdf")
            print("  python main.py --ingest-audio data/your_lecture.mp3")
            return
        
        print(f"\n✓ Knowledge base loaded: {stats['total_documents']} documents")
        
        # Test each question
        for i, question in enumerate(questions, 1):
            print(f"\n{'=' * 80}")
            print(f"TEST QUESTION {i}/{len(questions)}")
            print(f"{'=' * 80}")
            
            chatbot.ask_question(question, show_context=False)
            
            # Add separator
            if i < len(questions):
                print("\n" + "~" * 80 + "\n")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETED")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        raise


if __name__ == "__main__":
    test_questions()
