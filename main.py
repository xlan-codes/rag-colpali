"""
Main CLI interface for RAG Chatbot.
Handles document ingestion and interactive Q&A.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from config import Config
from data_loader import PDFLoader, AudioTranscriber
from text_chunker import TextChunker
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGChatbot:
    """Main RAG Chatbot orchestrator."""
    
    def __init__(self, config: Config):
        """
        Initialize RAG Chatbot.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logger
        
        # Initialize components
        self.pdf_loader = PDFLoader()
        self.audio_transcriber = AudioTranscriber(model_name="base")
        self.text_chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(
            collection_name="genai_databases_lecture",
            persist_directory=config.CHROMA_PERSIST_DIR,
            embedding_model_name=config.EMBEDDING_MODEL
        )
        self.rag_pipeline = RAGPipeline(
            vector_store=self.vector_store,
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
        
        self.logger.info("RAG Chatbot initialized successfully")
    
    def ingest_pdf(self, pdf_path: str) -> None:
        """
        Ingest a PDF document.
        
        Args:
            pdf_path: Path to PDF file
        """
        try:
            self.logger.info(f"Starting PDF ingestion: {pdf_path}")
            
            # Load and extract text
            text = self.pdf_loader.load(pdf_path)
            
            # Chunk text
            source_name = Path(pdf_path).name
            chunks = self.text_chunker.chunk_text(text, source=source_name)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            self.logger.info(f"PDF ingestion completed: {len(chunks)} chunks added")
            print(f"\n✓ Successfully ingested PDF: {source_name}")
            print(f"  - Added {len(chunks)} chunks to knowledge base")
            
        except Exception as e:
            self.logger.error(f"Error ingesting PDF: {e}")
            print(f"\n✗ Error ingesting PDF: {e}")
            raise
    
    def ingest_audio(self, audio_path: str) -> None:
        """
        Ingest an audio file.
        
        Args:
            audio_path: Path to audio file
        """
        try:
            self.logger.info(f"Starting audio ingestion: {audio_path}")
            
            # Transcribe audio
            transcript = self.audio_transcriber.transcribe(
                audio_path,
                save_transcript=True,
                output_dir=str(self.config.TRANSCRIPTIONS_DIR)
            )
            
            # Chunk text
            source_name = Path(audio_path).name
            chunks = self.text_chunker.chunk_text(transcript, source=source_name)
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            self.logger.info(f"Audio ingestion completed: {len(chunks)} chunks added")
            print(f"\n✓ Successfully ingested audio: {source_name}")
            print(f"  - Transcribed {len(transcript)} characters")
            print(f"  - Added {len(chunks)} chunks to knowledge base")
            
        except Exception as e:
            self.logger.error(f"Error ingesting audio: {e}")
            print(f"\n✗ Error ingesting audio: {e}")
            raise
    
    def ask_question(self, question: str, show_context: bool = False) -> None:
        """
        Ask a question and get an answer.
        
        Args:
            question: Question to ask
            show_context: Whether to show retrieved context
        """
        try:
            # Query RAG pipeline
            response = self.rag_pipeline.query(
                question,
                top_k=self.config.TOP_K_RESULTS,
                return_context=show_context
            )
            
            # Display results
            print("\n" + "=" * 80)
            print(f"QUESTION: {response['question']}")
            print("=" * 80)
            print(f"\nANSWER:\n{response['answer']}")
            print(f"\n(Based on {response['num_sources']} relevant sources)")
            
            # Show context if requested
            if show_context and "context" in response:
                print("\n" + "-" * 80)
                print("RETRIEVED CONTEXT:")
                print("-" * 80)
                for i, doc in enumerate(response["context"], 1):
                    print(f"\n[Source {i}: {doc['metadata']['source']} - "
                          f"Relevance: {doc['relevance_score']:.2f}]")
                    print(doc["text"][:300] + "...")
            
            print("\n" + "=" * 80)
            
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            print(f"\n✗ Error: {e}")
            raise
    
    def interactive_mode(self) -> None:
        """Start interactive Q&A session."""
        print("\n" + "=" * 80)
        print("RAG CHATBOT - INTERACTIVE MODE")
        print("=" * 80)
        print("\nAsk questions about the GenAI Databases lecture.")
        print("Commands:")
        print("  - Type 'quit' or 'exit' to exit")
        print("  - Type 'stats' to see knowledge base statistics")
        print("  - Type 'context' to toggle context display")
        print("=" * 80 + "\n")
        
        show_context = False
        
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if question.lower() == 'stats':
                    stats = self.vector_store.get_stats()
                    print("\n📊 Knowledge Base Statistics:")
                    for key, value in stats.items():
                        print(f"  - {key}: {value}")
                    continue
                
                if question.lower() == 'context':
                    show_context = not show_context
                    print(f"\n💡 Context display: {'ON' if show_context else 'OFF'}")
                    continue
                
                # Ask question
                self.ask_question(question, show_context=show_context)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}")
                print(f"\n✗ Error: {e}")
                continue
    
    def display_stats(self) -> None:
        """Display knowledge base statistics."""
        stats = self.vector_store.get_stats()
        
        print("\n" + "=" * 60)
        print("KNOWLEDGE BASE STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Chatbot for GenAI Databases Lecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest PDF
  python main.py --ingest-pdf data/presentation.pdf
  
  # Ingest audio
  python main.py --ingest-audio data/lecture.mp3
  
  # Ask a question
  python main.py --question "What are the production Do's for RAG?"
  
  # Interactive mode
  python main.py --chat
  
  # Show statistics
  python main.py --stats
        """
    )
    
    parser.add_argument(
        '--ingest-pdf',
        type=str,
        help='Path to PDF file to ingest'
    )
    
    parser.add_argument(
        '--ingest-audio',
        type=str,
        help='Path to audio file to ingest'
    )
    
    parser.add_argument(
        '--question', '-q',
        type=str,
        help='Question to ask'
    )
    
    parser.add_argument(
        '--chat', '-c',
        action='store_true',
        help='Start interactive chat mode'
    )
    
    parser.add_argument(
        '--show-context',
        action='store_true',
        help='Show retrieved context with answers'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show knowledge base statistics'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show current configuration'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    try:
        # Validate and load configuration
        Config.validate()
        
        if args.config:
            Config.display()
            return
        
        # Initialize chatbot
        chatbot = RAGChatbot(Config)
        
        # Handle ingestion
        if args.ingest_pdf:
            chatbot.ingest_pdf(args.ingest_pdf)
        
        if args.ingest_audio:
            chatbot.ingest_audio(args.ingest_audio)
        
        # Handle query
        if args.question:
            chatbot.ask_question(args.question, show_context=args.show_context)
        
        # Handle stats
        if args.stats:
            chatbot.display_stats()
        
        # Handle interactive mode
        if args.chat:
            chatbot.interactive_mode()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
