"""
Batch ingestion script for processing multiple files.
"""
import argparse
import sys
from pathlib import Path
from typing import List

from config import Config
from main import RAGChatbot


def find_files(directory: str, extensions: List[str]) -> List[Path]:
    """
    Find all files with given extensions in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.pdf', '.mp3'])
        
    Returns:
        List of file paths
    """
    dir_path = Path(directory)
    files = []
    
    for ext in extensions:
        files.extend(dir_path.glob(f"*{ext}"))
    
    return sorted(files)


def batch_ingest(data_dir: str = "data"):
    """
    Batch ingest all PDF and audio files from a directory.
    
    Args:
        data_dir: Directory containing files to ingest
    """
    print("\n" + "=" * 80)
    print("BATCH INGESTION")
    print("=" * 80)
    
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize chatbot
        print("\nInitializing chatbot...")
        chatbot = RAGChatbot(Config)
        
        # Find files
        print(f"\nSearching for files in: {data_dir}")
        pdf_files = find_files(data_dir, ['.pdf'])
        audio_files = find_files(data_dir, ['.mp3', '.wav', '.m4a', '.mp4'])
        
        print(f"\nFound:")
        print(f"  - {len(pdf_files)} PDF file(s)")
        print(f"  - {len(audio_files)} audio file(s)")
        
        if not pdf_files and not audio_files:
            print("\n⚠️  No files found to ingest!")
            print(f"Please add PDF or audio files to the '{data_dir}' directory.")
            return
        
        # Ingest PDFs
        if pdf_files:
            print("\n" + "-" * 80)
            print("INGESTING PDF FILES")
            print("-" * 80)
            
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
                try:
                    chatbot.ingest_pdf(str(pdf_file))
                except Exception as e:
                    print(f"✗ Error processing {pdf_file.name}: {e}")
                    continue
        
        # Ingest audio
        if audio_files:
            print("\n" + "-" * 80)
            print("INGESTING AUDIO FILES")
            print("-" * 80)
            print("\n⚠️  Note: Audio transcription may take several minutes per file.")
            
            for i, audio_file in enumerate(audio_files, 1):
                print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
                try:
                    chatbot.ingest_audio(str(audio_file))
                except Exception as e:
                    print(f"✗ Error processing {audio_file.name}: {e}")
                    continue
        
        # Show final statistics
        print("\n" + "=" * 80)
        print("BATCH INGESTION COMPLETED")
        print("=" * 80)
        chatbot.display_stats()
        
    except Exception as e:
        print(f"\n✗ Error during batch ingestion: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch ingest PDF and audio files into RAG knowledge base"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing files to ingest (default: data)'
    )
    
    args = parser.parse_args()
    
    batch_ingest(args.data_dir)


if __name__ == "__main__":
    main()
