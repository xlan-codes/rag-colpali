"""
Data loading module for PDF and audio processing.
Handles text extraction from PDFs and audio transcription.
"""
import logging
from pathlib import Path
from typing import Optional
import pdfplumber
import whisper
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads and extracts text from PDF files using pdfplumber."""
    
    def __init__(self):
        """Initialize PDF loader."""
        self.logger = logger
    
    def load(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If text extraction fails
        """
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Loading PDF: {pdf_path}")
        
        try:
            text_content = []
            
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                self.logger.info(f"Processing {total_pages} pages...")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text from page
                    text = page.extract_text()
                    
                    if text:
                        text_content.append(text)
                        self.logger.debug(f"Extracted text from page {page_num}/{total_pages}")
                    else:
                        self.logger.warning(f"No text found on page {page_num}")
            
            full_text = "\n\n".join(text_content)
            
            if not full_text.strip():
                raise ValueError("No text content extracted from PDF")
            
            self.logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            raise


class AudioTranscriber:
    """Transcribes audio files using OpenAI Whisper."""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize audio transcriber.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
                       'base' is a good balance of speed and accuracy
        """
        self.logger = logger
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.logger.info("Whisper model loaded successfully")
    
    def transcribe(self, audio_path: str, save_transcript: bool = True, 
                   output_dir: Optional[str] = None) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            save_transcript: Whether to save transcript to file
            output_dir: Directory to save transcript (default: transcriptions/)
            
        Returns:
            Transcribed text
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If transcription fails
        """
        audio_file = Path(audio_path)
        
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        self.logger.info(f"Transcribing audio: {audio_path}")
        self._load_model()
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(audio_file),
                language="en",  # Change if needed
                task="transcribe",
                verbose=False
            )
            
            transcript = result["text"]
            
            if not transcript.strip():
                raise ValueError("No text transcribed from audio")
            
            self.logger.info(f"Successfully transcribed {len(transcript)} characters")
            
            # Save transcript to file
            if save_transcript:
                if output_dir is None:
                    output_dir = Path("transcriptions")
                else:
                    output_dir = Path(output_dir)
                
                output_dir.mkdir(exist_ok=True)
                
                transcript_file = output_dir / f"{audio_file.stem}_transcript.txt"
                transcript_file.write_text(transcript, encoding="utf-8")
                self.logger.info(f"Transcript saved to: {transcript_file}")
            
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            raise


if __name__ == "__main__":
    # Test PDF loader
    print("\n=== Testing PDF Loader ===")
    pdf_loader = PDFLoader()
    # pdf_text = pdf_loader.load("data/sample.pdf")
    # print(f"Extracted {len(pdf_text)} characters")
    
    # Test Audio transcriber
    print("\n=== Testing Audio Transcriber ===")
    transcriber = AudioTranscriber(model_name="base")
    # audio_text = transcriber.transcribe("data/sample.mp3")
    # print(f"Transcribed {len(audio_text)} characters")
