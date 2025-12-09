# RAG Chatbot for GenAI Databases Lecture

A production-ready RAG (Retrieval-Augmented Generation) chatbot that ingests PDF documents and audio recordings to answer questions about the "Databases for GenAI" lecture.

## 🎯 Project Goal

Build a functional RAG chatbot from scratch that can:
- Ingest and process knowledge from PDFs and audio files
- Extract text from the "Databases for GenAI" presentation
- Transcribe video recordings of the lecture
- Answer questions accurately using retrieval-augmented generation

## ✨ Features

- ✅ **Multi-format ingestion**: PDFs and audio (MP3, WAV, M4A, MP4)
- ✅ **Robust text extraction**: pdfplumber for reliable PDF parsing
- ✅ **Audio transcription**: OpenAI Whisper for speech-to-text
- ✅ **Semantic chunking**: Intelligent text splitting
- ✅ **Vector database**: ChromaDB for efficient similarity search
- ✅ **Production LLM**: OpenAI GPT-5.1
- ✅ **Production-ready**: Error handling, logging, configuration
- ✅ **Interactive CLI**: Command-line and chat interfaces

## 🚀 Quick Start

### 1. Setup

```bash
# Run automated setup
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Add your OpenAI API key to .env
nano .env  # OPENAI_API_KEY=sk-...
```

### 2. Add Data

```bash
# Place files in data/ directory
mkdir -p data
# Copy your PDF and audio files to data/
```

### 3. Ingest Documents

```bash
# Batch ingest (recommended)
python batch_ingest.py
```

### 4. Ask Questions

```bash
# Interactive mode
python main.py --chat

# Test script
python test_chatbot.py
```

## 🏗️ Architecture

```
INGESTION: PDF/Audio → Extract/Transcribe → Chunk → Embed → ChromaDB
QUERY: Question → Embed → Search → Retrieve → LLM → Answer
```

## 💻 Usage

### Ingest Documents

```bash
# Batch ingestion
python batch_ingest.py

# Individual files
python main.py --ingest-pdf data/presentation.pdf
python main.py --ingest-audio data/lecture.mp3
```

### Ask Questions

```bash
# Interactive chat
python main.py --chat

# Single question
python main.py --question "What are the production 'Do's' for RAG?"

# With context
python main.py -q "Your question" --show-context

# Test script
python test_chatbot.py

# Example usage
python example_usage.py
```

### Monitoring

```bash
# Statistics
python main.py --stats

# Configuration
python main.py --config
```

## 🎯 Sample Questions

1. "What are the production 'Do's' for RAG?"
2. "What is the difference between standard retrieval and the ColPali approach?"
3. "Why is hybrid search better than vector-only search?"

## 🛠️ Technology Stack

| Component | Technology            |
|-----------|-----------------------|
| PDF Processing | pdfplumber     |
| Audio Transcription | OpenAI Whisper        |
| Text Chunking | LangChain             |
| Embeddings | Sentence-Transformers |
| Vector DB | ChromaDB      |
| LLM | OpenAI GPT-5.1        |

## ⚙️ Configuration

Edit `.env` file:

```bash
OPENAI_API_KEY=sk-...        # Required
CHUNK_SIZE=1000              # Text chunk size
CHUNK_OVERLAP=200            # Chunk overlap
TOP_K_RESULTS=5              # Retrieved docs
LLM_MODEL=gpt-5.1            # OpenAI model
LLM_TEMPERATURE=0.7          # Creativity
MAX_TOKENS=500               # Response length
```

## 🚀 Performance

- **PDF**: 1-2 seconds per file
- **Audio**: 1-5 minutes per file (Whisper base, CPU)
- **Query**: 2-4 seconds per question

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- 8GB RAM (16GB recommended)
- 5GB disk space

## 🔧 Troubleshooting

**Import errors**: Activate virtual environment
```bash
source venv/bin/activate
```

**API key error**: Check `.env` file has `OPENAI_API_KEY`

**No results**: Ingest documents first with `python batch_ingest.py`

**Memory issues**: Use smaller Whisper model (`tiny` or `base`)

## 📚 Commands Reference

```bash
# Setup
./setup.sh

# Ingest
python batch_ingest.py
python main.py --ingest-pdf FILE
python main.py --ingest-audio FILE

# Query
python main.py --chat
python main.py --question "Question"
python test_chatbot.py

# Monitor
python main.py --stats
python main.py --config
```
