#!/bin/bash

# Setup script for RAG Chatbot
# This script sets up the complete environment

set -e  # Exit on error

echo "=================================="
echo "RAG Chatbot Setup"
echo "=================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python3 --version || {
    echo "Error: Python 3 is not installed"
    exit 1
}

# Create virtual environment
echo -e "\n2. Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists"
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n3. Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo -e "\n4. Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo -e "\n5. Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

# Create necessary directories
echo -e "\n6. Creating directories..."
mkdir -p data
mkdir -p chroma_db
mkdir -p transcriptions
echo "✓ Directories created"

# Setup environment file
echo -e "\n7. Setting up environment configuration..."
if [ -f ".env" ]; then
    echo ".env file already exists"
else
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your OPENAI_API_KEY"
    echo "   You can get an API key from: https://platform.openai.com/api-keys"
fi

echo -e "\n=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OPENAI_API_KEY"
echo "2. Place your PDF and audio files in the data/ directory"
echo "3. Activate the virtual environment: source venv/bin/activate"
echo "4. Run batch ingestion: python batch_ingest.py"
echo "5. Test the chatbot: python test_chatbot.py"
echo "6. Start interactive mode: python main.py --chat"
echo ""
echo "For help: python main.py --help"
echo "=================================="
