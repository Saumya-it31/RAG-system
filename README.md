# RAG System - Retrieval-Augmented Generation

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)

## 🌟 Overview

This is a complete, production-ready RAG (Retrieval-Augmented Generation) system that can ingest PDF documents, answer questions with citations, and highlight exact evidence from source pages. The system uses a local, open-source stack for complete privacy and control.

**Perfect for academic assignments and enterprise applications!**

## ✨ Features

### 🔍 **Advanced PDF Processing**

- **Native Text Extraction**: Fast extraction from text-based PDFs using PyMuPDF
- **OCR Support**: Automatic fallback to OCR for scanned documents using Tesseract
- **Mixed Document Support**: Handles both native and scanned PDFs seamlessly
- **Deduplication**: Prevents duplicate document ingestion using file hashing

### 📚 **Intelligent Chunking**

- **Sentence-Aware Chunking**: Preserves semantic boundaries
- **Configurable Overlap**: Maintains context between chunks
- **Metadata Preservation**: Tracks page numbers, positions, and source information

### 🤖 **Local LLM Integration**

- **Ollama Integration**: Uses local LLMs (completely offline)
- **Multiple Model Support**: Easy switching between different models
- **Custom Prompting**: Optimized prompts for RAG tasks

### 🔍 **Advanced Retrieval**

- **Semantic Search**: Uses SentenceTransformers for embedding generation
- **Vector Database**: ChromaDB for efficient similarity search
- **Relevance Filtering**: Configurable similarity thresholds
- **Source Tracking**: Complete traceability from answer to source

### 💻 **Web Interface**

- **Streamlit UI**: Clean, intuitive web interface
- **Document Management**: Upload and manage PDF documents
- **Real-time Query**: Interactive question-answering
- **Citation Display**: Visual source citations and evidence highlighting

## 🖥️ System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, macOS 10.14+, or Ubuntu 18.04+
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB free space for models and data
- **Python**: 3.8 or higher
- **Internet**: Required for initial setup and model downloads

### Recommended Requirements

- **RAM**: 16GB+ for better performance
- **CPU**: Multi-core processor (4+ cores)
- **GPU**: Optional, for faster embedding generation
- **SSD**: For faster database operations

## 🚀 Installation Guide

### Step 1: Install System Dependencies

#### Windows

```powershell
# Install Ollama
winget install Ollama.Ollama

# Install Tesseract OCR
winget install UB-Mannheim.TesseractOCR
```

#### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install ollama tesseract
```

#### Ubuntu/Debian

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install Tesseract
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

### Step 2: Setup Python Environment

```bash
# Clone or download the project
cd path/to/rag-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt
```

### Step 3: Download LLM Model

```bash
# Start Ollama service (in background)
ollama serve

# In a new terminal, download the model
ollama pull llama2:7b
```

## 📦 Dependencies

### Core Python Packages

```
# Document Processing
PyMuPDF==1.23.14          # PDF text extraction
pytesseract==0.3.10        # OCR functionality
Pillow==10.0.1             # Image processing

# Machine Learning
sentence-transformers==2.2.2  # Text embeddings
numpy==1.24.3              # Numerical computations
pandas==2.0.3              # Data manipulation

# Vector Database
chromadb==0.4.15           # Vector storage and search

# Web Interface
streamlit==1.28.1          # Web UI framework

# Utilities
requests==2.31.0           # HTTP requests
tqdm==4.66.1              # Progress bars
```

### System Dependencies

- **Ollama**: Local LLM inference engine
- **Tesseract OCR**: Text extraction from scanned documents
- **Python 3.8+**: Core runtime environment

### Optional Dependencies

```
# For GPU acceleration (optional)
torch==2.0.1+cu118         # PyTorch with CUDA support
transformers==4.34.0       # Hugging Face transformers
```

## ⚡ Quick Start

### 1. Verify Installation

```bash
# Test the complete system
python test_rag.py
```

Expected output:

```
✅ All dependencies are installed
✅ Ollama is running and accessible
✅ Tesseract OCR is available
✅ RAG system initialized successfully
✅ All tests passed!
```

### 2. Start the Web Interface

```bash
# Run the Streamlit app
streamlit run streamlit_app.py
```

### 3. Access the Application

Open your web browser and go to: `http://localhost:8501`

### 4. Upload and Query Documents

1. **Upload PDF**: Use the sidebar to upload PDF documents
2. **Wait for Processing**: Documents are automatically processed and indexed
3. **Ask Questions**: Use the main interface to query your documents
4. **View Results**: Get answers with citations and source highlights

## 📖 Usage

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Input     │    │  Text Chunking  │    │   Embeddings    │
│                 │────▶│                 │────▶│                 │
│ Native/OCR Text │    │ Sentence-Aware  │    │ SentenceTransf. │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │    │   Retrieval     │    │  Vector Store   │
│                 │────▶│                 │◀───│                 │
│ Natural Lang.   │    │ Semantic Search │    │   ChromaDB      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Final Answer   │    │   LLM Engine    │    │   Context       │
│                 │◀───│                 │◀───│                 │
│ With Citations  │    │ Ollama/Llama2   │    │ Retrieved Docs  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

The system can be configured by modifying the `RAGConfig` class in `rag_system.py`:

```python
@dataclass
class RAGConfig:
    # Text Processing
    chunk_size: int = 512              # Size of text chunks (characters)
    chunk_overlap: int = 50            # Overlap between chunks
    min_chunk_size: int = 100          # Minimum chunk size

    # Embedding & Search
    embedding_model: str = "all-MiniLM-L6-v2"  # Embedding model
    vector_db_path: str = "./chroma_db"         # Vector database path
    top_k_retrieval: int = 5           # Number of chunks to retrieve
    similarity_threshold: float = 0.3   # Minimum similarity score

    # LLM Configuration
    max_context_length: int = 2048     # Maximum context for LLM
    ollama_base_url: str = "http://localhost:11434"  # Ollama URL
    ollama_model: str = "llama2:7b"    # LLM model name
```

### Alternative Models

#### Embedding Models (Higher Quality)

```python
config.embedding_model = "all-mpnet-base-v2"     # Better quality, slower
config.embedding_model = "all-distilroberta-v1"  # Good balance
```

#### LLM Models

```bash
# Install alternative models
ollama pull codellama:7b    # Better for code
ollama pull mistral:7b      # Faster responses
ollama pull llama2:13b      # Higher quality (requires more RAM)
```

## 📁 Project Structure

```
rag-system/
├── 📄 rag_system.py          # Main RAG implementation
├── 🌐 streamlit_app.py       # Web interface
├── 🧪 test_rag.py           # System tests
├── 📋 requirements.txt       # Python dependencies
├── 📖 README.md             # Documentation (this file)
├── 📊 chroma_db/            # Vector database (auto-created)
│   ├── documents.json       # Document metadata
│   └── ...                 # ChromaDB files
├── 📝 rag_system.log        # System logs
└── 🐍 venv/                # Virtual environment (created during setup)
```

## 🔧 Advanced Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# .env file
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:7b
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB_PATH=./chroma_db
CHUNK_SIZE=512
TOP_K_RETRIEVAL=5
```

### Performance Tuning

#### For Low-Memory Systems (8GB RAM)

```python
config.chunk_size = 256          # Smaller chunks
config.top_k_retrieval = 3       # Fewer results
config.max_context_length = 1024 # Shorter context
```

#### For High-Performance Systems (16GB+ RAM)

```python
config.chunk_size = 1024         # Larger chunks
config.top_k_retrieval = 10      # More results
config.max_context_length = 4096 # Longer context
```

## 🛠️ Troubleshooting

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. **"Ollama Connection Failed"**

```bash
# Check if Ollama is running
ollama serve

# Verify model is available
ollama list

# Test model directly
ollama run llama2:7b "Hello, how are you?"
```

#### 2. **"Tesseract is not installed or not in PATH"**

**Windows:**

1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Or install via winget: `winget install UB-Mannheim.TesseractOCR`
3. Restart your terminal/application

**macOS:**

```bash
brew install tesseract
```

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

#### 3. **"Module not found" Errors**

```bash
# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 4. **"Out of Memory" Errors**

- Reduce chunk size: `config.chunk_size = 256`
- Process fewer documents at once
- Use smaller embedding models
- Close other applications

#### 5. **Slow Performance**

- Use GPU acceleration (install PyTorch with CUDA)
- Increase chunk size to reduce total chunks
- Use faster embedding models
- Use SSD storage for vector database

#### 6. **PDF Processing Fails**

```python
# Check PDF file
import fitz
doc = fitz.open("your_file.pdf")
print(f"Pages: {len(doc)}")
print(f"Encrypted: {doc.needs_pass}")
```

#### 7. **Streamlit App Won't Start**

```bash
# Check if port is available
netstat -an | findstr :8501

# Use different port
streamlit run streamlit_app.py --server.port 8502
```

### 🔧 System Diagnostics

Run the diagnostic script to check your setup:

```bash
python test_rag.py
```

### 📋 Dependency Verification

```bash
# Check Python version
python --version

# Check pip packages
pip list | grep -E "(streamlit|chromadb|sentence-transformers)"

# Check system commands
ollama --version
tesseract --version
```

### 🐛 Debug Mode

Enable detailed logging by modifying `rag_system.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

### 📞 Getting Help

1. **Check Logs**: Review `rag_system.log` for detailed error messages
2. **Run Tests**: Execute `python test_rag.py` for system diagnostics
3. **Check Dependencies**: Ensure all requirements are installed
4. **Restart Services**: Restart Ollama and the Streamlit app
5. **System Resources**: Monitor RAM and CPU usage

## 🎯 Performance Optimization

### For Different Use Cases

#### Academic Research (High Quality)

```python
config = RAGConfig(
    chunk_size=1024,
    embedding_model="all-mpnet-base-v2",
    ollama_model="llama2:13b",
    top_k_retrieval=10
)
```

#### Production Deployment (Balanced)

```python
config = RAGConfig(
    chunk_size=512,
    embedding_model="all-MiniLM-L6-v2",
    ollama_model="llama2:7b",
    top_k_retrieval=5
)
```

#### Quick Prototyping (Fast)

```python
config = RAGConfig(
    chunk_size=256,
    embedding_model="all-MiniLM-L6-v2",
    ollama_model="mistral:7b",
    top_k_retrieval=3
)
```

## 📝 Usage Examples

### Web Interface Usage

1. **Start the application**: `streamlit run streamlit_app.py`
2. **Upload documents**: Use the sidebar file uploader
3. **Ask questions**: Type queries in the main interface
4. **Review answers**: See responses with citations and source highlights

### Programmatic Usage

#### Basic Usage

```python
from rag_system import RAGSystem

# Initialize
rag = RAGSystem()

# Add documents
doc_id = rag.ingest_document("research_paper.pdf")

# Query
response = rag.query("What are the main findings?")
print(response.answer)
```

#### Advanced Usage

```python
from rag_system import RAGSystem, RAGConfig

# Custom configuration
config = RAGConfig(
    chunk_size=1024,
    top_k_retrieval=10,
    similarity_threshold=0.4
)

# Initialize with config
rag = RAGSystem(config)

# Batch processing
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for doc_path in documents:
    doc_id = rag.ingest_document(doc_path)
    print(f"Processed: {doc_path}")

# Filtered queries
response = rag.query(
    "What is the methodology?",
    document_filter=doc_id  # Query specific document
)

# Review detailed results
for result in response.retrieval_results:
    print(f"Score: {result.similarity_score:.3f}")
    print(f"Source: Page {result.chunk.page_number}")
    print(f"Content: {result.chunk.content[:200]}...")
```

## 🎬 Video Demonstration

For your assignment submission, create a screen recording showing:

1. **System Setup**: Running `python test_rag.py`
2. **Document Upload**: Adding a PDF through the web interface
3. **Query Processing**: Asking questions and getting answers
4. **Citation Review**: Showing source highlighting and references
5. **Performance Metrics**: Displaying processing times and statistics

## 📄 Assignment Deliverables Checklist

- ✅ **High-Level Design (HLD)**: Architecture diagram included
- ✅ **Ingestion Pipeline**: PDF processing with native and OCR support
- ✅ **Chunking & Embeddings**: Sentence-aware chunking with SentenceTransformers
- ✅ **Vector Index**: ChromaDB implementation
- ✅ **Retriever**: Semantic search with relevance scoring
- ✅ **Web UI**: Streamlit interface for document upload and querying
- ✅ **Documentation**: Comprehensive README with setup instructions
- ✅ **Code Quality**: Clean, modular architecture with error handling
- ✅ **Testing**: Automated test script for system verification

## 📜 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For issues and questions:

1. Run the diagnostic script: `python test_rag.py`
2. Check the troubleshooting section above
3. Review system logs in `rag_system.log`
4. Ensure all dependencies are properly installed
5. Verify Ollama is running and models are available

---

**Built with ❤️ for educational and research purposes**
