#!/usr/bin/env python3
"""
Test script for RAG System
This script demonstrates the basic functionality of the RAG system
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem, RAGConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag_system():
    """Test the RAG system functionality"""
    
    print("🚀 Starting RAG System Test")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        print("1. Initializing RAG System...")
        rag = RAGSystem()
        print("✅ RAG System initialized successfully!")
        
        # Get system stats
        print("\n2. System Statistics:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Check if documents exist
        documents = rag.get_documents()
        print(f"\n3. Existing Documents: {len(documents)}")
        
        if documents:
            for doc in documents:
                print(f"   📄 {doc.filename} ({doc.total_pages} pages)")
        else:
            print("   No documents found. Please upload PDFs using the web interface.")
        
        # Test query if documents exist
        if documents:
            print("\n4. Testing Query...")
            test_question = "What is this document about?"
            response = rag.query(test_question)
            
            print(f"   Question: {test_question}")
            print(f"   Answer: {response.answer[:200]}...")
            print(f"   Sources: {len(response.citations)}")
            print(f"   Processing time: {response.processing_time:.2f}s")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test failed with exception:")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("🔍 Checking Dependencies")
    print("=" * 30)
    
    required_packages = [
        'fitz',  # PyMuPDF
        'pytesseract',
        'PIL',  # Pillow
        'numpy',
        'pandas',
        'sentence_transformers',
        'chromadb',
        'streamlit',
        'tqdm',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'fitz':
                import fitz
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_ollama():
    """Check if Ollama is running and model is available"""
    
    print("\n🤖 Checking Ollama")
    print("=" * 20)
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            print(f"✅ Ollama is running")
            print(f"   Available models: {model_names}")
            
            if 'llama2:7b' in model_names:
                print("✅ llama2:7b model is available")
                return True
            else:
                print("⚠️  llama2:7b model not found")
                print("   Run: ollama pull llama2:7b")
                return False
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is installed and running")
        print("   Run: ollama serve")
        return False

def main():
    """Main test function"""
    
    print("🧪 RAG System Test Suite")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check Ollama
    ollama_ok = check_ollama()
    
    if not deps_ok:
        print("\n❌ Dependencies check failed. Please install required packages.")
        return False
    
    if not ollama_ok:
        print("\n⚠️  Ollama check failed. The system will work but LLM queries may fail.")
    
    # Test RAG system
    print("\n" + "=" * 60)
    test_ok = test_rag_system()
    
    print("\n" + "=" * 60)
    if test_ok:
        print("🎉 All tests passed! The RAG system is ready to use.")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Upload PDF documents")
        print("3. Start asking questions!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return test_ok

if __name__ == "__main__":
    main()
