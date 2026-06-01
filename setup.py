#!/usr/bin/env python3
"""
Setup script for RAG System
Automates the installation and configuration process
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} is not compatible. Requires Python 3.8+")
        return False

def install_system_dependencies():
    """Install system dependencies based on OS"""
    system = platform.system()
    print(f"\n🖥️ Detected OS: {system}")
    
    if system == "Windows":
        print("📋 For Windows, please manually install:")
        print("1. Ollama: winget install Ollama.Ollama")
        print("2. Tesseract: winget install UB-Mannheim.TesseractOCR")
        print("3. Or download from: https://github.com/UB-Mannheim/tesseract/wiki")
        
    elif system == "Darwin":  # macOS
        print("🍎 Installing dependencies for macOS...")
        run_command("brew install ollama tesseract", "Installing Ollama and Tesseract")
        
    elif system == "Linux":
        print("🐧 Installing dependencies for Linux...")
        run_command("curl -fsSL https://ollama.ai/install.sh | sh", "Installing Ollama")
        run_command("sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng", "Installing Tesseract")
    
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if not venv_path.exists():
        if run_command(f"{sys.executable} -m venv venv", "Creating virtual environment"):
            print("✅ Virtual environment created")
        else:
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script based on OS
    if platform.system() == "Windows":
        activate_script = "venv\\Scripts\\activate"
        pip_path = "venv\\Scripts\\pip"
    else:
        activate_script = "source venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    print(f"💡 To activate virtual environment, run: {activate_script}")
    return pip_path

def install_python_dependencies(pip_path):
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip first
    run_command(f"{pip_path} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if run_command(f"{pip_path} install -r requirements.txt", "Installing Python packages"):
        print("✅ All Python dependencies installed")
        return True
    else:
        print("❌ Failed to install Python dependencies")
        return False

def download_ollama_model():
    """Download required Ollama model"""
    print("\n🤖 Setting up LLM model...")
    print("📋 Please run these commands manually:")
    print("1. ollama serve  (start Ollama service)")
    print("2. ollama pull llama2:7b  (download model - this may take a while)")
    return True

def verify_installation():
    """Verify that everything is installed correctly"""
    print("\n🔍 Verifying installation...")
    
    # Check if test script exists
    if Path("test_rag.py").exists():
        print("💡 Run 'python test_rag.py' to verify your installation")
    else:
        print("⚠️ test_rag.py not found - manual verification required")
    
    print("\n🚀 To start the application:")
    if platform.system() == "Windows":
        print("1. venv\\Scripts\\activate")
    else:
        print("1. source venv/bin/activate")
    print("2. streamlit run streamlit_app.py")
    print("3. Open http://localhost:8501 in your browser")

def main():
    """Main setup function"""
    print("🎯 RAG System Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    install_system_dependencies()
    
    # Setup Python environment
    pip_path = setup_virtual_environment()
    if not pip_path:
        print("❌ Failed to setup virtual environment")
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies(pip_path):
        sys.exit(1)
    
    # Setup Ollama model
    download_ollama_model()
    
    # Final verification
    verify_installation()
    
    print("\n🎉 Setup completed!")
    print("📖 Check README.md for detailed usage instructions")
    print("🐛 If you encounter issues, run 'python test_rag.py' for diagnostics")

if __name__ == "__main__":
    main()
