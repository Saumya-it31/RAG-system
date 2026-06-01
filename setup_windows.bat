@echo off
echo 🎯 RAG System Quick Setup for Windows
echo ====================================

echo.
echo 📋 This script will help you set up the RAG system
echo.

echo 🔧 Step 1: Installing system dependencies...
echo Installing Ollama...
winget install Ollama.Ollama
echo.

echo Installing Tesseract OCR...
winget install UB-Mannheim.TesseractOCR
echo.

echo 🐍 Step 2: Setting up Python environment...
python -m venv venv
echo.

echo 📦 Step 3: Installing Python packages...
venv\Scripts\activate && venv\Scripts\pip install --upgrade pip
venv\Scripts\activate && venv\Scripts\pip install -r requirements.txt
echo.

echo 🤖 Step 4: Starting Ollama service...
start ollama serve
echo Ollama service started in background
echo.

echo ⏳ Step 5: Downloading LLM model (this may take a few minutes)...
timeout /t 5 /nobreak > NUL
ollama pull llama2:7b
echo.

echo ✅ Setup completed!
echo.
echo 🚀 To run the application:
echo 1. venv\Scripts\activate
echo 2. streamlit run streamlit_app.py
echo 3. Open http://localhost:8501 in your browser
echo.
echo 🧪 To test the system:
echo venv\Scripts\activate && python test_rag.py
echo.

pause
