@echo off
echo 🚀 Starting RAG System
echo ====================

echo 📋 Checking if virtual environment exists...
if not exist "venv\" (
    echo ❌ Virtual environment not found!
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

echo ✅ Virtual environment found

echo 🔧 Activating virtual environment...
call venv\Scripts\activate

echo 🤖 Checking Ollama service...
ollama list >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Ollama not running, starting service...
    start ollama serve
    echo ⏳ Waiting for Ollama to start...
    timeout /t 10 /nobreak > NUL
)

echo ✅ Ollama is running

echo 🧪 Running system test...
python test_rag.py
if errorlevel 1 (
    echo ❌ System test failed! Check the output above.
    pause
    exit /b 1
)

echo ✅ System test passed!

echo 🌐 Starting web interface...
echo Opening http://localhost:8501 in your default browser...
start http://localhost:8501

echo 🎯 Starting Streamlit application...
streamlit run streamlit_app.py --server.port 8501

echo.
echo 📋 Application stopped. Press any key to exit...
pause
