@echo off
echo ========================================
echo   Galactic Data Forge - Setup
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8+ first.
    pause
    exit /b
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Starting Galactic Data Forge...
streamlit run app.py

pause