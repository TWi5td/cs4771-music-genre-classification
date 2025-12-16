@echo off
REM =============================================================================
REM Music Genre Classifier - Windows Setup Script
REM For Windows 11 Development
REM =============================================================================

echo ==============================================
echo   Music Genre Classifier - Windows Setup
echo ==============================================

set PROJECT_DIR=%~dp0..
set VENV_DIR=%PROJECT_DIR%\venv

echo.
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
python --version
echo OK

echo.
echo [2/5] Creating virtual environment...
if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
    echo Created virtual environment
) else (
    echo Virtual environment already exists
)

echo.
echo [3/5] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo [4/5] Installing dependencies...
pip install --upgrade pip
pip install -r "%PROJECT_DIR%\requirements.txt"

echo.
echo [5/5] Creating directories...
if not exist "%PROJECT_DIR%\data\raw" mkdir "%PROJECT_DIR%\data\raw"
if not exist "%PROJECT_DIR%\data\processed" mkdir "%PROJECT_DIR%\data\processed"
if not exist "%PROJECT_DIR%\models" mkdir "%PROJECT_DIR%\models"
if not exist "%PROJECT_DIR%\static\uploads" mkdir "%PROJECT_DIR%\static\uploads"

echo.
echo ==============================================
echo Setup Complete!
echo ==============================================
echo.
echo To run the application:
echo   1. Open Command Prompt in project directory
echo   2. Run: venv\Scripts\activate
echo   3. Run: python app.py
echo.
echo The web interface will be available at:
echo   http://localhost:5000
echo.
echo Note: Download the GTZAN dataset and run preprocessing:
echo   python src\preprocess.py
echo   python src\train.py
echo.
pause
