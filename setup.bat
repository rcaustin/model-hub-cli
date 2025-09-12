@echo off
echo Setting up model-hub-cli environment...

REM Try different Python commands
set PYTHON_CMD=
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python
    goto :python_found
)

python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=python3
    goto :python_found
)

py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :python_found
)

echo ERROR: Python is not installed or not in PATH
echo.
echo Please install Python 3.8+ from one of these sources:
echo 1. Official Python website: https://www.python.org/downloads/
echo    - Make sure to check "Add Python to PATH" during installation
echo 2. Microsoft Store: Search for "Python 3.12"
echo.
echo After installation, restart your terminal and try again.
exit /b 1

:python_found
echo Found Python: %PYTHON_CMD%
%PYTHON_CMD% --version

REM Create a virtual environment in the .venv directory
echo Creating virtual environment...
%PYTHON_CMD% -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

REM Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install Python dependencies
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo ✅ Setup complete! Virtual environment created in .venv/
echo To activate the environment manually, run: .venv\Scripts\activate.bat
echo To run the application, use: python run.py [command]