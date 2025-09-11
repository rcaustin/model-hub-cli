@echo off
REM Activate the virtual environment
call .venv\bin\activate.bat
REM Run the program script
python run.py %*
