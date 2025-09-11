@echo off
REM Create a virtual environment in the .venv directory
python -m venv .venv

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install Python dependencies
pip install -r requirements.txt