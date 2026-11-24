@echo off
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo Missing .venv. Run: py -3.11 -m venv .venv
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0loop.py" %*
exit /b %errorlevel%
