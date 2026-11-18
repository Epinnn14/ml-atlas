@echo off
cd /d "%~dp0"
where py >nul 2>nul && (py serve.py & goto :eof)
where python >nul 2>nul && (python serve.py & goto :eof)
echo Python 3 was not found. Install it from https://www.python.org/downloads/ and run this file again.
pause
