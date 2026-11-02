@echo off
rem Starts the desktop client with .venv\Scripts\pythonw.exe when it exists, otherwise pythonw.exe from PATH.
set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=pythonw.exe"
start "" "%PYTHONW%" -s "%~dp0launch.pyw"
