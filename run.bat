@echo off
echo [run.bat] start
title Xiaoban - Start
chcp 65001 >nul
cd /d "%~dp0"
if not exist "main.py" (
    echo  [ERROR] main.py not found. Run this bat in project folder: AI
    echo  Path: %CD%
    pause
    exit /b 1
)
echo.
echo  [Xiaoban] Starting...
echo  Dir: %CD%
echo.

REM use python or py
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY where py >nul 2>nul && set "PY=py"
if not defined PY (
    echo  [ERROR] Python not found. Install Python and add to PATH.
    echo  https://www.python.org/downloads/
    goto :fail
)

echo  Using: %PY%
%PY% --version 2>nul
echo.

REM check deps
%PY% -c "import fastapi, uvicorn, openai, aiosqlite" 2>nul
if errorlevel 1 (
    echo  [TIP] Missing deps. Run: %PY% -m pip install -r requirements.txt
    echo.
    goto :fail
)

echo  Starting server...
echo.
%PY% main.py
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo  --------
echo  Failed. Fix above error:
echo  - Missing module: %PY% -m pip install -r requirements.txt
echo  - No API key: copy .env.example to .env and set OPENAI_API_KEY
echo  --------
echo.
echo  Press any key to close...
pause >nul
exit /b 1
