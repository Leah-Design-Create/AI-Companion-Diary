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

REM use py or python (Windows 上优先 py，避免命中错误解释器)
set "PY="
where py >nul 2>nul && set "PY=py"
if not defined PY where python >nul 2>nul && set "PY=python"
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
    goto :fail_deps
)

REM auto-install bcrypt if missing (multi-user auth)
%PY% -c "import bcrypt" 2>nul
if errorlevel 1 (
    echo  [TIP] Installing bcrypt for multi-user auth...
    %PY% -m pip install "bcrypt>=4.0.0" -q
)

REM check .env and OPENAI_API_KEY
if not exist ".env" (
    echo  [TIP] Missing .env file. Run: copy .env.example .env
    echo.
    goto :fail_env
)
%PY% -c "import os, pathlib; p=pathlib.Path('.env'); s=p.read_text(encoding='utf-8', errors='ignore') if p.exists() else ''; ok=any((ln.strip().startswith('OPENAI_API_KEY=') and len(ln.split('=',1)[1].strip())>0) for ln in s.splitlines() if not ln.strip().startswith('#')); raise SystemExit(0 if ok else 1)" 2>nul
if errorlevel 1 (
    echo  [TIP] No OPENAI_API_KEY in .env
    echo.
    goto :fail_env
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
echo  - Please check the message above.
echo  --------
echo.
echo  Press any key to close...
pause >nul
exit /b 1

:fail_deps
echo.
echo  --------
echo  Failed. Fix above error:
echo  - Missing module: %PY% -m pip install -r requirements.txt
echo  --------
echo.
echo  Press any key to close...
pause >nul
exit /b 1

:fail_env
echo.
echo  --------
echo  Failed. Fix above error:
echo  - No API key: copy .env.example to .env and set OPENAI_API_KEY
echo  --------
echo.
echo  Press any key to close...
pause >nul
exit /b 1
