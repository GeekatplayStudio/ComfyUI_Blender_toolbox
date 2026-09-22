@echo off
REM (c) Geekatplay Studio - Vladimir Chopine
REM ComfyUI-Blender-Toolbox - AI Scene Builder setup
setlocal
cd /d "%~dp0\.."

set "PYTHON=python"
if exist "..\..\..\python_embeded\python.exe" (
    set "PYTHON=..\..\..\python_embeded\python.exe"
    echo Found ComfyUI Portable Python.
)

"%PYTHON%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Use the ComfyUI portable version or install Python 3.10+.
    pause
    exit /b 1
)

"%PYTHON%" installer\install_ai_builder.py --smoke-test %*
echo.
pause
