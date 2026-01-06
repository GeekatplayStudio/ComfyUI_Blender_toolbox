@echo off
REM (c) Geekatplay Studio
REM ComfyUI-360-HDRI-Suite
setlocal enabledelayedexpansion

echo ========================================================
echo ComfyUI-360-HDRI-Suite Installer
echo ========================================================

cd /d "%~dp0"

:: Attempt to find Python
set "PYTHON=python"

:: Check for ComfyUI Portable Python (relative to this script if inside custom_nodes)
if exist "..\..\..\python_embeded\python.exe" (
    set "PYTHON=..\..\..\python_embeded\python.exe"
    echo Found ComfyUI Portable Python.
)

:: Check if python is available
"%PYTHON%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python or use the ComfyUI portable version.
    pause
    exit /b 1
)

:: Run the installer
"%PYTHON%" installer\install.py

echo.
echo ========================================================
if %ERRORLEVEL% EQU 0 (
    echo Installation Finished Successfully.
) else (
    echo Installation Failed.
)
echo ========================================================
pause
