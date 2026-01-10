@echo off
echo "Installing PBR Extractor & Texture Studio..."
if exist "..\..\..\..\python_embeded\python.exe" (
    ..\..\..\..\python_embeded\python.exe install_pbr_extractor.py
) else (
    echo "Embedded Python not found, trying system python..."
    python install_pbr_extractor.py
)
pause
