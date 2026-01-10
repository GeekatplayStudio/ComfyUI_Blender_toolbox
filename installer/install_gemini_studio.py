import os
import sys
import subprocess
import shutil
import urllib.request

# Configuration
COMFY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CUSTOM_NODES_PATH = os.path.join(COMFY_PATH, "custom_nodes")
THIS_NODE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(COMFY_PATH, "models", "gemini_pbr")
WORKFLOW_SOURCE = os.path.join(THIS_NODE_PATH, "workflows", "Gemini_PBR_Texture_Studio_workflow.json")
WORKFLOW_DEST_DIR = os.path.join(COMFY_PATH, "user", "default", "workflows") # Hypothetical standard location

def install_requirements():
    print(">>> Installing Python dependencies for Gemini PBR Studio...")
    req_file = os.path.join(THIS_NODE_PATH, "requirements.txt")
    if os.path.exists(req_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    else:
        # Minimal requirements if file doesn't exist
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "numpy", "pillow"])

def setup_directories():
    print(f">>> Creating model directory: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Create a readme in the models dir
    readme_path = os.path.join(MODELS_DIR, "PUT_MODELS_HERE.txt")
    with open(readme_path, "w") as f:
        f.write("Place optional PBR inference models (CHORD, DepthAnything, etc.) in this folder.\n")
        f.write("The Gemini PBR Extractor will check this folder for overriding weights.")

def deploy_workflow():
    if os.path.exists(WORKFLOW_SOURCE):
        # Determine where to put it. 
        # For now, we just ensure it's in the repo's nodes/workflows folder which is accessible.
        # But let's try to copy it to the root or a 'workflows' folder if the user keeps one.
        print(f">>> Gemini PBR Workflow is located at: {WORKFLOW_SOURCE}")
        print(">>> You can load this directly in ComfyUI.")
    else:
        print("!!! Warning: Workflow file not found in source.")

def main():
    print("===========================================")
    print("   Gemini PBR & Texture Studio Installer   ")
    print("===========================================")
    
    install_requirements()
    setup_directories()
    deploy_workflow()
    
    print("\n>>> Installation Complete!")
    print(">>> 1. Restart ComfyUI")
    print(f">>> 2. Load the workflow from: {WORKFLOW_SOURCE}")
    print("===========================================")

if __name__ == "__main__":
    main()
