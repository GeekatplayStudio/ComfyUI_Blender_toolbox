import os
import sys
import subprocess
import shutil
import urllib.request

# Configuration
COMFY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CUSTOM_NODES_PATH = os.path.join(COMFY_PATH, "custom_nodes")
THIS_NODE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(COMFY_PATH, "models", "ubsoft_pbr")
REPOS_DIR = os.path.join(THIS_NODE_PATH, "repos")
CHORD_REPO_URL = "https://github.com/ubisoft/ubisoft-laforge-chord.git"
CHORD_MODEL_URL = "https://huggingface.co/Ubisoft/ubisoft-laforge-chord/resolve/main/chord_v1.safetensors"
WORKFLOW_SOURCE = os.path.join(THIS_NODE_PATH, "workflows", "Geekatplay_PBR_Texture_Studio_workflow.json")

def install_requirements():
    print(">>> Installing Python dependencies for PBR Extractor...")
    # Core requirements (excluding torch to avoid environment conflicts)
    deps = ["numpy", "pillow", "timm", "einops", "omegaconf", "safetensors", "diffusers", "transformers", "accelerate"]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)

def clone_chord_repo():
    print(f">>> Cloning Ubisoft CHORD repository into {REPOS_DIR}...")
    chord_path = os.path.join(REPOS_DIR, "ubisoft-laforge-chord")
    
    if not os.path.exists(REPOS_DIR):
        os.makedirs(REPOS_DIR)
        
    if not os.path.exists(chord_path):
        subprocess.check_call(["git", "clone", CHORD_REPO_URL, chord_path])
    else:
        print(">>> CHORD repo already exists. Pulling latest...")
        subprocess.check_call(["git", "-C", chord_path, "pull"])

def download_model():
    print(f">>> Creating model directory: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, "chord_v1.safetensors")
    if not os.path.exists(model_path):
        print(f">>> Downloading CHORD model weights (this may take a while)...")
        print(f"    Source: Ubisoft/ubisoft-laforge-chord")
        
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="Ubisoft/ubisoft-laforge-chord", filename="chord_v1.safetensors", local_dir=MODELS_DIR, local_dir_use_symlinks=False)
            print(">>> Download complete!")
        except Exception as e:
            print(f"\n!!! Error downloading model: {e}")
            print("\nIMPORTANT: This model is GATED.")
            print("1. Go to https://huggingface.co/Ubisoft/ubisoft-laforge-chord and accept the license.")
            print("2. Run 'huggingface-cli login' in your terminal and enter your token.")
            print("3. OR download 'chord_v1.safetensors' manually and place it in:")
            print(f"{MODELS_DIR}\n")
    else:
        print(">>> CHORD model weights found.")

def deploy_workflow():
    if os.path.exists(WORKFLOW_SOURCE):
        # Determine where to put it. 
        # For now, we just ensure it's in the repo's nodes/workflows folder which is accessible.
        # But let's try to copy it to the root or a 'workflows' folder if the user keeps one.
        print(f">>> PBR Extractor Workflow is located at: {WORKFLOW_SOURCE}")
        print(">>> You can load this directly in ComfyUI.")
    else:
        print("!!! Warning: Workflow file not found in source.")

def main():
    print("===========================================")
    print("   PBR Extractor & Texture Studio Installer   ")
    print("===========================================")
    
    install_requirements()
    clone_chord_repo()
    download_model()
    deploy_workflow()
    
    print("\n>>> Installation Complete!")
    print(">>> 1. Restart ComfyUI")
    print(f">>> 2. Load the workflow from: {WORKFLOW_SOURCE}")
    print("===========================================")

if __name__ == "__main__":
    main()
