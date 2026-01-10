# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

import os
import subprocess
import sys
import shutil
from pathlib import Path
import urllib.request

def run_command(command, cwd=None):
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    # Determine paths
    script_dir = Path(__file__).parent.absolute()
    suite_root = script_dir.parent
    
    # 1. Check if ComfyUI is installed
    current_dir = Path.cwd()
    
    # Check if we are inside ComfyUI/custom_nodes/...
    # suite_root is .../ComfyUI-360-HDRI-Suite
    potential_comfy_root = suite_root.parent.parent
    if (potential_comfy_root / "main.py").exists() and (potential_comfy_root / "custom_nodes").exists():
        print(f"Detected running inside ComfyUI at {potential_comfy_root}")
        comfyui_path = potential_comfy_root
    else:
        # Fallback to looking for ComfyUI in current dir or cloning
        comfyui_path = current_dir / "ComfyUI"
        if not comfyui_path.exists():
            print("ComfyUI not found in current directory. Cloning...")
            run_command("git clone https://github.com/comfyanonymous/ComfyUI.git")
        else:
            print("ComfyUI found.")

    # 2. Install standard requirements
    req_path = suite_root / "requirements.txt"
    if req_path.exists():
        print(f"Installing requirements from {req_path}...")
        try:
            run_command(f'"{sys.executable}" -m pip install -r "{req_path}"')
        except Exception as e:
            print(f"Warning: Failed to install requirements via pip: {e}")
            print("Please ensure you have internet access and pip is configured correctly.")

    print("Installation complete.")

    # 3b. Download FreeImage binaries (required for EXR)
    print("Downloading FreeImage binaries...")
    try:
        run_command(f'{sys.executable} -c "import imageio; imageio.plugins.freeimage.download()"')
    except Exception as e:
        print(f"Warning: Failed to download FreeImage binaries: {e}")
        print("EXR saving might not work without them.")

    # 4. Clone ComfyUI-Manager into custom_nodes
    custom_nodes_path = comfyui_path / "custom_nodes"
    manager_path = custom_nodes_path / "ComfyUI-Manager"
    if not manager_path.exists():
        print("Cloning ComfyUI-Manager...")
        run_command("git clone https://github.com/ltdrdata/ComfyUI-Manager.git", cwd=custom_nodes_path)

    # 5. Clone ComfyUI-Seamless-Tiling
    seamless_path = custom_nodes_path / "ComfyUI-Seamless-Tiling"
    if not seamless_path.exists():
        print("Cloning ComfyUI-Seamless-Tiling...")
        try:
            # Using the standard repo for seamless tiling
            run_command("git clone https://github.com/spinagon/ComfyUI-seamless-tiling.git", cwd=custom_nodes_path)
        except SystemExit:
            print("Warning: Failed to clone ComfyUI-Seamless-Tiling. Continuing installation...")

    # 5b. Clone ComfyUI_IPAdapter_plus (Required for accurate terrain replication)
    ipadapter_path = custom_nodes_path / "ComfyUI_IPAdapter_plus"
    if not ipadapter_path.exists():
        print("Cloning ComfyUI_IPAdapter_plus...")
        try:
            run_command("git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git", cwd=custom_nodes_path)
        except SystemExit:
            print("Warning: Failed to clone ComfyUI_IPAdapter_plus.")

    # 6. Create models/loras and download LoRAs
    loras_path = comfyui_path / "models" / "loras"
    loras_path.mkdir(parents=True, exist_ok=True)
    
    # 6a. 360 Flux LoRA
    lora_360_url = "https://huggingface.co/ProGamerGov/human-360-lora-flux-dev/resolve/main/human_360diffusion_lora_flux_dev_v1.safetensors"
    lora_360_file = loras_path / "human_360diffusion_lora_flux_dev_v1.safetensors"
    
    if not lora_360_file.exists():
        print("Downloading 360 Flux LoRA...")
        try:
            urllib.request.urlretrieve(lora_360_url, lora_360_file)
            print("LoRA downloaded.")
        except Exception as e:
            print(f"Failed to download 360 LoRA: {e}")

    # 6b. Seamless Texture LoRA (SDXL)
    # TODO: Update URL when public
    seamless_lora_url = "" 
    seamless_lora_file = loras_path / "seamless_texture.safetensors"
    
    if seamless_lora_url and not seamless_lora_file.exists():
        print("Downloading Seamless Texture LoRA...")
        try:
            urllib.request.urlretrieve(seamless_lora_url, seamless_lora_file)
            print("Seamless LoRA downloaded.")
        except Exception as e:
            print(f"Failed to download Seamless LoRA: {e}")

    # 7. Download IPAdapter Models (Required for Terrain Workflow)
    ipadapter_models_path = comfyui_path / "models" / "ipadapter"
    ipadapter_models_path.mkdir(parents=True, exist_ok=True)
    
    clip_vision_path = comfyui_path / "models" / "clip_vision"
    clip_vision_path.mkdir(parents=True, exist_ok=True)
    
    # IPAdapter Plus SDXL Model
    ipa_model_url = "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
    ipa_model_file = ipadapter_models_path / "ip-adapter-plus_sdxl_vit-h.safetensors"
    
    if not ipa_model_file.exists():
        print("Downloading IPAdapter Plus SDXL Model...")
        try:
            urllib.request.urlretrieve(ipa_model_url, ipa_model_file)
            print("IPAdapter Model downloaded.")
        except Exception as e:
            print(f"Failed to download IPAdapter Model: {e}")
            
    # CLIP Vision Model (ViT-H)
    clip_vision_url = "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
    clip_vision_file = clip_vision_path / "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    
    if not clip_vision_file.exists():
        print("Downloading CLIP Vision Model...")
        try:
            urllib.request.urlretrieve(clip_vision_url, clip_vision_file)
            print("CLIP Vision Model downloaded.")
        except Exception as e:
            print(f"Failed to download CLIP Vision Model: {e}")

    # 7. Setup Ollama
    print("\n--- Setting up Ollama ---")
    ollama_installed = False
    try:
        subprocess.check_call("ollama --version", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Ollama is already installed.")
        ollama_installed = True
    except subprocess.CalledProcessError:
        print("Ollama not found.")
        if sys.platform == "win32":
            print("Attempting to install Ollama via winget...")
            try:
                run_command("winget install Ollama.Ollama")
                ollama_installed = True
            except Exception as e:
                print(f"Failed to install Ollama via winget: {e}")
                print("Please install Ollama manually from https://ollama.com/download")
        else:
            print("Please install Ollama manually from https://ollama.com/download")

    if ollama_installed:
        # Check if running
        try:
            import requests
            try:
                requests.get("http://127.0.0.1:11434")
                print("Ollama service is running.")
            except requests.exceptions.ConnectionError:
                print("Ollama service is NOT running. Attempting to start...")
                if sys.platform == "win32":
                    subprocess.Popen("ollama serve", shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen(["ollama", "serve"])
                print("Started Ollama in background.")
        except ImportError:
            print("Requests library not installed yet (will be installed with requirements). Skipping service check.")

        # Pull llava model
        print("Pulling 'llava' model (this may take a while)...")
        try:
            run_command("ollama pull llava")
            print("Model 'llava' pulled successfully.")
        except Exception as e:
            print(f"Failed to pull model: {e}")
            print("You may need to run 'ollama pull llava' manually.")

    # 8. Symlink or copy the suite
    target_suite_path = custom_nodes_path / "ComfyUI-360-HDRI-Suite"
    
    if target_suite_path.exists():
        print(f"Target path {target_suite_path} already exists. Skipping link/copy.")
    else:
        print(f"Linking {suite_root} to {target_suite_path}...")
        try:
            # Try symlink first
            os.symlink(suite_root, target_suite_path)
            print("Symlink created.")
        except OSError:
            print("Symlink failed (admin rights might be needed on Windows). Copying instead...")
            try:
                shutil.copytree(suite_root, target_suite_path)
                print("Copied successfully.")
            except Exception as e:
                print(f"Copy failed: {e}")

if __name__ == "__main__":
    main()
