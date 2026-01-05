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
        print("Installing requirements...")
        run_command(f"{sys.executable} -m pip install -r {req_path}")
    
    # 3. Explicitly install imageio[ffmpeg], imageio[freeimage] and numpy
    print("Installing imageio[ffmpeg], imageio[freeimage] and numpy...")
    run_command(f"{sys.executable} -m pip install imageio[ffmpeg] imageio[freeimage] numpy")

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

    # 6. Create models/loras and download LoRA
    loras_path = comfyui_path / "models" / "loras"
    loras_path.mkdir(parents=True, exist_ok=True)
    lora_url = "https://huggingface.co/ProGamerGov/human-360-lora-flux-dev/resolve/main/human_360diffusion_lora_flux_dev_v1.safetensors"
    lora_file = loras_path / "human_360diffusion_lora_flux_dev_v1.safetensors"
    
    if not lora_file.exists():
        print("Downloading LoRA...")
        try:
            urllib.request.urlretrieve(lora_url, lora_file)
            print("LoRA downloaded.")
        except Exception as e:
            print(f"Failed to download LoRA: {e}")

    # 7. Symlink or copy the suite
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
