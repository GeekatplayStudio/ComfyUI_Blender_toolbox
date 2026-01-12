import argparse
import os
import sys
import subprocess
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime

bl_info = {
    "name": "ComfyUI 360 Auto Rigger CLI",
    "author": "Geekatplay Studio",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "CLI",
    "description": "CLI orchestrator for auto-rigging",
    "category": "Development",
}

# --- Configuration ---
COMFY_SERVER = "http://127.0.0.1:8188"
BLENDER_EXEC = "blender" # Assume in PATH or set via env var
# Now we are in blender_scripts, so go up one level to root
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
WORKFLOW_TEMPLATE = os.path.join(ROOT_DIR, "workflows", "autorig_api.json")
# clean_and_rig.py is in the same directory as this script
BLENDER_SCRIPT = os.path.join(os.path.dirname(__file__), "clean_and_rig.py")

# --- ComfyUI Client Utils ---

def queue_prompt(prompt, client_id=None):
    p = {"prompt": prompt}
    if client_id:
        p["client_id"] = client_id
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"{COMFY_SERVER}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())

def upload_image(filepath):
    import requests
    with open(filepath, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{COMFY_SERVER}/upload/image", files=files)
    return response.json()

def check_comfy_connection():
    try:
        urllib.request.urlopen(f"{COMFY_SERVER}/history/1")
        return True
    except:
        return False

# --- Main Logic ---

def run_blender_process(input_mesh, output_path, voxel_size=0.05):
    print(f"[Auto-Rigger] Launching Blender processing...")
    
    # Check if Blender is in PATH, if not try common paths or ask user
    blender_cmd = BLENDER_EXEC
    
    # Simple check if "blender" works
    try:
        subprocess.run([blender_cmd, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Try finding it in common windows path
        common_paths = [
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.0\blender.exe"
        ]
        found = False
        for p in common_paths:
            if os.path.exists(p):
                blender_cmd = p
                found = True
                break
        
        if not found:
            print("[Error] Blender executable not found. Please ensure 'blender' is in your PATH or update the script.")
            return False

    cmd = [
        blender_cmd,
        "--background",
        "--python", BLENDER_SCRIPT,
        "--",
        input_mesh,
        output_path,
        str(voxel_size)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Error] Blender process failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="360Pack Studio Auto-Rigger")
    parser.add_argument("--input", "-i", required=True, help="Input file (Image for generation, or Mesh .glb/.obj for just rigging)")
    parser.add_argument("--output", "-o", default="output_rigged.glb", help="Output path for the final model")
    parser.add_argument("--quality", "-q", choices=["low", "mid", "high"], default="mid", help="Mesh quality (remesh size)")
    parser.add_argument("--skip-gen", action="store_true", help="Skip ComfyUI generation (treat input as mesh)")
    
    args = parser.parse_args()
    
    # Determine mode
    ext = os.path.splitext(args.input)[1].lower()
    is_mesh_already = ext in ['.glb', '.gltf', '.obj', '.fbx']
    
    mesh_to_process = None
    
    if is_mesh_already or args.skip_gen:
        print(f"[Auto-Rigger] Mode: Cleanup & Rigging Only")
        mesh_to_process = args.input
    else:
        print(f"[Auto-Rigger] Mode: Generation + Rigging")
        
        if not check_comfy_connection():
             print(f"[Error] Cannot connect to ComfyUI at {COMFY_SERVER}. Is it running?")
             sys.exit(1)
             
        print(f"[Auto-Rigger] Generating mesh from: {args.input}")
        
        # 1. Load Template
        if not os.path.exists(WORKFLOW_TEMPLATE):
            print(f"[Error] Workflow template not found at {WORKFLOW_TEMPLATE}")
            sys.exit(1)
            
        with open(WORKFLOW_TEMPLATE, 'r') as f:
            workflow = json.load(f)
            
        # 2. Upload Image (Placeholder logic - requires actual node support in JSON)
        # Assuming workflow has a specific "Load Image" node we want to replace.
        # For this v0.1, we assume the User has manually set the input path in ComfyUI or 
        # we rely on the specific "LoadImage" api format.
        
        # upload_res = upload_image(args.input)
        # workflow["3"]["inputs"]["image"] = upload_res["name"] 
        
        print("[Warn] Image upload logic requires specific node IDs matching your ComfyUI setup.")
        print("[Auto-Rigger] Sending generation prompt...")
        
        # 3. Queue Prompt
        try:
            res = queue_prompt(workflow)
            prompt_id = res['prompt_id']
            print(f"[Auto-Rigger] Generation queued. ID: {prompt_id}")
            
            # 4. Wait for result (Simplified: polling or wait for file)
            # In a real impl, we'd websocket listen. 
            # Here we just wait a bit or check output folder.
            print("...Waiting for generation (Simulating 10s)...")
            time.sleep(10) 
            
            # Assume output is some temp file for now
            # mesh_to_process = "path/to/generated/temp.glb"
            print("[Error] ComfyUI Generation integration requires specific 'Save Mesh' node setup.")
            print("[Tip] run with --skip-gen and provide a .glb file to test the Blender part.")
            sys.exit(1)
            
        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            sys.exit(1)

    # Blender Step
    quality_map = {
        "low": 0.1,
        "mid": 0.05,
        "high": 0.02
    }
    voxel_size = quality_map[args.quality]
    
    if mesh_to_process and os.path.exists(mesh_to_process):
        success = run_blender_process(mesh_to_process, args.output, voxel_size)
        if success:
             print(f"[Auto-Rigger] Success! Output saved to: {args.output}")
        else:
             print("[Auto-Rigger] Failed.")
    else:
        print(f"[Error] Input file not found: {mesh_to_process}")

if __name__ == "__main__":
    main()
