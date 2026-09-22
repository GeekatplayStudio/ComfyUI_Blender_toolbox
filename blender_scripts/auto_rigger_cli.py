import argparse
import os
import sys
import subprocess
import json
import time
import urllib.request
import urllib.parse
import re
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
    req = urllib.request.Request(f"{COMFY_SERVER}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI rejected the workflow ({e.code}):\n{body[:2000]}")

def upload_image(filepath):
    import requests
    with open(filepath, 'rb') as f:
        files = {'image': (os.path.basename(filepath), f, 'image/png')}
        response = requests.post(f"{COMFY_SERVER}/upload/image",
                                 files=files, data={"overwrite": "true"}, timeout=120)
    response.raise_for_status()
    return response.json()

def check_comfy_connection():
    try:
        urllib.request.urlopen(f"{COMFY_SERVER}/history/1")
        return True
    except:
        return False

def get_history(prompt_id):
    with urllib.request.urlopen(f"{COMFY_SERVER}/history/{prompt_id}") as r:
        return json.loads(r.read()).get(prompt_id)

def wait_for_prompt(prompt_id, timeout=1800, poll=2.0):
    """Block until ComfyUI finishes the prompt. Returns its history entry."""
    start = time.time()
    last_note = 0.0
    while time.time() - start < timeout:
        entry = get_history(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed") or status.get("status_str") == "success":
                return entry
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                raise RuntimeError(f"Generation failed in ComfyUI: {json.dumps(messages)[:1500]}")
        elapsed = time.time() - start
        if elapsed - last_note >= 15:
            last_note = elapsed
            print(f"[Auto-Rigger] ...generating ({int(elapsed)}s elapsed)")
        time.sleep(poll)
    raise TimeoutError(f"ComfyUI did not finish within {timeout}s (prompt {prompt_id})")

def download_output(entry, dest_dir):
    """Find the 3D file produced by the workflow and download it. Returns the local path."""
    candidates = []
    for node_output in entry.get("outputs", {}).values():
        for key in ("3d", "gltf", "mesh", "result", "files", "images"):
            for item in node_output.get(key, []) or []:
                if isinstance(item, dict) and item.get("filename"):
                    candidates.append(item)
    meshes = [c for c in candidates
              if os.path.splitext(c["filename"])[1].lower() in (".glb", ".gltf", ".obj", ".ply", ".stl")]
    if not meshes:
        raise RuntimeError("The workflow finished but produced no .glb/.gltf/.obj output. "
                           "Check that it ends in a SaveGLB (or equivalent) node.\n"
                           f"Outputs seen: {json.dumps(entry.get('outputs', {}))[:800]}")
    item = meshes[-1]
    params = urllib.parse.urlencode({"filename": item["filename"],
                                     "subfolder": item.get("subfolder", ""),
                                     "type": item.get("type", "output")})
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(item["filename"]))
    with urllib.request.urlopen(f"{COMFY_SERVER}/view?{params}") as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest

def patch_workflow(workflow, image_name):
    """Point every LoadImage node at the uploaded reference and randomise sampler seeds."""
    patched = False
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") == "LoadImage":
            node["inputs"]["image"] = image_name
            patched = True
        if "seed" in node.get("inputs", {}):
            node["inputs"]["seed"] = int(time.time() * 1000) % (2 ** 31)
    if not patched:
        raise RuntimeError("The workflow template has no LoadImage node to receive the reference image.")
    return workflow

# --- Main Logic ---

def run_blender_process(input_mesh, output_path, voxel_size=0.05):
    print(f"[Auto-Rigger] Launching Blender processing...")
    
    # Check if Blender is in PATH, if not try common paths or ask user
    blender_cmd = BLENDER_EXEC
    
    # Simple check if "blender" works
    try:
        subprocess.run([blender_cmd, "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Try finding it in Blender Foundation or common paths
        common_paths = [
            r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
            r"C:\Program Files\Blender Foundation\Blender 3.0\blender.exe"
        ]
        bf_dir = r"C:\Program Files\Blender Foundation"
        if os.path.exists(bf_dir):
            try:
                candidates = []
                for entry in os.listdir(bf_dir):
                    candidate_exe = os.path.join(bf_dir, entry, "blender.exe")
                    if os.path.isfile(candidate_exe):
                        match = re.search(r"(\d+(?:\.\d+)?)", entry)
                        ver = float(match.group(1)) if match else 0.0
                        candidates.append((ver, candidate_exe))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    common_paths = [c[1] for c in candidates] + common_paths
            except Exception:
                pass

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
    global COMFY_SERVER
    parser = argparse.ArgumentParser(description="360Pack Studio Auto-Rigger")
    parser.add_argument("--input", "-i", required=True, help="Input file (Image for generation, or Mesh .glb/.obj for just rigging)")
    parser.add_argument("--output", "-o", default="output_rigged.glb", help="Output path for the final model")
    parser.add_argument("--quality", "-q", choices=["low", "mid", "high"], default="mid", help="Mesh quality (remesh size)")
    parser.add_argument("--skip-gen", action="store_true", help="Skip ComfyUI generation (treat input as mesh)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Seconds to wait for ComfyUI generation (default 1800)")
    parser.add_argument("--server", default=COMFY_SERVER,
                        help=f"ComfyUI server URL (default {COMFY_SERVER})")

    args = parser.parse_args()
    COMFY_SERVER = args.server.rstrip("/")
    
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
            
        with open(WORKFLOW_TEMPLATE, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        try:
            # 2. Upload the reference image into ComfyUI's input folder
            upload = upload_image(args.input)
            image_name = upload["name"]
            if upload.get("subfolder"):
                image_name = f"{upload['subfolder']}/{image_name}"
            print(f"[Auto-Rigger] Uploaded reference as '{image_name}'")

            # 3. Point the workflow at it and queue
            workflow = patch_workflow(workflow, image_name)
            res = queue_prompt(workflow)
            prompt_id = res['prompt_id']
            print(f"[Auto-Rigger] Generation queued. ID: {prompt_id}")

            # 4. Wait for ComfyUI to finish, then fetch the generated mesh
            entry = wait_for_prompt(prompt_id, timeout=args.timeout)
            download_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".", "autorig_generated")
            mesh_to_process = download_output(entry, download_dir)
            print(f"[Auto-Rigger] Generated mesh: {mesh_to_process}")

        except Exception as e:
            print(f"[Error] Generation failed: {e}")
            print("[Tip] Run with --skip-gen and pass a .glb/.obj to use only the Blender clean+rig step.")
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
