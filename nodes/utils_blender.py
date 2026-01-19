# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox

import os
import subprocess
import tempfile
import sys
import shutil
import re

try:
    import trimesh
except ImportError:
    trimesh = None

def get_blender_path():
    """
    Attempts to find Blender executable.
    Priorities:
    1. Environment Variable BLENDER_PATH
    2. Common installation paths
    3. PATH
    """
    # 1. Check Env Var
    env_path = os.environ.get("BLENDER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
        
    # 2. Check Common Paths (Windows)
    common_paths = [
        r"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.3\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.1\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
            
    # 3. Check PATH
    path_executable = shutil.which("blender")
    if path_executable:
        return path_executable
        
    return None

def run_blender_script(script, timeout=300):
    """
    Runs a python script in Blender background mode.
    script: The python code as string
    """
    blender_path = get_blender_path()
    if not blender_path:
        raise RuntimeError("Blender executable not found. Please install Blender or set BLENDER_PATH.")

    # Create a temp python script file
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w', encoding='utf-8') as f:
        f.write(script)
        script_path = f.name
    
    try:
        # Run Blender
        # --background: no UI
        # --python: run the script file
        cmd = [blender_path, "--background", "--python", script_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Blender Execution Failed:\n{result.stderr}\n{result.stdout}")
            
        return result
    finally:
        pass
        # try:
        #     os.unlink(script_path)
        # except: pass

def run_blender_mesh_operation(input_mesh, blender_script_template, output_format='glb', timeout=300, input_mesh_b=None):
    """
    Export mesh -> Run Blender Script -> Import result.
    input_mesh: trimesh object
    input_mesh_b: Optional second trimesh object (e.g. for boolean ops)
    blender_script_template: String with {input_path}, {output_path}, and optionally {input_path_b}
    """
    if trimesh is None:
        raise ImportError("Trimesh is required for mesh operations.")

    # 1. Export Input Mesh to Temp
    with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f_in:
        f_in.close()
        input_path = f_in.name
    
    input_path_b = ""
    if input_mesh_b is not None:
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as f_in_b:
            f_in_b.close()
            input_path_b = f_in_b.name
    
    with tempfile.NamedTemporaryFile(suffix=f'.{output_format}', delete=False) as f_out:
        f_out.close()
        output_path = f_out.name
        
    try:
        # Handle Batch or Single for Mesh A
        mesh_to_export = input_mesh
        if isinstance(input_mesh, list):
            mesh_to_export = input_mesh[0] # Single item batch
        
        # GLB is safest for Blender import
        mesh_to_export.export(input_path)
        
        # Handle Mesh B
        if input_mesh_b is not None:
             mesh_b_to_export = input_mesh_b
             if isinstance(input_mesh_b, list):
                 mesh_b_to_export = input_mesh_b[0]
             mesh_b_to_export.export(input_path_b)
        
        # 2. Prepare Script
        # Escape backslashes for Windows paths in Python string
        safe_in = input_path.replace('\\', '/')
        safe_out = output_path.replace('\\', '/')
        safe_in_b = input_path_b.replace('\\', '/') if input_path_b else ""
        
        formatted_script = blender_script_template.replace("{input_path}", safe_in)
        formatted_script = formatted_script.replace("{output_path}", safe_out)
        if input_path_b:
             formatted_script = formatted_script.replace("{input_path_b}", safe_in_b)

        
        # Add basic imports if missing (helper)
        if "import bpy" not in formatted_script:
             formatted_script = "import bpy\nimport os\n" + formatted_script

        # 3. Execution
        run_blender_script(formatted_script, timeout=timeout)
        
        # 4. Import Result
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Blender did not generate output at {output_path}")
            
        result_mesh = trimesh.load(output_path, process=False)
        
        # If Scene (GLB often imports as Scene), dump
        if isinstance(result_mesh, trimesh.Scene):
             result_mesh = result_mesh.dump(concatenate=True)

        return result_mesh

    finally:
        # Cleanup
        try:
            if os.path.exists(input_path): os.unlink(input_path)
            if input_path_b and os.path.exists(input_path_b): os.unlink(input_path_b)
            if os.path.exists(output_path): os.unlink(output_path)
        except: pass
