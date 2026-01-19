# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox

import numpy as np
import copy
try:
    import trimesh
except ImportError:
    trimesh = None

from .utils_blender import run_blender_mesh_operation

class GapFillHoles:
    """
    Fills holes in a mesh using Trimesh's native hole filling algorithms.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info_string")
    FUNCTION = "fill_holes"
    CATEGORY = "Geekatplay Studio/Geometry"

    def fill_holes(self, mesh):
        if trimesh is None:
            return (mesh, "Error: Trimesh not installed.")

        work_mesh = None
        if isinstance(mesh, trimesh.Trimesh):
            work_mesh = mesh.copy()
        elif isinstance(mesh, list) and len(mesh) > 0:
             if isinstance(mesh[0], trimesh.Trimesh):
                 work_mesh = mesh[0].copy()
             else:
                 try:
                     work_mesh = trimesh.Trimesh(vertices=mesh[0][0], faces=mesh[0][1], process=False)
                 except: pass

        if work_mesh is None:
             return (mesh, "Invalid Mesh Input")

        initial_faces = len(work_mesh.faces)
        initial_watertight = work_mesh.is_watertight
        
        try:
            work_mesh.fill_holes()
        except Exception as e:
            return (mesh, f"Fill Holes Failed: {e}")
            
        final_faces = len(work_mesh.faces)
        final_watertight = work_mesh.is_watertight
        
        info = f"""Topological Analysis:
- Initial Faces: {initial_faces}
- Initial Watertight: {initial_watertight}
----------------
- Final Faces: {final_faces} (+{final_faces - initial_faces})
- Final Watertight: {final_watertight}
"""
        return (work_mesh, info)

class GapBlenderVoxelRemesh:
    """
    Uses a background Blender process to perform a Voxel Remesh.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "voxel_size": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 1.0, "step": 0.005}),
                "adaptivity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Currently unused by standard Voxel Remesh but kept for future QuadriFlow"}),
            }
        }
        
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "Geekatplay Studio/Geometry"
    
    def remesh(self, mesh, voxel_size, adaptivity):
        script = """
import bpy
import os
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)

input_path = r'{input_path}'
output_path = r'{output_path}'

try:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.obj':
        bpy.ops.import_scene.obj(filepath=input_path)
    elif ext == '.glb' or ext == '.gltf':
        bpy.ops.import_scene.gltf(filepath=input_path)
    
    obj = None
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            obj = o
            break
            
    if not obj:
        print("Error: No Mesh object found in selection.")
        sys.exit(1)

    bpy.context.view_layer.objects.active = obj

    mod = obj.modifiers.new(name="Remesh", type='REMESH')
    mod.mode = 'VOXEL'
    mod.voxel_size = {voxel_size}
    mod.adaptivity = {adaptivity}
    
    bpy.ops.object.modifier_apply(modifier=mod.name)
    
    out_ext = os.path.splitext(output_path)[1].lower()
    if out_ext == '.glb':
        bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=True)
    elif out_ext == '.obj':
         bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)
except Exception as e:
    print(f"Blender Error: {e}")
    sys.exit(1)
"""
        script = script.replace("{voxel_size}", str(voxel_size))
        script = script.replace("{adaptivity}", str(adaptivity))
        
        try:
            result_mesh = run_blender_mesh_operation(mesh, script, output_format='glb')
            info = f"Remesh Complete.\nVoxel Size: {voxel_size}\nVertices: {len(result_mesh.vertices)}\nFaces: {len(result_mesh.faces)}"
            return (result_mesh, info)
        except Exception as e:
            return (mesh, f"Error: {e}")

class GapFaceNormalsFix:
    """
    Recalculates or flips normals.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "method": (["fix_winding", "flip"],),
            }
        }
    RETURN_TYPES = ("MESH",)
    FUNCTION = "process"
    CATEGORY = "Geekatplay Studio/Geometry"
    
    def process(self, mesh, method):
        work_mesh = None
        if isinstance(mesh, trimesh.Trimesh):
             work_mesh = mesh.copy()
        elif isinstance(mesh, list) and len(mesh) > 0 and hasattr(mesh[0], 'vertices'):
             work_mesh = mesh[0].copy()
             
        if not work_mesh: return (mesh,)
        
        if method == "fix_winding":
            work_mesh.fix_normals()
        elif method == "flip":
            work_mesh.invert()
            
        return (work_mesh,)

class GapBlenderDecimate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.05}),
            }
        }
    RETURN_TYPES = ("MESH",)
    FUNCTION = "decimate"
    CATEGORY = "Geekatplay Studio/Geometry"
    
    def decimate(self, mesh, ratio):
        script = """
import bpy
import os
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)
input_path = r'{input_path}'
output_path = r'{output_path}'

try:
    bpy.ops.import_scene.gltf(filepath=input_path)
    
    obj = None
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            obj = o
            break
            
    if not obj:
        print("Error: No Mesh object found in selection")
        sys.exit(1)

    bpy.context.view_layer.objects.active = obj
    
    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    mod.ratio = {ratio}
    bpy.ops.object.modifier_apply(modifier=mod.name)
    
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=True)
except Exception as e:
    print(f"Decimate Error: {e}")
    sys.exit(1)
"""
        script = script.replace("{ratio}", str(ratio))
        try:
            return (run_blender_mesh_operation(mesh, script),)
        except Exception as e:
            print(f"Decimate failed: {e}")
            return (mesh,)

class GapBlenderSubdivide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "levels": ("INT", {"default": 1, "min": 1, "max": 4}),
                "type": (["CATMULL_CLARK", "SIMPLE"],),
            }
        }
    RETURN_TYPES = ("MESH",)
    FUNCTION = "subdivide"
    CATEGORY = "Geekatplay Studio/Geometry"
    
    def subdivide(self, mesh, levels, type):
        script = """
import bpy
import os
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)
input_path = r'{input_path}'
output_path = r'{output_path}'

try:
    bpy.ops.import_scene.gltf(filepath=input_path)
    
    obj = None
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            obj = o
            break
            
    if not obj:
        print("Error: No Mesh object found")
        sys.exit(1)

    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new(name="Subsurf", type='SUBSURF')
    mod.levels = {levels}
    mod.render_levels = {levels}
    mod.subdivision_type = '{type}'
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=True)
except Exception as e:
    print(f"Subdivide Error: {e}")
    sys.exit(1)
"""
        script = script.replace("{levels}", str(levels))
        script = script.replace("{type}", type)
        try:
            return (run_blender_mesh_operation(mesh, script),)
        except Exception as e:
            print(f"Subdivide failed: {e}")
            return (mesh,)

class GapBlenderBoolean:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh_a": ("MESH",),
                "mesh_b": ("MESH",),
                "operation": (["UNION", "DIFFERENCE", "INTERSECT"],),
                "solver": (["FAST", "EXACT"], {"default": "EXACT"}),
            }
        }
    RETURN_TYPES = ("MESH",)
    FUNCTION = "boolean"
    CATEGORY = "Geekatplay Studio/Geometry"
    
    def boolean(self, mesh_a, mesh_b, operation, solver):
        script = """
import bpy
import os
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)
input_path = r'{input_path}'
input_path_b = r'{input_path_b}'
output_path = r'{output_path}'

try:
    # Import Mesh A
    bpy.ops.import_scene.gltf(filepath=input_path)
    
    obj_a = None
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            obj_a = o
            break
    if not obj_a:
        print("Error: Mesh A object not found")
        sys.exit(1)
        
    obj_a.name = "MeshA"
    bpy.ops.object.select_all(action='DESELECT')

    # Import Mesh B
    bpy.ops.import_scene.gltf(filepath=input_path_b)
    
    obj_b = None
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            obj_b = o
            break
    if not obj_b:
        print("Error: Mesh B object not found")
        sys.exit(1)
        
    obj_b.name = "MeshB"

    # Active object must be Mesh A for modifier
    bpy.context.view_layer.objects.active = obj_a
    obj_a.select_set(True)

    mod = obj_a.modifiers.new(name="Boolean", type='BOOLEAN')
    mod.object = obj_b
    mod.operation = '{operation}'
    
    # Solver compatibility check (FAST renamed to FLOAT in 5.0)
    solver_arg = '{solver}'
    try:
        mod.solver = solver_arg
    except TypeError:
        if solver_arg == 'FAST':
            mod.solver = 'FLOAT'
        else:
            raise

    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Select only Mesh A for export
    bpy.ops.object.select_all(action='DESELECT')
    obj_a.select_set(True)

    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=True)
except Exception as e:
    print(f"Boolean Error: {e}")
    sys.exit(1)
"""
        script = script.replace("{operation}", operation)
        script = script.replace("{solver}", solver)
        try:
            return (run_blender_mesh_operation(mesh_a, script, input_mesh_b=mesh_b),)
        except Exception as e:
            print(f"Boolean failed: {e}")
            return (mesh_a,)

class GapBlenderSmartUV:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "angle_limit": ("FLOAT", {"default": 66.0, "min": 1.0, "max": 89.0}),
                "island_margin": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step":0.001}),
            }
        }
    RETURN_TYPES = ("MESH",)
    FUNCTION = "unwrap"
    CATEGORY = "Geekatplay Studio/Geometry"
    
    def unwrap(self, mesh, angle_limit, island_margin):
        script = """
import bpy
import os
import math
import sys

bpy.ops.wm.read_factory_settings(use_empty=True)
input_path = r'{input_path}'
output_path = r'{output_path}'

try:
    bpy.ops.import_scene.gltf(filepath=input_path)
    
    obj = None
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            obj = o
            break
    if not obj:
        print("Error: No Mesh object found")
        sys.exit(1)
        
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians({angle_limit}), island_margin={island_margin})
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB', use_selection=True)
except Exception as e:
    print(f"UV Unwrap Error: {e}")
    sys.exit(1)
"""
        script = script.replace("{angle_limit}", str(angle_limit))
        script = script.replace("{island_margin}", str(island_margin))
        try:
            return (run_blender_mesh_operation(mesh, script),)
        except Exception as e:
            print(f"UV Unwrap failed: {e}")
            return (mesh,)
