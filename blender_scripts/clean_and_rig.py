import bpy
import sys
import os

bl_info = {
    "name": "ComfyUI 360 Clean & Rig Script",
    "author": "Geekatplay Studio",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "CLI",
    "description": "CLI script for mesh cleaning and rigging",
    "category": "Development",
}

# Argument parsing hack for Blender background mode
# Blender arguments are after "--", so we look for that
try:
    separator_idx = sys.argv.index("--")
    args = sys.argv[separator_idx + 1:]
except ValueError:
    args = []

if len(args) < 2:
    print("Usage: blender --background --python clean_and_rig.py -- <input_mesh_path> <output_gltf_path> [voxel_size]")
    sys.exit(1)

input_path = args[0]
output_path = args[1]
voxel_size = float(args[2]) if len(args) > 2 else 0.05
decimate_ratio = float(args[3]) if len(args) > 3 else 0.1
# Simple toggle for rigging
do_rig = True 
if len(args) > 4 and args[4].lower() == "false":
    do_rig = False

print(f"[Auto-Rigger] Starting cleanup pipeline...")
print(f"[Auto-Rigger] Input: {input_path}")
print(f"[Auto-Rigger] Output: {output_path}")
print(f"[Auto-Rigger] Voxel Size: {voxel_size}")
print(f"[Auto-Rigger] Decimate Ratio: {decimate_ratio}")
print(f"[Auto-Rigger] Apply Basic Rig: {do_rig}")

# 1. Clear Scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# 2. Import Mesh
# Determine format by extension
ext = os.path.splitext(input_path)[1].lower()
try:
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=input_path)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=input_path)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=input_path)
    else:
        print(f"[Auto-Rigger] Error: Unsupported file extension {ext}")
        sys.exit(1)
except Exception as e:
    print(f"[Auto-Rigger] Failed to import mesh: {e}")
    sys.exit(1)

# Select the main mesh object
# We assume the imported object is the first selected or active one
obj = None
for o in bpy.context.selected_objects:
    if o.type == 'MESH':
        obj = o
        break

if not obj:
    print("[Auto-Rigger] No mesh object found after import.")
    sys.exit(1)

bpy.context.view_layer.objects.active = obj
print(f"[Auto-Rigger] Processing object: {obj.name}")

# 3. Clean Geometry
# Voxel Remesh
print(f"[Auto-Rigger] Applying Voxel Remesh (size={voxel_size})...")
remesh = obj.modifiers.new(name="Remesh", type='REMESH')
remesh.mode = 'VOXEL'
remesh.voxel_size = voxel_size
bpy.ops.object.modifier_apply(modifier="Remesh")

# Decimate
print(f"[Auto-Rigger] Applying Decimate (ratio={decimate_ratio})...")
decimate = obj.modifiers.new(name="Decimate", type='DECIMATE')
decimate.ratio = decimate_ratio
bpy.ops.object.modifier_apply(modifier="Decimate")

# Smooth Shade
bpy.ops.object.shade_smooth()

# 4. Rigging
print("[Auto-Rigger] Centering and Aligning...")
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
obj.location = (0, 0, 0)
# Ensure base is at 0 (approximate)
min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
obj.location.z -= min_z 

if do_rig:
    print("[Auto-Rigger] Generating Basic Armature...")
    # Add simple armature
    bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
    armature = bpy.context.active_object
    armature.name = "AutoRig_Armature"
    
    # Scale armature roughly to mesh height
    max_z = max((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    # Default single bone is 1 unit high usually.
    # We want a very simple rig: Spine.
    
    # Let's try to add a Human Metarig if available (Rigify)
    try:
        # Enable Rigify if not verified
        import addon_utils
        addon_utils.enable("rigify")
        
        # Remove the simple bone we just made
        bpy.data.objects.remove(armature, do_unlink=True)
        
        # Add basic human (Basic Human is simpler than full Human)
        bpy.ops.object.armature_basic_human_metarig_add()
        armature = bpy.context.active_object
        armature.name = "AutoRig_Skeleton"
        
        # Scale logic: A standard human is ~1.7m. 
        # Calculate mesh height
        mesh_height = max_z - min_z
        scale_factor = mesh_height / 1.75 # approx
        armature.scale = (scale_factor, scale_factor, scale_factor)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        print(f"[Auto-Rigger] Added Rigify Basic Human (Scaled x{scale_factor:.2f})")
    except Exception as e:
        print(f"[Auto-Rigger] Rigify not available ({e}), keeping simple bone...")
        # Fallback to the single bone we created earlier (if we hadn't deleted it)
        # Re-create simple bone
        bpy.ops.object.armature_add(enter_editmode=False, location=(0, 0, 0))
        armature = bpy.context.active_object
        armature.scale = (1, 1, mesh_height)
        bpy.ops.object.transform_apply(scale=True)

    # Parenting
    print("[Auto-Rigger] Binding with Automatic Weights...")
    # Select Mesh then Armature
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    
    # Parent
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

# 5. Export
print(f"[Auto-Rigger] Exporting to {output_path}...")
bpy.ops.export_scene.gltf(
    filepath=output_path, 
    export_format='GLB',
    use_selection=True
)

print("[Auto-Rigger] Done.")
