import bpy
import tempfile
import os
import sys

print("Starting Round Trip Test")
temp_dir = tempfile.gettempdir()
out_file = os.path.join(temp_dir, "test_out.glb")

# Clean
bpy.ops.wm.read_factory_settings(use_empty=True)

# Create a cube
bpy.ops.mesh.primitive_cube_add()
print("Cube added")

# Select it
obj = bpy.context.active_object
obj.select_set(True)

try:
    bpy.ops.export_scene.gltf(filepath=out_file, export_format='GLB', use_selection=True)
    print(f"Export Success to {out_file}")
    if os.path.exists(out_file):
        print(f"File exists: True, Size: {os.path.getsize(out_file)}")
    else:
        print("File exists: False")
except Exception as e:
    print(f"Export Failed: {e}")
    sys.exit(1)
