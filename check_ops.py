import bpy
print("CHECK_OP: import_scene.gltf exists:", hasattr(bpy.ops.import_scene, 'gltf'))
try:
    print("CHECK_OP: wm.glb_import exists:", hasattr(bpy.ops.wm, 'glb_import'))
except:
    print("CHECK_OP: wm.glb_import exists: False")
