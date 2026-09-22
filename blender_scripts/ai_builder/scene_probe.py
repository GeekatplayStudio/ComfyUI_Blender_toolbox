# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""Describe the current Blender scene as JSON so the model knows what already exists."""

import os

import bpy
from mathutils import Vector


def _r(v, nd=2):
    return [round(float(x), nd) for x in v]


def probe_scene(max_objects=200):
    bpy.context.view_layer.update()
    scene = bpy.context.scene
    objects, lights, cameras = [], [], []
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    has_bounds = False
    for ob in scene.objects:
        entry = {
            "name": ob.name,
            "type": ob.type,
            "location": _r(ob.matrix_world.translation),
            "rotation_deg": _r([a * 57.2957795 for a in ob.rotation_euler], 1),
            "dimensions": _r(ob.dimensions),
            "collection": ob.users_collection[0].name if ob.users_collection else "",
            "parent": ob.parent.name if ob.parent else "",
        }
        if ob.type == "MESH":
            entry["vertices"] = len(ob.data.vertices)
            entry["faces"] = len(ob.data.polygons)
            entry["materials"] = [m.name for m in ob.data.materials if m]
            entry["modifiers"] = [m.type for m in ob.modifiers]
            if ob.hide_render:
                entry["hidden"] = True
                objects.append(entry)
                continue
            for corner in ob.bound_box:
                w = ob.matrix_world @ Vector(corner)
                lo = Vector((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
                hi = Vector((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
                has_bounds = True
        elif ob.type == "LIGHT":
            lights.append({"name": ob.name, "type": ob.data.type, "energy": round(float(ob.data.energy), 2),
                           "color": _r(ob.data.color, 3)})
        elif ob.type == "CAMERA":
            cameras.append(ob.name)
        objects.append(entry)
    mesh_objects = [o for o in objects if o["type"] == "MESH"]
    world = None
    if scene.world:
        world = {"name": scene.world.name}
        try:
            bg = next((n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"), None)
            if bg is not None:
                world["color"] = _r(bg.inputs["Color"].default_value, 3)
                world["strength"] = round(float(bg.inputs["Strength"].default_value), 2)
            env = next((n for n in scene.world.node_tree.nodes if n.type == "TEX_ENVIRONMENT"), None)
            if env is not None and env.image:
                world["hdri"] = env.image.name
        except Exception:
            pass
    images = []
    for img in bpy.data.images:
        if img.name in ("Render Result", "Viewer Node"):
            continue
        images.append({"name": img.name, "filepath": bpy.path.abspath(img.filepath) if img.filepath else "",
                       "packed": bool(img.packed_file), "size": list(img.size), "users": img.users})
    return {
        "blender_version": bpy.app.version_string,
        "object_count": len(objects),
        "mesh_object_count": len(mesh_objects),
        "total_vertices": sum(o.get("vertices", 0) for o in mesh_objects),
        "total_faces": sum(o.get("faces", 0) for o in mesh_objects),
        "objects": sorted(objects, key=lambda o: o["name"])[:max_objects],
        "collections": [c.name for c in bpy.data.collections],
        "materials": [m.name for m in bpy.data.materials if m.users > 0],
        "lights": lights,
        "cameras": cameras,
        "active_camera": scene.camera.name if scene.camera else "",
        "world": world,
        "images": images,
        "bbox": {"min": _r(lo), "max": _r(hi)} if has_bounds else None,
        "render": {"engine": scene.render.engine, "resolution": [scene.render.resolution_x, scene.render.resolution_y]},
        "blend_file": bpy.data.filepath,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(probe_scene(), indent=2))
