# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
Scene validation that runs inside Blender.

Checks (per mesh object): vertex/face/triangle counts, non-manifold edges, boundary edges,
loose vertices, loose edges, zero-area faces, duplicate vertices, flipped/inconsistent normals,
n-gons, UV presence, material presence, default "Cube.001"-style names.
Checks (scene): missing or broken image textures, unused images, materials without a Principled
node, no camera, no light.

Severity:
  error   -> the step FAILS and the model gets a retry (non-manifold, loose geometry, zero-area
             faces, flipped normals, missing textures)
  warning -> reported, does not fail the step (open/boundary edges, n-gons, naming, no UV,
             no material, no camera/light)

Optional auto-fixes (explicit flags): recalc normals outward, merge duplicate vertices,
delete loose vertices/edges, dissolve degenerate faces.

Run standalone:  blender -b file.blend --python validate_scene.py -- out.json [--fix-normals] [--fix-doubles]
"""

import json
import os
import re
import sys

import bmesh
import bpy
from mathutils import Vector

DEFAULT_NAME_RE = re.compile(
    r"^(Cube|Sphere|Cylinder|Plane|Cone|Torus|Mesh|Object|Material|Light|Point|Sun|Spot|Area|Camera|Empty|"
    r"Suzanne|Grid|Circle|Icosphere|Monkey|Text|Curve|BezierCurve|Collection)(\.\d{3})?$"
)
SKIP_IMAGES = ("Render Result", "Viewer Node")


def _issue(issues, severity, category, message, obj=None):
    issues.append({"severity": severity, "category": category, "object": obj or "", "message": message})


def validate_mesh_object(ob, issues, auto_fix_normals=False, auto_fix_doubles=False, doubles_dist=1e-5,
                         auto_fixes=None):
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    report = {"name": ob.name, "vertices": len(bm.verts), "faces": len(bm.faces),
              "triangles": sum(len(f.verts) - 2 for f in bm.faces)}

    fixed = []
    if auto_fix_doubles:
        def _nonmanifold(mesh):
            return sum(1 for e in mesh.edges if not e.is_manifold and not e.is_boundary)

        # Welding by distance fuses vertices where two separate closed shells touch (a rivet sitting
        # on a hull, a trim ring on a cone). That turns valid intersecting geometry into non-manifold
        # edges, so the weld is only kept when it does not make the topology worse.
        before_nm = _nonmanifold(bm)
        trial = bm.copy()
        res = bmesh.ops.find_doubles(trial, verts=trial.verts, dist=doubles_dist)
        n = len(res.get("targetmap", {}))
        if n:
            bmesh.ops.weld_verts(trial, targetmap=res["targetmap"])
            if _nonmanifold(trial) <= before_nm:
                bm.free()
                bm = trial
                bm.verts.ensure_lookup_table()
                bm.faces.ensure_lookup_table()
                fixed.append(f"{ob.name}: merged {n} duplicate vertices")
            else:
                trial.free()  # welding would fuse touching shells - leave the mesh as modelled
        else:
            trial.free()
        bmesh.ops.dissolve_degenerate(bm, dist=1e-7, edges=bm.edges)
        loose_e = [e for e in bm.edges if not e.link_faces]
        if loose_e:
            bmesh.ops.delete(bm, geom=loose_e, context="EDGES")
            fixed.append(f"{ob.name}: deleted {len(loose_e)} loose edges")
        loose_v = [v for v in bm.verts if not v.link_edges]
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
            fixed.append(f"{ob.name}: deleted {len(loose_v)} loose vertices")
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

    boundary = sum(1 for e in bm.edges if e.is_boundary)
    nonmanifold = sum(1 for e in bm.edges if not e.is_manifold and not e.is_boundary)
    loose_verts = sum(1 for v in bm.verts if not v.link_edges)
    loose_edges = sum(1 for e in bm.edges if not e.link_faces)
    zero_area = sum(1 for f in bm.faces if f.calc_area() < 1e-9)
    ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
    doubles = len(bmesh.ops.find_doubles(bm, verts=bm.verts, dist=doubles_dist).get("targetmap", {})) if not auto_fix_doubles else 0

    # Flipped / inconsistent normals: compare with a recalculated copy. Only meaningful for closed meshes.
    flipped = 0
    closed = boundary == 0 and nonmanifold == 0 and len(bm.faces) > 0
    if closed:
        bm2 = bm.copy()
        bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces)
        bm2.faces.ensure_lookup_table()
        for f, g in zip(bm.faces, bm2.faces):
            if f.normal.dot(g.normal) < 0:
                flipped += 1
        if flipped and auto_fix_normals:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            fixed.append(f"{ob.name}: recalculated {flipped} face normals outward")
            flipped = 0
        bm2.free()

    if fixed:
        bm.to_mesh(me)
        me.update()
        if auto_fixes is not None:
            auto_fixes.extend(fixed)
    bm.free()

    has_uv = bool(me.uv_layers)
    mats = [m for m in me.materials if m]
    report.update({
        "boundary_edges": boundary, "nonmanifold_edges": nonmanifold, "loose_vertices": loose_verts,
        "loose_edges": loose_edges, "zero_area_faces": zero_area, "duplicate_vertices": doubles,
        "flipped_normals": flipped, "ngons": ngons, "closed": closed, "has_uv": has_uv,
        "materials": [m.name for m in mats], "empty_material_slots": sum(1 for m in me.materials if m is None),
        "scale": [round(s, 3) for s in ob.scale], "default_name": bool(DEFAULT_NAME_RE.match(ob.name) or DEFAULT_NAME_RE.match(me.name)),
    })

    n = ob.name
    if nonmanifold:
        _issue(issues, "error", "topology", f"{nonmanifold} non-manifold edges (edges shared by 3+ faces or bad geometry)", n)
    if loose_verts:
        _issue(issues, "error", "topology", f"{loose_verts} loose vertices", n)
    if loose_edges:
        _issue(issues, "error", "topology", f"{loose_edges} loose edges (edges without faces)", n)
    if zero_area:
        _issue(issues, "error", "topology", f"{zero_area} zero-area (degenerate) faces", n)
    if flipped:
        _issue(issues, "error", "normals", f"{flipped} faces with flipped/inconsistent normals (recalculate outward)", n)
    if doubles:
        _issue(issues, "warning", "topology", f"{doubles} duplicate vertices within {doubles_dist} (merge by distance)", n)
    if boundary and len(me.polygons) > 0:
        _issue(issues, "warning", "topology", f"{boundary} boundary edges - mesh is open (fine for planes/terrain, not for solids)", n)
    if ngons:
        _issue(issues, "warning", "topology", f"{ngons} n-gons (faces with more than 4 vertices)", n)
    if not mats and len(me.polygons) > 0:
        _issue(issues, "warning", "materials", "no material assigned", n)
    if report["empty_material_slots"]:
        _issue(issues, "warning", "materials", f"{report['empty_material_slots']} empty material slots", n)
    if not has_uv and mats and any(_material_uses_image(m) for m in mats):
        _issue(issues, "warning", "uv", "uses image textures but has no UV map", n)
    if report["default_name"]:
        _issue(issues, "warning", "naming", f"default name ('{ob.name}' / mesh '{me.name}') - give it a descriptive name", n)
    return report


def _material_uses_image(mat):
    try:
        return any(n.type == "TEX_IMAGE" for n in mat.node_tree.nodes) if mat.node_tree else False
    except Exception:
        return False


def validate_textures(issues):
    reports = []
    for img in bpy.data.images:
        if img.name in SKIP_IMAGES:
            continue
        path = bpy.path.abspath(img.filepath) if img.filepath else ""
        entry = {"name": img.name, "filepath": path, "packed": bool(img.packed_file), "size": list(img.size),
                 "users": img.users, "source": img.source}
        if img.source == "FILE" and not img.packed_file:
            if not path or not os.path.exists(path):
                _issue(issues, "error", "textures", f"image '{img.name}' points to a missing file: {path or '(empty path)'}")
                entry["missing"] = True
            elif img.size[0] == 0 or img.size[1] == 0:
                _issue(issues, "error", "textures", f"image '{img.name}' could not be loaded (size 0): {path}")
                entry["broken"] = True
        if img.users == 0:
            _issue(issues, "warning", "textures", f"image '{img.name}' is not used by any material")
        reports.append(entry)
    return reports


def validate_materials(issues):
    for mat in bpy.data.materials:
        if mat.users == 0:
            _issue(issues, "warning", "materials", f"material '{mat.name}' is unused")
            continue
        if DEFAULT_NAME_RE.match(mat.name):
            _issue(issues, "warning", "naming", f"material '{mat.name}' has a default name")
        tree = mat.node_tree
        if tree is not None:
            has_out = any(n.type == "OUTPUT_MATERIAL" and n.inputs["Surface"].is_linked for n in tree.nodes)
            if not has_out:
                _issue(issues, "warning", "materials", f"material '{mat.name}' has no connected shader output (renders black/pink)")


def validate_scene_level(issues):
    scene = bpy.context.scene
    if not scene.camera:
        _issue(issues, "warning", "scene", "no active camera (a preview camera will be created for renders)")
    if not any(o.type == "LIGHT" for o in scene.objects):
        strength = 0.0
        try:
            bg = next((n for n in scene.world.node_tree.nodes if n.type == "BACKGROUND"), None)
            strength = float(bg.inputs["Strength"].default_value) if bg else 0.0
        except Exception:
            pass
        if strength <= 0.0:
            _issue(issues, "warning", "scene", "no lights and no world lighting - renders will be black")
    for ob in scene.objects:
        if ob.type in ("LIGHT", "CAMERA") and DEFAULT_NAME_RE.match(ob.name):
            _issue(issues, "warning", "naming", f"{ob.type.lower()} '{ob.name}' has a default name", ob.name)
    for coll in bpy.data.collections:
        if DEFAULT_NAME_RE.match(coll.name):
            _issue(issues, "warning", "naming", f"collection '{coll.name}' has a default name")


def validate_scene(auto_fix_normals=False, auto_fix_doubles=False, exclude_names=(), doubles_dist=1e-5):
    bpy.context.view_layer.update()
    issues, objects, auto_fixes = [], [], []
    seen_meshes = {}  # linked duplicates (scatter) share mesh data - validate the geometry once
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH" or ob.name in exclude_names:
            continue
        try:
            if ob.data.name in seen_meshes:
                base = dict(seen_meshes[ob.data.name])
                base["name"] = ob.name
                base["shared_mesh_of"] = seen_meshes[ob.data.name]["name"]
                base["default_name"] = bool(DEFAULT_NAME_RE.match(ob.name))
                if base["default_name"]:
                    _issue(issues, "warning", "naming", f"default name ('{ob.name}') - give it a descriptive name", ob.name)
                objects.append(base)
                continue
            report = validate_mesh_object(ob, issues, auto_fix_normals, auto_fix_doubles, doubles_dist, auto_fixes)
            seen_meshes[ob.data.name] = report
            objects.append(report)
        except Exception as e:
            _issue(issues, "error", "validator", f"validation crashed on '{ob.name}': {e}", ob.name)
    textures = validate_textures(issues)
    validate_materials(issues)
    validate_scene_level(issues)

    def total(key):
        return sum(o.get(key, 0) for o in objects)

    totals = {
        "mesh_objects": len(objects),
        "vertices": total("vertices"), "faces": total("faces"), "triangles": total("triangles"),
        "nonmanifold_edges": total("nonmanifold_edges"), "boundary_edges": total("boundary_edges"),
        "loose_vertices": total("loose_vertices"), "loose_edges": total("loose_edges"),
        "zero_area_faces": total("zero_area_faces"), "duplicate_vertices": total("duplicate_vertices"),
        "flipped_normals": total("flipped_normals"), "ngons": total("ngons"),
        "closed_meshes": sum(1 for o in objects if o.get("closed")),
        "objects_without_material": sum(1 for o in objects if not o.get("materials")),
        "objects_without_uv": sum(1 for o in objects if not o.get("has_uv")),
        "missing_textures": sum(1 for t in textures if t.get("missing") or t.get("broken")),
        "naming_issues": sum(1 for i in issues if i["category"] == "naming"),
        "errors": sum(1 for i in issues if i["severity"] == "error"),
        "warnings": sum(1 for i in issues if i["severity"] == "warning"),
    }
    return {
        "blender_version": bpy.app.version_string,
        "passed": totals["errors"] == 0,
        "totals": totals,
        "objects": objects,
        "textures": textures,
        "issues": issues,
        "auto_fixes": auto_fixes,
    }


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = argv[0] if argv else "validation.json"
    result = validate_scene(auto_fix_normals="--fix-normals" in argv, auto_fix_doubles="--fix-doubles" in argv)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("VALIDATION", json.dumps(result["totals"]), flush=True)
