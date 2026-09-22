# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
gap_helpers - tested, version-safe building blocks for generated scripts.

Generated scripts start with `from gap_helpers import *`. Everything here works in Blender
background mode (no viewport/context tricks) on Blender 3.6 through 5.x.

Geometry is produced as explicit vertex/face lists (Builder) so meshes are closed and normals
are recalculated outward - that is what makes the validator pass on the first try.
"""

import math
import random
from math import cos, pi, sin

import bmesh
import bpy
from mathutils import Vector, noise

__all__ = [
    "blender_version", "get_or_create_collection", "link_object", "Builder", "mesh_from_pydata",
    "make_material", "assign_material", "set_smooth", "add_light", "add_sun", "add_camera",
    "frame_camera_to_scene", "set_world", "terrain", "scatter", "array_copies", "duplicate",
    "delete_objects", "clear_scene", "log_built", "scene_bounds", "Vector", "math", "random",
    "bpy", "bmesh", "mathutils", "noise",
    # detail helpers
    "ring_positions", "rivet_ring", "bolt", "trim_ring", "panel_seams", "porthole",
    "star_shape", "crescent_shape",
]

import mathutils  # noqa: E402  (re-exported so `mathutils.Vector` works after `from gap_helpers import *`)


def _resolve_material(mat):
    """Accept a Material, a material name, or None. Names that do not exist yet get a plain material."""
    if mat is None or isinstance(mat, bpy.types.Material):
        return mat
    if isinstance(mat, str):
        found = bpy.data.materials.get(mat)
        return found if found is not None else make_material(mat, (0.7, 0.7, 0.7))
    if isinstance(mat, (list, tuple)):
        return [_resolve_material(m) for m in mat]
    return mat


# ----------------------------------------------------------------------------- basics
def blender_version():
    return bpy.app.version


def get_or_create_collection(name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if coll.name not in parent.children and coll.name not in [c.name for c in _all_children(parent)]:
        parent.children.link(coll)
    return coll


def _all_children(coll):
    out = []
    for c in coll.children:
        out.append(c)
        out.extend(_all_children(c))
    return out


def link_object(obj, collection=None):
    collection = collection or bpy.context.scene.collection
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def log_built(obj):
    try:
        n = len(obj.data.vertices) if obj.type == "MESH" else 0
    except Exception:
        n = 0
    print(f"BUILT {obj.name} {n}", flush=True)


def _finalize_mesh(me, smooth_faces=None, angle_deg=35.0, merge_dist=1e-5):
    bm = bmesh.new()
    bm.from_mesh(me)
    if merge_dist > 0:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    bmesh.ops.dissolve_degenerate(bm, dist=1e-7, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    thr = math.radians(angle_deg)
    for e in bm.edges:
        if e.is_manifold:
            e.smooth = e.calc_face_angle(0.0) < thr
    bm.to_mesh(me)
    bm.free()
    me.update()
    if smooth_faces is not None:
        for p, s in zip(me.polygons, smooth_faces):
            p.use_smooth = s


def mesh_from_pydata(name, verts, faces, collection=None, material=None, smooth=False, angle_deg=35.0):
    if isinstance(collection, str):
        collection = get_or_create_collection(collection)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], [tuple(f) for f in faces])
    me.update()
    _finalize_mesh(me, [smooth] * len(me.polygons), angle_deg)
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    if material is not None:
        assign_material(obj, material)
    return obj


# ----------------------------------------------------------------------------- Builder DSL
class Builder:
    """Accumulate closed primitives, then build() one mesh object.

    Every primitive appends closed geometry; material index per face is supported by passing a
    list of materials to build() and `mat=<index>` to each primitive."""

    def __init__(self):
        self.v, self.f, self.m, self.s = [], [], [], []

    def add(self, verts, faces, mat=0, smooth=False):
        off = len(self.v)
        self.v.extend(tuple(p) for p in verts)
        self.f.extend(tuple(off + i for i in face) for face in faces)
        self.m.extend([mat] * len(faces))
        self.s.extend([smooth] * len(faces))
        return self

    def box(self, center, dims, mat=0, angle_z=0.0):
        x, y, z = center
        dx, dy, dz = [d / 2.0 for d in dims]
        ca, sa = cos(angle_z), sin(angle_z)
        corners = [(-dx, -dy, -dz), (dx, -dy, -dz), (dx, dy, -dz), (-dx, dy, -dz),
                   (-dx, -dy, dz), (dx, -dy, dz), (dx, dy, dz), (-dx, dy, dz)]
        verts = [(x + u * ca - v * sa, y + u * sa + v * ca, z + w) for u, v, w in corners]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        return self.add(verts, faces, mat)

    def plane(self, center, size, mat=0, thickness=0.02):
        """A thin closed slab (planes with zero thickness are not manifold)."""
        return self.box(center, (size[0], size[1], thickness), mat)

    def cylinder(self, a, b, r, mat=0, n=16, r2=None, smooth=True):
        a, b = Vector(a), Vector(b)
        axis = (b - a)
        if axis.length < 1e-9:
            return self
        axis.normalize()
        ref = Vector((0, 0, 1)) if abs(axis.z) < 0.95 else Vector((1, 0, 0))
        u = axis.cross(ref).normalized()
        w = axis.cross(u).normalized()
        r2 = r if r2 is None else r2
        ring = lambda p, rr: [p + rr * (u * cos(2 * pi * i / n) + w * sin(2 * pi * i / n)) for i in range(n)]
        verts = ring(a, r) + ring(b, r2)
        faces = [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
        faces.append(tuple(reversed(range(n))))
        faces.append(tuple(n + i for i in range(n)))
        return self.add(verts, faces, mat, smooth)

    def cone(self, a, b, r1, r2=0.001, mat=0, n=16, smooth=True):
        return self.cylinder(a, b, r1, mat, n, max(r2, 0.0005), smooth)

    def lathe(self, center, profile, mat=0, n=32, smooth=True):
        """profile: list of (z, radius) from bottom to top; radii at ends may be tiny but > 0."""
        x, y, z0 = center
        prof = [(z, max(r, 0.0005)) for z, r in profile]
        verts = [(x + r * cos(2 * pi * i / n), y + r * sin(2 * pi * i / n), z0 + z) for z, r in prof for i in range(n)]
        faces = []
        for j in range(len(prof) - 1):
            for i in range(n):
                faces.append((j * n + i, j * n + (i + 1) % n, (j + 1) * n + (i + 1) % n, (j + 1) * n + i))
        faces.append(tuple(reversed(range(n))))
        faces.append(tuple((len(prof) - 1) * n + i for i in range(n)))
        return self.add(verts, faces, mat, smooth)

    def sphere(self, center, r, mat=0, n=24, smooth=True):
        rings = max(4, n // 2)
        profile = [(-r * cos(pi * k / rings), max(0.0005, r * sin(pi * k / rings))) for k in range(rings + 1)]
        return self.lathe(center, profile, mat, n, smooth)

    def torus(self, center, R, r, mat=0, n=32, k=12, axis=(0, 0, 1), smooth=True):
        c = Vector(center)
        ax = Vector(axis).normalized()
        ref = Vector((1, 0, 0)) if abs(ax.x) < 0.9 else Vector((0, 1, 0))
        u = ax.cross(ref).normalized()
        w = ax.cross(u).normalized()
        verts = []
        for i in range(n):
            d = u * cos(2 * pi * i / n) + w * sin(2 * pi * i / n)
            for j in range(k):
                verts.append(c + d * (R + r * cos(2 * pi * j / k)) + ax * r * sin(2 * pi * j / k))
        faces = [(i * k + j, ((i + 1) % n) * k + j, ((i + 1) % n) * k + (j + 1) % k, i * k + (j + 1) % k)
                 for i in range(n) for j in range(k)]
        return self.add(verts, faces, mat, smooth)

    def tube(self, points, r, mat=0, n=10, smooth=True):
        pts = [Vector(p) for p in points]
        if len(pts) < 2:
            return self
        verts, prev_u = [], None
        for i, p in enumerate(pts):
            axis = (pts[min(i + 1, len(pts) - 1)] - pts[max(0, i - 1)])
            axis = axis.normalized() if axis.length > 1e-9 else Vector((0, 0, 1))
            if prev_u is None:
                ref = Vector((0, 0, 1)) if abs(axis.z) < 0.9 else Vector((1, 0, 0))
                u = axis.cross(ref).normalized()
            else:
                u = (prev_u - axis * prev_u.dot(axis))
                u = u.normalized() if u.length > 1e-9 else prev_u
            w = axis.cross(u).normalized()
            prev_u = u
            verts.extend(p + r * (u * cos(2 * pi * j / n) + w * sin(2 * pi * j / n)) for j in range(n))
        faces = [(i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j)
                 for i in range(len(pts) - 1) for j in range(n)]
        faces.append(tuple(reversed(range(n))))
        faces.append(tuple((len(pts) - 1) * n + j for j in range(n)))
        return self.add(verts, faces, mat, smooth)

    def prism(self, polygon_xz, depth, origin, mat=0, angle_z=0.0):
        """Extrude a 2D outline given in the X/Z plane through local Y (depth). Closed at both ends.
        polygon_xz must be counter-clockwise, non self-intersecting."""
        ca, sa = cos(angle_z), sin(angle_z)
        o = Vector(origin)
        N = len(polygon_xz)
        verts = []
        for y in (-depth / 2.0, depth / 2.0):
            for x, z in polygon_xz:
                verts.append((o.x + x * ca - y * sa, o.y + x * sa + y * ca, o.z + z))
        faces = [tuple(reversed(range(N))), tuple(N + i for i in range(N))]
        faces += [(i, (i + 1) % N, (i + 1) % N + N, i + N) for i in range(N)]
        return self.add(verts, faces, mat)

    def build(self, name, collection=None, material=None, angle_deg=35.0, merge_dist=0.0):
        # merge_dist defaults to 0: every primitive added here is already a closed shell, and welding
        # by distance fuses vertices where two shells touch (a rivet meeting a hull, a trim ring on a
        # cone), which creates non-manifold edges. Pass a small value only for hand-built geometry.
        if not self.f:
            raise ValueError(f"Builder.build('{name}'): no geometry was added before build()")
        if isinstance(collection, str):
            collection = get_or_create_collection(collection)
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        material = _resolve_material(material)
        mats = list(material) if isinstance(material, (list, tuple)) else ([material] if material else [])
        for m in mats:
            me.materials.append(m)
        for p, mi, sm in zip(me.polygons, self.m, self.s):
            p.material_index = min(mi, max(len(mats) - 1, 0))
            p.use_smooth = sm
        smooth_flags = [p.use_smooth for p in me.polygons]
        _finalize_mesh(me, None, angle_deg, merge_dist)
        # remove_doubles can change face count; re-apply smoothing by rebuilding flags when counts match
        if len(me.polygons) == len(smooth_flags):
            for p, s in zip(me.polygons, smooth_flags):
                p.use_smooth = s
        obj = bpy.data.objects.new(name, me)
        (collection or bpy.context.scene.collection).objects.link(obj)
        self.v, self.f, self.m, self.s = [], [], [], []
        return obj


# ----------------------------------------------------------------------------- materials
def _set_input(node, names, value):
    for n in names:
        sock = node.inputs.get(n)
        if sock is not None:
            try:
                sock.default_value = value
                return sock
            except Exception:
                pass
    return None


def _enable_nodes(datablock):
    if hasattr(datablock, "use_nodes"):
        try:
            datablock.use_nodes = True
        except Exception:
            pass
    return datablock.node_tree


def _rgba(c, a=1.0):
    c = tuple(c)
    return (c[0], c[1], c[2], c[3] if len(c) > 3 else a)


def make_material(name, base_color=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5, emission_color=None,
                  emission_strength=0.0, alpha=1.0, noise=None, textures=None, ior=1.45):
    """Principled material, version-safe socket names.

    noise = dict(scale=5, detail=4, color_a=(r,g,b), color_b=(r,g,b), bump=0.15)  -> procedural variation
    textures = dict(base_color=path, roughness=path, normal=path, metallic=path, scale=1.0) -> image maps
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.diffuse_color = _rgba(base_color)
    tree = _enable_nodes(mat)
    if tree is None:
        return mat
    nodes, links = tree.nodes, tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out = nodes.get("Material Output") or nodes.new("ShaderNodeOutputMaterial")
        links.new(bsdf.outputs[0], out.inputs["Surface"])
    _set_input(bsdf, ["Base Color"], _rgba(base_color))
    _set_input(bsdf, ["Metallic"], metallic)
    _set_input(bsdf, ["Roughness"], roughness)
    _set_input(bsdf, ["IOR"], ior)
    _set_input(bsdf, ["Alpha"], alpha)
    if alpha < 1.0:
        for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
    if emission_color is not None and emission_strength > 0:
        _set_input(bsdf, ["Emission Color", "Emission"], _rgba(emission_color))
        _set_input(bsdf, ["Emission Strength"], emission_strength)
    if noise:
        tex = nodes.new("ShaderNodeTexNoise")
        tex.location = (-700, 200)
        _set_input(tex, ["Scale"], noise.get("scale", 5.0))
        _set_input(tex, ["Detail"], noise.get("detail", 4.0))
        _set_input(tex, ["Roughness"], noise.get("roughness", 0.6))
        coord = nodes.new("ShaderNodeTexCoord")
        coord.location = (-900, 200)
        links.new(coord.outputs["Object"], tex.inputs["Vector"])
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.location = (-450, 200)
        ramp.color_ramp.elements[0].color = _rgba(noise.get("color_a", base_color))
        ramp.color_ramp.elements[1].color = _rgba(noise.get("color_b", tuple(min(1, c * 1.25) for c in base_color[:3])))
        ramp.color_ramp.elements[0].position = noise.get("pos_a", 0.3)
        ramp.color_ramp.elements[1].position = noise.get("pos_b", 0.7)
        links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bump_strength = noise.get("bump", 0.0)
        if bump_strength > 0:
            bump = nodes.new("ShaderNodeBump")
            bump.location = (-300, -200)
            _set_input(bump, ["Strength"], bump_strength)
            _set_input(bump, ["Distance"], noise.get("bump_distance", 0.02))
            links.new(tex.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    if textures:
        coord = nodes.new("ShaderNodeTexCoord")
        coord.location = (-1100, 0)
        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (-900, 0)
        s = textures.get("scale", 1.0)
        _set_input(mapping, ["Scale"], (s, s, s))
        links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        y = 300
        for key, socket_names, noncolor in (("base_color", ["Base Color"], False), ("roughness", ["Roughness"], True),
                                            ("metallic", ["Metallic"], True), ("normal", None, True)):
            path = textures.get(key)
            if not path:
                continue
            try:
                img = bpy.data.images.load(path, check_existing=True)
            except Exception as e:
                print(f"[gap_helpers] could not load texture {path}: {e}")
                continue
            if noncolor:
                img.colorspace_settings.name = "Non-Color"
            node = nodes.new("ShaderNodeTexImage")
            node.image = img
            node.location = (-650, y)
            node.extension = "REPEAT"
            y -= 300
            links.new(mapping.outputs["Vector"], node.inputs["Vector"])
            if socket_names:
                sock = next((bsdf.inputs[n] for n in socket_names if n in bsdf.inputs), None)
                if sock is not None:
                    links.new(node.outputs["Color"], sock)
            else:
                nm = nodes.new("ShaderNodeNormalMap")
                nm.location = (-350, y + 300)
                links.new(node.outputs["Color"], nm.inputs["Color"])
                links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def assign_material(obj, mat, slot=0):
    if obj.type != "MESH":
        return obj
    mat = _resolve_material(mat)
    if isinstance(mat, list):
        for i, m in enumerate(mat):
            assign_material(obj, m, i)
        return obj
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        if slot < len(obj.data.materials):
            obj.data.materials[slot] = mat
        else:
            obj.data.materials.append(mat)
    return obj


def set_smooth(obj, angle_deg=35.0):
    if obj.type != "MESH":
        return obj
    me = obj.data
    for p in me.polygons:
        p.use_smooth = True
    bm = bmesh.new()
    bm.from_mesh(me)
    thr = math.radians(angle_deg)
    for e in bm.edges:
        if e.is_manifold:
            e.smooth = e.calc_face_angle(0.0) < thr
    bm.to_mesh(me)
    bm.free()
    me.update()
    return obj


# ----------------------------------------------------------------------------- lights / camera / world
def _look_at(obj, target):
    direction = Vector(target) - obj.location
    if direction.length < 1e-9:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_light(kind="POINT", location=(0, 0, 5), energy=1000.0, color=(1.0, 1.0, 1.0), name="Light",
              target=None, size=1.0, spot_angle_deg=45.0, collection=None):
    kind = kind.upper()
    data = bpy.data.lights.new(name, kind)
    data.energy = energy
    data.color = tuple(color)[:3]
    if kind == "AREA":
        data.size = size
    if kind == "SPOT":
        data.spot_size = math.radians(spot_angle_deg)
    if kind in ("POINT", "SPOT"):
        data.shadow_soft_size = size * 0.25
    obj = bpy.data.objects.new(name, data)
    obj.location = Vector(location)
    (collection or bpy.context.scene.collection).objects.link(obj)
    if target is not None:
        _look_at(obj, target)
    return obj


def add_sun(elevation_deg=45.0, azimuth_deg=135.0, strength=3.0, color=(1.0, 0.95, 0.9), name="Sun_Key",
            angle_deg=1.0, collection=None):
    data = bpy.data.lights.new(name, "SUN")
    data.energy = strength
    data.color = tuple(color)[:3]
    data.angle = math.radians(angle_deg)
    obj = bpy.data.objects.new(name, data)
    obj.location = (0, 0, 20)
    obj.rotation_euler = (math.radians(90 - elevation_deg), 0, math.radians(azimuth_deg))
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def add_camera(location=(12, -12, 8), target=(0, 0, 1), lens_mm=35.0, name="Camera_Main", ortho=False,
               ortho_scale=20.0, make_active=True, collection=None):
    data = bpy.data.cameras.new(name)
    data.lens = lens_mm
    if ortho:
        data.type = "ORTHO"
        data.ortho_scale = ortho_scale
    data.clip_end = 5000.0
    obj = bpy.data.objects.new(name, data)
    obj.location = Vector(location)
    (collection or bpy.context.scene.collection).objects.link(obj)
    _look_at(obj, target)
    if make_active:
        bpy.context.scene.camera = obj
    return obj


def scene_bounds(exclude_types=("CAMERA", "LIGHT", "EMPTY")):
    lo = Vector((1e18, 1e18, 1e18))
    hi = Vector((-1e18, -1e18, -1e18))
    found = False
    for ob in bpy.context.scene.objects:
        if ob.type in exclude_types or ob.type != "MESH" or ob.hide_render:
            continue
        for corner in ob.bound_box:
            w = ob.matrix_world @ Vector(corner)
            lo = Vector((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
            hi = Vector((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
            found = True
    if not found:
        return Vector((-1, -1, 0)), Vector((1, 1, 2))
    return lo, hi


def frame_camera_to_scene(camera, margin=1.15, direction=(1.0, -1.3, 0.75)):
    """Move `camera` along `direction` so the whole scene fills the frame (perspective or ortho).

    Works at any scale: a 0.3 m ornament frames as tightly as a 30 m tower. The distance accounts
    for BOTH the horizontal and vertical field of view and the render aspect ratio, so the subject
    fits in the narrow dimension instead of overflowing it.
    """
    bpy.context.view_layer.update()
    lo, hi = scene_bounds()
    center = (lo + hi) / 2.0
    radius = max((hi - lo).length / 2.0, 1e-4)
    d = Vector(direction).normalized()
    scene = bpy.context.scene
    res_x = max(scene.render.resolution_x, 1)
    res_y = max(scene.render.resolution_y, 1)
    aspect = (res_x * scene.render.pixel_aspect_x) / (res_y * scene.render.pixel_aspect_y)
    if camera.data.type == "ORTHO":
        # ortho_scale covers the larger image dimension
        camera.data.ortho_scale = radius * 2.0 * margin / (1.0 if aspect >= 1 else 1.0)
        camera.location = center + d * max(radius * 4.0, 1e-3)
    else:
        # camera.data.angle applies to the larger image dimension (sensor_fit AUTO)
        half = camera.data.angle / 2.0
        if aspect >= 1.0:
            half_x, half_y = half, math.atan(math.tan(half) / aspect)
        else:
            half_y, half_x = half, math.atan(math.tan(half) * aspect)
        limiting = max(min(half_x, half_y), math.radians(1.0))
        dist = radius / math.sin(limiting) * margin
        camera.location = center + d * dist
    _look_at(camera, center)
    camera.data.clip_start = min(camera.data.clip_start, max(radius * 0.01, 1e-4))
    camera.data.clip_end = max(camera.data.clip_end, radius * 50.0)
    return camera


def set_world(color=(0.05, 0.07, 0.1), strength=1.0, hdri_path=None, name="World"):
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.get(name) or bpy.data.worlds.new(name)
    scene.world = world
    tree = _enable_nodes(world)
    if tree is None:
        return world
    nodes, links = tree.nodes, tree.links
    bg = next((n for n in nodes if n.type == "BACKGROUND"), None) or nodes.new("ShaderNodeBackground")
    out = next((n for n in nodes if n.type == "OUTPUT_WORLD"), None) or nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    _set_input(bg, ["Strength"], strength)
    if hdri_path:
        try:
            img = bpy.data.images.load(hdri_path, check_existing=True)
            env = next((n for n in nodes if n.type == "TEX_ENVIRONMENT"), None) or nodes.new("ShaderNodeTexEnvironment")
            env.image = img
            links.new(env.outputs["Color"], bg.inputs["Color"])
            return world
        except Exception as e:
            print(f"[gap_helpers] HDRI load failed ({e}); using flat color")
    _set_input(bg, ["Color"], _rgba(color))
    return world


# ----------------------------------------------------------------------------- terrain / scatter
def terrain(name="Terrain", size=(60.0, 60.0), resolution=96, height=3.0, noise_scale=0.05, seed=1,
            octaves=4, collection=None, material=None, thickness=1.0, falloff=0.0):
    """Closed terrain slab: displaced top surface + skirt + flat bottom (manifold, validator-friendly).
    falloff>0 lowers the edges towards zero (0..1 of half-size) so terrain fades flat at the border."""
    random.seed(seed)
    ox, oy = random.uniform(-1000, 1000), random.uniform(-1000, 1000)
    nx, ny = max(2, int(resolution)), max(2, int(resolution))
    sx, sy = size
    top = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = -sx / 2 + sx * i / nx
            y = -sy / 2 + sy * j / ny
            h = 0.0
            amp, freq = 1.0, noise_scale
            for _ in range(octaves):
                h += amp * noise.noise(Vector((x * freq + ox, y * freq + oy, 0.0)))
                amp *= 0.5
                freq *= 2.0
            h = (h + 1.0) * 0.5 * height
            if falloff > 0:
                ex = max(abs(x) / (sx / 2), abs(y) / (sy / 2))
                fade = 1.0 - max(0.0, (ex - (1.0 - falloff)) / max(falloff, 1e-6))
                h *= max(0.0, min(1.0, fade))
            top.append((x, y, h))
    n_top = len(top)
    bottom = [(x, y, -thickness) for x, y, _ in top]
    verts = top + bottom
    faces = []
    idx = lambda i, j: j * (nx + 1) + i
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            faces.append((a, b, c, d))
            faces.append((n_top + a, n_top + d, n_top + c, n_top + b))
    # skirt
    ring = [idx(i, 0) for i in range(nx + 1)] + [idx(nx, j) for j in range(1, ny + 1)] + \
           [idx(i, ny) for i in range(nx - 1, -1, -1)] + [idx(0, j) for j in range(ny - 1, 0, -1)]
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        faces.append((a, n_top + a, n_top + b, b))
    obj = mesh_from_pydata(name, verts, faces, collection, material, smooth=True, angle_deg=60)
    return obj


def _surface_height(surface, x, y):
    if surface is None:
        return 0.0
    from mathutils.bvhtree import BVHTree
    bvh = _BVH_CACHE.get(surface.name)
    if bvh is None:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bvh = BVHTree.FromObject(surface, depsgraph)
        _BVH_CACHE[surface.name] = bvh
    inv = surface.matrix_world.inverted()
    origin = inv @ Vector((x, y, 10000.0))
    direction = (inv.to_3x3() @ Vector((0, 0, -1))).normalized()
    hit = bvh.ray_cast(origin, direction)
    if hit[0] is None:
        return 0.0
    return (surface.matrix_world @ hit[0]).z


_BVH_CACHE = {}


def duplicate(obj, name, location=None, collection=None, linked_data=True):
    new = obj.copy()
    if not linked_data and obj.data is not None:
        new.data = obj.data.copy()
    new.name = name
    if location is not None:
        new.location = Vector(location)
    (collection or (obj.users_collection[0] if obj.users_collection else bpy.context.scene.collection)).objects.link(new)
    return new


def scatter(template, count=50, area=(-20, 20, -20, 20), seed=1, scale_range=(0.8, 1.2), min_distance=1.0,
            surface=None, collection=None, rotate_z=True, name_prefix=None, max_tries=30):
    """Place linked duplicates of `template` on a rectangle (or on `surface` via ray cast)."""
    rng = random.Random(seed)
    placed, objects = [], []
    prefix = name_prefix or template.name
    xmin, xmax, ymin, ymax = area
    for i in range(count):
        for _ in range(max_tries):
            x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
            if all((x - px) ** 2 + (y - py) ** 2 >= min_distance ** 2 for px, py in placed):
                break
        else:
            continue
        placed.append((x, y))
        z = _surface_height(surface, x, y)
        new = duplicate(template, f"{prefix}_{i + 1:03d}", (x, y, z), collection)
        s = rng.uniform(*scale_range)
        new.scale = (s, s, s)
        if rotate_z:
            new.rotation_euler = (0, 0, rng.uniform(0, 2 * pi))
        objects.append(new)
    if template not in objects and template.users_collection:
        # keep the template out of the way but present for data sharing
        template.hide_render = True
        template.hide_viewport = True
        template.location = (0, 0, -1000)
    return objects


def array_copies(obj, count, offset=(2.0, 0.0, 0.0), collection=None):
    out = []
    for i in range(1, count):
        loc = obj.location + Vector(offset) * i
        out.append(duplicate(obj, f"{obj.name}_{i + 1:03d}", loc, collection))
    return out


# ----------------------------------------------------------------------------- detail helpers
# These exist so generated scripts can add believable fine detail (rivets, trim, panel seams,
# portholes, motifs) with one call instead of hand-rolling geometry or shader node trees.

_AXES = {"X": Vector((1, 0, 0)), "Y": Vector((0, 1, 0)), "Z": Vector((0, 0, 1))}


def _basis(axis):
    """Return (axis_vector, u, v) for a named axis, u/v spanning the perpendicular plane."""
    a = _AXES[str(axis).upper()] if isinstance(axis, str) else Vector(axis).normalized()
    ref = Vector((0, 0, 1)) if abs(a.z) < 0.9 else Vector((1, 0, 0))
    u = a.cross(ref).normalized()
    v = a.cross(u).normalized()
    return a, u, v


def ring_positions(radius, count, z=0.0, phase=0.0, center=(0.0, 0.0), axis="Z"):
    """World positions evenly spaced on a circle. Use to place your own parts in a ring."""
    a, u, v = _basis(axis)
    origin = Vector((center[0], center[1], z)) if len(center) == 2 else Vector(center)
    out = []
    for i in range(int(count)):
        ang = phase + 2 * pi * i / max(int(count), 1)
        out.append(origin + radius * (u * cos(ang) + v * sin(ang)))
    return out


def rivet_ring(builder, center, radius, count, rivet_r=0.004, mat=0, protrusion=0.6, axis="Z",
               phase=0.0, n=8):
    """A ring of dome-headed rivets/studs around `center` at `radius`.

    protrusion: how far the head stands out, as a multiple of rivet_r (0.6 = a low dome).
    axis: the axis the ring circles around ("Z" for a ring around an upright body).
    """
    a, u, v = _basis(axis)
    c = Vector(center)
    for i in range(int(count)):
        ang = phase + 2 * pi * i / max(int(count), 1)
        d = (u * cos(ang) + v * sin(ang))
        p = c + d * radius
        builder.sphere(p + d * rivet_r * (protrusion - 1.0), rivet_r, mat=mat, n=max(6, n))
    return builder


def bolt(builder, center, r=0.005, height=0.004, mat=0, n=6, axis="Z"):
    """A single hex bolt head standing proud of a surface."""
    a, _, _ = _basis(axis)
    c = Vector(center)
    builder.cylinder(c, c + a * height, r, mat=mat, n=max(3, n), smooth=False)
    return builder


def trim_ring(builder, center, radius, tube_r=0.004, mat=0, axis="Z", n=48, k=10):
    """A raised collar / beading / trim band encircling a body."""
    builder.torus(center, radius, tube_r, mat=mat, n=n, k=k, axis=_basis(axis)[0])
    return builder


def panel_seams(builder, center, radius, z0, z1, count=6, width=0.003, depth=0.004, mat=0, phase=0.0):
    """Vertical seams dividing a cylindrical surface into panels.

    Modelled as thin raised strips on the surface (closed boxes), which reads as a panel line in
    render and keeps the mesh manifold - no boolean cutting.
    """
    c = Vector(center)
    height = abs(z1 - z0)
    zc = min(z0, z1) + height / 2.0
    for i in range(int(count)):
        ang = phase + 2 * pi * i / max(int(count), 1)
        x = c.x + radius * cos(ang)
        y = c.y + radius * sin(ang)
        builder.box((x, y, zc), (depth, width, height), mat=mat, angle_z=ang)
    return builder


def porthole(builder, center, outer_r=0.05, inner_r=0.035, depth=0.02, mat_frame=0, mat_glass=1,
             normal="Y", rivets=0, rivet_r=0.004, phase=0.0):
    """A round window: raised frame ring + recessed glass disc, facing along `normal`.

    normal: the direction the window looks out ("Y" = facing -Y/+Y, "X", or a vector).
    rivets: number of rivets around the frame (0 for none).
    """
    a, u, v = _basis(normal)
    c = Vector(center)
    # frame: a short thick ring standing proud of the hull
    builder.torus(c, (outer_r + inner_r) / 2.0, (outer_r - inner_r) / 2.0, mat=mat_frame,
                  n=48, k=12, axis=a)
    # glass: a thin disc set back inside the frame
    builder.cylinder(c - a * depth * 0.5, c - a * depth * 0.15, inner_r, mat=mat_glass, n=48)
    if rivets:
        rivet_ring(builder, c, outer_r + (outer_r - inner_r) * 0.15, int(rivets), rivet_r,
                   mat=mat_frame, axis=a, phase=phase)
    return builder


def _extrude_polygon(builder, points_2d, center, depth, mat, normal, angle):
    """Extrude a 2D outline (in the plane perpendicular to `normal`) into a closed solid."""
    a, u, v = _basis(normal)
    c = Vector(center)
    ca, sa = cos(angle), sin(angle)
    verts = []
    N = len(points_2d)
    for sign in (-0.5, 0.5):
        for (px, py) in points_2d:
            rx = px * ca - py * sa
            ry = px * sa + py * ca
            verts.append(c + u * rx + v * ry + a * depth * sign)
    faces = [tuple(range(N))[::-1], tuple(N + i for i in range(N))]
    faces += [(i, (i + 1) % N, (i + 1) % N + N, i + N) for i in range(N)]
    builder.add(verts, faces, mat)
    return builder


def star_shape(builder, center, outer_r=0.02, inner_r=None, points=5, depth=0.004, mat=0,
               normal="Y", angle=0.0):
    """A raised N-pointed star badge, extruded along `normal`."""
    inner_r = outer_r * 0.42 if inner_r is None else inner_r
    pts = []
    for i in range(int(points) * 2):
        r = outer_r if i % 2 == 0 else inner_r
        ang = pi / 2 + i * pi / int(points)
        pts.append((r * cos(ang), r * sin(ang)))
    return _extrude_polygon(builder, pts, center, depth, mat, normal, angle)


def crescent_shape(builder, center, r=0.02, cut_offset=None, depth=0.004, mat=0, normal="Y",
                   angle=0.0, segments=28):
    """A raised crescent moon badge: a disc with a smaller disc bitten out of one side."""
    cut_offset = r * 0.45 if cut_offset is None else cut_offset
    cut_r = r * 0.82
    outer, inner = [], []
    for i in range(segments + 1):
        t = pi * (-0.5 + i / segments)  # outer arc, right half
        outer.append((r * cos(t), r * sin(t)))
    for i in range(segments + 1):
        t = pi * (0.5 - i / segments)   # inner arc back, offset circle
        inner.append((cut_offset + cut_r * cos(t), cut_r * sin(t)))
    pts = outer + inner
    # drop duplicate endpoints that would create zero-area faces
    cleaned = [pts[0]]
    for p in pts[1:]:
        if (p[0] - cleaned[-1][0]) ** 2 + (p[1] - cleaned[-1][1]) ** 2 > 1e-12:
            cleaned.append(p)
    if (cleaned[0][0] - cleaned[-1][0]) ** 2 + (cleaned[0][1] - cleaned[-1][1]) ** 2 < 1e-12:
        cleaned.pop()
    return _extrude_polygon(builder, cleaned, center, depth, mat, normal, angle)


# ----------------------------------------------------------------------------- removal (explicit only)
def delete_objects(names):
    removed = 0
    for n in list(names):
        ob = bpy.data.objects.get(n)
        if ob is not None:
            data = ob.data
            bpy.data.objects.remove(ob, do_unlink=True)
            if data is not None and data.users == 0:
                try:
                    if isinstance(data, bpy.types.Mesh):
                        bpy.data.meshes.remove(data)
                except Exception:
                    pass
            removed += 1
    print(f"[gap_helpers] removed {removed} objects", flush=True)
    return removed


def clear_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.images):
        for item in list(block):
            if item.users == 0:
                try:
                    block.remove(item)
                except Exception:
                    pass
    for coll in list(bpy.data.collections):
        if not coll.objects and not coll.children:
            bpy.data.collections.remove(coll)
    print("[gap_helpers] scene cleared", flush=True)
