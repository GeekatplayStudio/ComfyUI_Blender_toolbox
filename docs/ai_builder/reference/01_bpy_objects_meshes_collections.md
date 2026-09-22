# bpy data API: objects, meshes, collections (background-safe)

## Creating a mesh object from vertices and faces
```python
import bpy, bmesh
me = bpy.data.meshes.new("Wall_Mesh")
me.from_pydata(verts, [], faces)      # verts: [(x,y,z),...]  faces: [(i0,i1,i2,i3),...] counter-clockwise seen from outside
me.update()
obj = bpy.data.objects.new("Wall_North", me)
bpy.context.scene.collection.objects.link(obj)   # or my_collection.objects.link(obj)
```
Prefer `gap_helpers.mesh_from_pydata(...)` or `Builder` - they also merge doubles, recalc normals and
mark sharp edges.

## Collections
```python
coll = bpy.data.collections.get("Step_02_Houses") or bpy.data.collections.new("Step_02_Houses")
if coll.name not in bpy.context.scene.collection.children:
    bpy.context.scene.collection.children.link(coll)
coll.objects.link(obj)
```
An object can be in several collections; `obj.users_collection` lists them. `gap_helpers.link_object`
moves an object to exactly one collection.

## Transforms
```python
obj.location = (x, y, z)
obj.rotation_euler = (rx, ry, rz)          # radians
obj.scale = (sx, sy, sz)
obj.matrix_world.translation             # world position
bpy.context.view_layer.update()          # refresh matrix_world / dimensions after changes
obj.dimensions                           # world-space size (read-only-ish, after update)
```
Prefer baking geometry at the right size instead of non-uniform object scale (scaled objects give
uneven displacement/bevels and confuse later steps).

## Duplicating
```python
new = obj.copy()                # new object sharing the same mesh data (cheap, ideal for scatter)
new.data = obj.data.copy()      # only if the copy must be edited independently
new.name = "Tree_Pine_017"
coll.objects.link(new)
```

## Deleting (only when the instruction asks)
```python
ob = bpy.data.objects.get("Old_Fence")
if ob: bpy.data.objects.remove(ob, do_unlink=True)
for me in [m for m in bpy.data.meshes if m.users == 0]: bpy.data.meshes.remove(me)
```

## bmesh essentials
```python
bm = bmesh.new(); bm.from_mesh(me)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)          # outward normals for closed meshes
bmesh.ops.inset_region(bm, faces=[f], thickness=0.1, depth=0.0)
bmesh.ops.extrude_face_region(bm, geom=[f])
bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.02, segments=2, affect='EDGES')
bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
bm.to_mesh(me); bm.free(); me.update()
```
Iterate `bm.verts / bm.edges / bm.faces`; call `bm.verts.ensure_lookup_table()` before indexing.
`e.is_manifold`, `e.is_boundary`, `f.calc_area()`, `f.normal`, `e.calc_face_angle(0.0)`.

## Modifiers (data API, no operators needed)
```python
m = obj.modifiers.new("Bevel", 'BEVEL'); m.width = 0.03; m.segments = 2; m.limit_method = 'ANGLE'
m = obj.modifiers.new("Array", 'ARRAY'); m.count = 8; m.relative_offset_displace = (1.1, 0, 0)
m = obj.modifiers.new("Solidify", 'SOLIDIFY'); m.thickness = 0.1
m = obj.modifiers.new("Subsurf", 'SUBSURF'); m.levels = 1; m.render_levels = 2
m = obj.modifiers.new("Mirror", 'MIRROR'); m.use_axis = (True, False, False)
```
The validator inspects the base mesh; modifiers do not hide topology problems. A BOOLEAN modifier
often creates non-manifold results - prefer building closed parts that intersect (allowed) instead
of booleans.

## Smooth shading without operators
Set `p.use_smooth = True` on polygons and mark sharp edges by angle in bmesh (`e.smooth = False`).
`gap_helpers.set_smooth(obj, angle_deg)` does exactly this and works on 3.6 - 5.x
(`use_auto_smooth` was removed in Blender 4.1).

## Operators that DO work in background mode (use sparingly)
`bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,1))` - result is `bpy.context.active_object` /
`bpy.context.object`. Works, but produces default names ("Cube") - rename immediately. Anything that
needs `context.selected_objects` in EDIT mode does NOT work; use bmesh instead.
