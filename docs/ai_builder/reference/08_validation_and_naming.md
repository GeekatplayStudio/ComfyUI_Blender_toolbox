# Passing validation: causes and fixes

## "N non-manifold edges"
Edges shared by 3+ faces or internal faces. Causes: two shells sharing vertices after
`remove_doubles` merged them; a face inserted inside a solid; boolean modifiers.
Fix: keep separate shells from touching exactly (offset by 0.01 m), or build the joined shape as one
outline (`prism`), or increase separation; avoid BOOLEAN modifiers.

## "loose vertices / loose edges"
Vertices without edges or edges without faces: leftover points, helper edges, zero-length extrusions.
Fix: only append complete faces; let `Builder.build()` (dissolve_degenerate) clean tiny leftovers.

## "zero-area faces"
Degenerate faces from coincident vertices (e.g. a lathe profile with radius 0, a cone tip at radius 0,
a box with a 0 dimension). Fix: use radius >= 0.0005 at tips, never pass a 0 dimension.

## "flipped / inconsistent normals"
Faces wound clockwise when seen from outside, or a mixed mesh. Fix: build with `Builder` (recalc
outward), or run `bmesh.ops.recalc_face_normals` after `from_pydata`. Auto-fix can repair this when
the node enables `auto_fix_normals`.

## "boundary edges (open mesh)" - warning
A surface without thickness. Acceptable for a ground plane; for props/buildings use closed slabs
(`Builder.plane`, `terrain`) so lighting and later booleans behave.

## "n-gons" - warning
Cylinder/cone caps are n-gons by design and render fine. Reduce by using `n` <= 32 or leave as is.

## "default name"
Objects/meshes/materials/lights/cameras named like `Cube`, `Cube.001`, `Material.002`, `Light`,
`Camera`, collections named `Collection`. Rename on creation. Pattern: `Type_Detail[_Index]`.

## "missing texture"
`bpy.data.images.load()` on a path that does not exist (or wrong slashes). Only use paths given in
the instruction; write Windows paths as raw strings `r"C:\tex\brick.png"` or forward slashes.

## "no material" / "no UV with image textures"
Give every mesh a material; when using image textures add a UV layer (box projection snippet in
02_materials_shading.md).

## "no camera" / "no lights" (scene warnings)
The preview renderer adds a temporary `AI_Preview_Camera` / `AI_Preview_Sun` if missing. Add real
ones (`add_camera`, `add_sun`) in the lighting/camera step so the scene is self-contained.

## Polygon budgets
Keep a step under ~300k triangles. `terrain(resolution=128)` = 32k faces; `sphere(n=24)` = ~300
faces; `scatter(count=200)` of a 60-face tree = 12k faces (linked duplicates share data).
