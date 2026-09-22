# Blender version differences that break scripts (3.6 LTS -> 4.x -> 5.x)

## Render engines
- `'CYCLES'` everywhere.
- EEVEE: `'BLENDER_EEVEE'` in 3.6-4.1 and again in 5.x; `'BLENDER_EEVEE_NEXT'` in 4.2-4.5.
  Setting an invalid enum raises TypeError - the runner tries both. `'BLENDER_WORKBENCH'` always exists.
- EEVEE samples: `scene.eevee.taa_render_samples`. Cycles: `scene.cycles.samples`.

## Principled BSDF
- 4.0+ renamed sockets: `Specular` -> `Specular IOR Level`, `Transmission` -> `Transmission Weight`,
  `Emission` -> `Emission Color`, `Subsurface` -> `Subsurface Weight`, `Clearcoat` -> `Coat Weight`,
  `Sheen` -> `Sheen Weight`. 5.x keeps the 4.x names. Always look sockets up by name with a fallback.

## Removed / deprecated API
- `mesh.use_auto_smooth` removed in 4.1 -> mark sharp edges (bmesh `e.smooth = False`) or add the
  "Smooth by Angle" node-group modifier. `gap_helpers.set_smooth` handles it.
- `ShaderNodeTexMusgrave` removed in 4.1 (merged into Noise texture).
- `ShaderNodeMixRGB` still works but `ShaderNodeMix` (data_type='RGBA') is the 4.x+ node.
- `material.use_nodes` / `world.use_nodes`: still work in 5.x but print a deprecation warning
  (removal planned for 6.0). Guard with `if hasattr(mat, "use_nodes")`.
- `scene.use_nodes` (compositor) replaced in 5.x by `scene.compositing_node_group`.
- `bpy.ops.import_scene.obj` (3.x) -> `bpy.ops.wm.obj_import` (4.x+). Imports are not allowed in
  generated scripts anyway (bpy.ops.wm.*).
- `obj.select_set()` / `bpy.context.view_layer.objects.active` are fine in background mode, but
  operators that need an EDIT-mode context will still fail - use bmesh.
- Blender 4.2+ transparency: `mat.surface_render_method = 'BLENDED'`; older: `mat.blend_method = 'BLEND'`.
- Color management: default view transform is `'AgX'` (4.0+) instead of `'Filmic'`.

## Background-mode gotchas
- `bpy.context.object` / `active_object` may be None; never rely on selection state.
- `bpy.ops.object.mode_set(mode='EDIT')` fails without an active object; avoid modes entirely.
- `bpy.context.temp_override(area=..., region=...)` needs a window - not available in background.
- `bpy.context.evaluated_depsgraph_get()` works and is needed for BVH ray casts on modified meshes.
- Printing is captured by the runner; keep it informative but short (`BUILT` lines + key values).

## Python version
Blender 4.x ships Python 3.11; 5.x ships 3.13. Standard library only; no pip packages (numpy is
bundled and importable, but the safety scan only allows bpy/bmesh/mathutils/math/random/itertools/gap_helpers).
