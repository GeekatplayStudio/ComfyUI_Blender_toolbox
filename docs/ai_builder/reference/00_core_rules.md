# Core rules for generated Blender scripts (always read first)

## Execution contract
- The script runs in Blender BACKGROUND mode through `step_runner.py`: no window, no viewport, no
  selection, no active object. `bpy.context.scene` and `bpy.context.view_layer` exist and work.
- The runner opens the session `.blend` (or an empty scene), `exec()`s the script, validates, probes,
  SAVES the file, renders a preview. The script must never save, open or quit: no `bpy.ops.wm.*`.
- Allowed imports: `bpy`, `bmesh`, `mathutils`, `math`, `random`, `itertools`, `gap_helpers`.
  Anything touching files, network or processes is rejected by the safety scan before execution.
- Print `BUILT <name> <vertices>` for every object (use `log_built(obj)`); the runner parses those lines.

## Geometry rules the validator enforces (errors fail the step)
- No non-manifold edges, no loose vertices/edges, no zero-area faces, no flipped normals.
- Closed solids: use `Builder` primitives (box/cylinder/cone/sphere/torus/tube/lathe/prism) - they
  produce closed, consistently oriented meshes and recalculate normals outward in `build()`.
- Open surfaces (a single `plane`) are reported as warnings only, but prefer `Builder.plane()`
  (a thin closed slab) or `terrain()` (closed slab with skirt) for ground.
- Materials: every visible mesh gets one (`make_material` + pass it to `build()` or `assign_material`).
- Names: descriptive, never `Cube.001`. Pattern: `Category_Description` e.g. `Tower_North_Wall`,
  `Mat_WeatheredOak`, `Sun_Key`, `Camera_Main`, collection `Step_03_Bridge`.

## Coordinate conventions
- Units are meters. Z is up. Ground is Z = 0 unless the scene already has terrain (then place objects
  at the terrain height; `scatter(..., surface=terrain_obj)` does this automatically).
- `Builder.box(center, dims)` takes the CENTER: a 4 m tall wall sitting on the ground has center z = 2.
- `Builder.cylinder(a, b, r)` goes from point a to point b (any direction) with radius r.
- Rotation about Z for boxes/prisms: `angle_z` in radians (`math.radians(30)`).

## Scene continuity (multi-pass sessions)
- Read the CURRENT SCENE block in the prompt: it lists existing objects with locations, dimensions,
  materials and collections. Place new things relative to them; reuse existing materials by name:
  `mat = bpy.data.materials.get("Mat_Stone") or make_material("Mat_Stone", ...)`.
- Never delete or clear existing objects unless the instruction explicitly says remove/replace.
- Put each step's objects into `get_or_create_collection("<collection_name given in the prompt>")`.

## Minimal complete example
```python
from gap_helpers import *
import bpy, math

coll = get_or_create_collection("Step_01_Well")
stone = make_material("Mat_Stone_Grey", (0.45, 0.44, 0.4), roughness=0.9,
                      noise=dict(scale=6, detail=4, color_a=(0.35, 0.34, 0.31), color_b=(0.55, 0.54, 0.5), bump=0.15))
wood = make_material("Mat_Oak", (0.36, 0.22, 0.1), roughness=0.8)

B = Builder()
B.lathe((0, 0, 0), [(0.0, 1.2), (0.9, 1.2), (0.9, 1.05), (0.05, 1.05)], mat=0, n=32)   # ring wall
B.cylinder((-1.0, 0, 0.9), (-1.0, 0, 3.0), 0.08, mat=1, n=10)                        # posts
B.cylinder((1.0, 0, 0.9), (1.0, 0, 3.0), 0.08, mat=1, n=10)
B.cylinder((-1.1, 0, 2.9), (1.1, 0, 2.9), 0.06, mat=1, n=10)                          # axle
B.prism([(-1.4, 0), (1.4, 0), (0, 0.9)], 1.4, (0, 0, 3.0), mat=1)                    # gable roof block
well = B.build("Well_Stone", coll, [stone, wood])
log_built(well)
```
