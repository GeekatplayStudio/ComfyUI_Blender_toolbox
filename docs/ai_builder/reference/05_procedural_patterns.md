# Procedural patterns (copy, adapt, keep closed geometry)

## Building with walls, window recesses and a roof
```python
from gap_helpers import *
import math, random
coll = get_or_create_collection("Step_02_House")
plaster = make_material("Mat_Plaster", (0.85, 0.8, 0.7), roughness=0.9, noise=dict(scale=12, detail=3, bump=0.1))
timber  = make_material("Mat_Timber_Dark", (0.2, 0.12, 0.07), roughness=0.85)
tile    = make_material("Mat_Roof_Tile", (0.55, 0.25, 0.15), roughness=0.8)
glass   = make_material("Mat_Glass", (0.5, 0.65, 0.75), roughness=0.05, alpha=0.3)
W, D, H = 8.0, 6.0, 3.2
B = Builder()
B.box((0, 0, H / 2), (W, D, H), mat=0)                                   # body
B.prism([(-W/2 - 0.4, 0), (W/2 + 0.4, 0), (0, 2.6)], D + 0.8, (0, 0, H), mat=2)   # gable roof (closed prism)
for x in (-2.5, 0.0, 2.5):                                               # windows: recessed frame + glass slab
    B.box((x, -D/2 - 0.02, 1.6), (1.1, 0.25, 1.3), mat=1)               # frame proud of the wall
    B.box((x, -D/2 - 0.06, 1.6), (0.9, 0.05, 1.1), mat=3)               # glass
B.box((0, D/2 + 0.02, 1.05), (1.0, 0.25, 2.1), mat=1)                    # door block
house = B.build("House_Main", coll, [plaster, timber, tile, glass]); log_built(house)
```
Windows as recesses: a slightly protruding frame box + thin glass box reads as a window from any
distance and stays manifold (no booleans needed).

## Cylindrical tower with battlements and conical roof
```python
B = Builder()
B.lathe((0, 0, 0), [(0, 3.2), (0.6, 3.0), (11.0, 2.8), (11.6, 3.4), (12.6, 3.4), (12.6, 2.6), (11.8, 2.6)], mat=0, n=40)
for i in range(12):                                                     # merlons
    a = i * 2 * math.pi / 12
    B.box((3.05 * math.cos(a), 3.05 * math.sin(a), 13.1), (0.9, 0.6, 1.0), mat=0, angle_z=a)
B.cone((0, 0, 12.6), (0, 0, 17.5), 3.6, 0.05, mat=1, n=40)              # roof
B.sphere((0, 0, 17.7), 0.35, mat=1)                                     # finial
tower = B.build("Tower_Keep", coll, [stone, copper]); log_built(tower)
```

## Arches, bridges, columns
- Arch: `B.torus(center, R, r, axis=(0,1,0), n=32, k=8)` gives a full ring; for a half arch build a
  `lathe`-like ring of boxes: `for i in range(13): a = math.pi * i / 12; B.box((R*cos(a), 0, R*sin(a)), (0.5, w, 0.5), angle_z=0)`.
- Bridge deck: `B.box((0, 0, 3), (12, 3, 0.4))`; piers `B.cylinder((x, 0, 0), (x, 0, 2.8), 0.5)`;
  railing posts every 1.5 m with `B.cylinder(..., 0.05, n=8)` and a rail `B.tube([...], 0.04)`.
- Column: `B.lathe((x, y, 0), [(0, .6), (.3, .6), (.3, .4), (5.5, .35), (5.8, .55), (6.0, .6)], n=24)`.

## Fences, walls with posts, palisades
```python
def fence_ring(B, radius, posts=36, height=1.6, gap_deg=(160, 200)):
    for i in range(posts):
        a = i * 360 / posts
        if gap_deg[0] <= a <= gap_deg[1]: continue          # opening
        r = math.radians(a)
        B.cylinder((radius*math.cos(r), radius*math.sin(r), 0), (radius*math.cos(r), radius*math.sin(r), height), 0.12, mat=0, n=8)
        B.cone((radius*math.cos(r), radius*math.sin(r), height), (radius*math.cos(r), radius*math.sin(r), height + 0.3), 0.12, mat=0, n=8)
```
Straight wall of posts + rails: posts via `cylinder`, rails via `box((mid, y, 1.2), (length, 0.06, 0.12))`.

## Paths, roads, rivers (flat features on terrain)
Use a thin closed slab following a polyline: `B.tube(points, 0.15, n=6)` for a rounded path, or a
chain of thin boxes between consecutive points rotated with `angle_z=math.atan2(dy, dx)` and
`center z = terrain height + 0.05`. Water: `B.plane((0, 0, 0.2), (30, 6), thickness=0.05)` with a
glossy blue material (`roughness=0.05, base_color=(0.05, 0.25, 0.35)`).

## Query terrain height (for placing custom objects on a terrain object)
```python
from mathutils.bvhtree import BVHTree
dg = bpy.context.evaluated_depsgraph_get()
bvh = BVHTree.FromObject(ground, dg)
def ground_z(x, y):
    hit = bvh.ray_cast(ground.matrix_world.inverted() @ Vector((x, y, 1000)), Vector((0, 0, -1)))
    return (ground.matrix_world @ hit[0]).z if hit[0] else 0.0
```
`scatter(template, ..., surface=ground)` already does this.

## Trees, rocks, props as templates then scatter
```python
T = Builder(); T.cylinder((0,0,0), (0,0,1.5), 0.18, mat=0, n=8)
for k, (z, r) in enumerate([(1.2, 1.6), (2.6, 1.3), (3.9, 0.9)]):
    T.cone((0,0,z), (0,0,z + 2.2), r, 0.02, mat=1, n=10)
pine = T.build("Tree_Pine_Template", coll, [bark, needles])
trees = scatter(pine, 40, area=(-25, 25, -25, 25), seed=7, scale_range=(0.7, 1.5), min_distance=2.5, surface=ground, collection=coll)
for t in trees[:5]: log_built(t)
```
Rocks: `sphere(center, r, n=10)` then jitter vertices with noise:
```python
import mathutils
for v in rock.data.vertices:
    v.co += v.normal * 0.25 * mathutils.noise.noise(v.co * 1.7)
rock.data.update()
```
(only vertex offsets along normals keep the mesh closed).

## Randomness: always seed
`random.seed(42)` at the top makes retries reproducible; use `rng = random.Random(seed)` per feature.
