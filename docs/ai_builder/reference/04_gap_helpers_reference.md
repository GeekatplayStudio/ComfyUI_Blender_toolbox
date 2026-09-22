# gap_helpers API reference (blender_scripts/ai_builder/gap_helpers.py)

`from gap_helpers import *` exposes everything below plus `Vector`, `math`, `random`.

## Collections and logging
- `get_or_create_collection(name, parent=None) -> Collection` - linked to the scene, idempotent.
- `link_object(obj, collection)` - moves obj into exactly that collection.
- `log_built(obj)` - prints `BUILT <name> <vertex count>` (required for the step report).

## Builder - closed primitives accumulated into ONE mesh object
```python
B = Builder()
B.box(center, (sx, sy, sz), mat=0, angle_z=0.0)          # center-based box, rotation about Z in radians
B.plane(center, (sx, sy), mat=0, thickness=0.02)         # thin closed slab (use for floors/panels)
B.cylinder(a, b, r, mat=0, n=16, r2=None, smooth=True)   # from point a to point b; r2 = other radius (tapered)
B.cone(a, b, r1, r2=0.001, mat=0, n=16)                  # tapered cylinder ending in a tip
B.sphere(center, r, mat=0, n=24)                         # UV sphere via lathe (capped poles)
B.torus(center, R, r, mat=0, n=32, k=12, axis=(0,0,1))   # ring: R major radius, r tube radius
B.tube(points, r, mat=0, n=10)                           # pipe along a polyline [(x,y,z), ...]
B.lathe(center, [(z, radius), ...], mat=0, n=32)         # revolve a profile (bottom to top) around Z
B.prism(polygon_xz, depth, origin, mat=0, angle_z=0.0)   # extrude a 2D outline in the X/Z plane along Y
obj = B.build("Name", collection, material_or_list, angle_deg=35, merge_dist=0.0)
```
- `mat=` is the index into the material list given to `build()`.
- `build()` dissolves degenerate faces, recalculates normals OUTWARD, marks sharp edges above
  `angle_deg`, links the object and resets the builder for reuse.
- `build()` does NOT weld vertices by default (`merge_dist=0`). Primitives are already closed shells;
  welding would fuse a rivet to the hull it sits on and produce non-manifold edges.
- Overlapping/intersecting primitives are fine for the validator (each is a closed shell).

## Detail helpers - what makes a model look real
A bare cylinder reads as a cylinder. These add the fine work that makes it read as an object.
Each takes the `Builder` as its first argument and adds geometry into it.

```python
rivet_ring(B, center, radius, count, rivet_r=0.004, mat=0, protrusion=0.6, axis="Z", phase=0, n=8)
bolt(B, center, r=0.005, height=0.004, mat=0, n=6, axis="Z")       # single hex bolt head
trim_ring(B, center, radius, tube_r=0.004, mat=0, axis="Z")        # raised collar / beading / band
panel_seams(B, center, radius, z0, z1, count=6, width=0.003, depth=0.004, mat=0)
porthole(B, center, outer_r, inner_r, depth, mat_frame=0, mat_glass=1, normal="Y", rivets=12)
star_shape(B, center, outer_r, inner_r=None, points=5, depth=0.004, mat=0, normal="Y", angle=0)
crescent_shape(B, center, r, cut_offset=None, depth=0.004, mat=0, normal="Y", angle=0)
ring_positions(radius, count, z=0, phase=0, center=(0,0), axis="Z") -> [Vector, ...]
```
- `axis` / `normal` is the direction the feature faces or circles: "Z" for a ring around an upright
  hull, "Y" for a porthole or badge on a forward-facing surface.
- `panel_seams` breaks up any large smooth surface into panels - use it on hulls, towers, tanks.
- `ring_positions` gives evenly spaced points for placing your own parts (fins, legs, lamps):
```python
for p in ring_positions(0.055, 4, z=0.02, phase=math.radians(45)):
    ang = math.atan2(p.y, p.x)
    B.prism([(-0.004,0),(0.004,0),(0.004,0.075),(-0.030,0.010)], 0.010, (p.x,p.y,0), mat=1, angle_z=ang)
```

### Worked example: a detailed 0.30 m rocket ornament (validated, 0 topology errors)
```python
B = Builder()
B.lathe((0,0,0.06), [(0.0,0.070),(0.10,0.072),(0.16,0.058),(0.185,0.030)], mat=0, n=48)  # hull
B.cone((0,0,0.245),(0,0,0.295), 0.030, 0.004, mat=1, n=48)                               # nose
rivet_ring(B, (0,0,0.250), 0.029, 14, 0.0035, mat=1)          # rivets around the nose base
trim_ring(B, (0,0,0.243), 0.031, 0.0035, mat=1)               # collar under the nose
panel_seams(B, (0,0,0), 0.071, 0.07, 0.22, count=6, width=0.0025, depth=0.003, mat=1)
porthole(B, (0,-0.070,0.150), 0.030, 0.021, 0.012, mat_frame=1, mat_glass=2, normal="Y", rivets=12)
star_shape(B, (0.030,-0.064,0.196), 0.010, depth=0.003, mat=1, normal="Y")
crescent_shape(B, (0.040,-0.058,0.222), r=0.011, depth=0.003, mat=1, normal="Y")
body = B.build("Rocket_Body", coll, [steel, brass, glass]); log_built(body)
```
- Profile for `lathe`: list of `(height, radius)` from bottom to top, e.g. a vase
  `[(0, 0.4), (0.3, 0.6), (1.2, 0.35), (1.5, 0.45)]`. Radii must be > 0 (use 0.0005 for points).
- `prism` outline is counter-clockwise in X/Z: a house gable `[(-3,0),(3,0),(3,2.5),(0,4),(-3,2.5)]`
  extruded `depth=8` along Y gives a closed house body.

## Materials
- `make_material(name, base_color, metallic=0, roughness=0.5, emission_color=None, emission_strength=0,
  alpha=1, noise=None, textures=None, ior=1.45) -> Material` (idempotent by name).
  `noise=dict(scale, detail, roughness, color_a, color_b, pos_a, pos_b, bump, bump_distance)`.
  `textures=dict(base_color, roughness, metallic, normal, scale)` - image paths.
- `assign_material(obj, mat, slot=0)`.
- `set_smooth(obj, angle_deg=35)` - smooth shading with sharp edges by angle (3.6 - 5.x safe).

## Lights / camera / world
- `add_light(kind, location, energy, color, name, target=None, size=1.0, spot_angle_deg=45)`.
- `add_sun(elevation_deg, azimuth_deg, strength=3, color, name="Sun_Key", angle_deg=1)`.
- `add_camera(location, target, lens_mm=35, name, ortho=False, ortho_scale=20, make_active=True)`.
- `frame_camera_to_scene(camera, margin=1.15, direction=(1,-1.3,0.75))` - fits all render-visible meshes.
- `scene_bounds() -> (Vector min, Vector max)` of render-visible meshes.
- `set_world(color, strength=1.0, hdri_path=None)`.

## Terrain and distribution
- `terrain(name, size=(60,60), resolution=96, height=3, noise_scale=0.05, seed=1, octaves=4, collection,
  material, thickness=1.0, falloff=0.0)` - closed displaced slab; `falloff=0.3` flattens the border.
  Top surface z ranges 0..height; place objects with `scatter(surface=terrain_obj)` or query
  height yourself with a ray cast (see 05_procedural_patterns.md).
- `scatter(template, count, area=(xmin,xmax,ymin,ymax), seed, scale_range=(0.8,1.2), min_distance=1.0,
  surface=None, collection=None, rotate_z=True, name_prefix=None) -> [objects]` - linked duplicates
  dropped on the surface height; the template itself is hidden and moved away (it stays as data owner).
- `array_copies(obj, count, offset=(dx,dy,dz), collection=None) -> [objects]` - linear repeats.
- `duplicate(obj, name, location=None, collection=None, linked_data=True) -> object`.

## Removal (only when explicitly instructed)
- `delete_objects([names])` - removes objects and orphaned meshes.
- `clear_scene()` - removes everything. Triggers a safety warning; use only for "start over" requests.

## Typical sizes (meters) for believable scale
Door 0.9 x 2.1 - window 1.0 x 1.2 at sill height 0.9 - floor height 3.0 - wall thickness 0.3
(stone 0.6-1.0) - castle tower diameter 6-10, height 15-25 - pine tree 8-20 tall, trunk 0.3-0.6
- car 4.5 x 1.8 x 1.5 - person 1.75 - street width 6-8 - fence 1.2-1.8 tall, posts every 2.5.
