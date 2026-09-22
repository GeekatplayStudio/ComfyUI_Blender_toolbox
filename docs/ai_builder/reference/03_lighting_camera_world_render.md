# Lighting, camera, world, render settings

## Lights (data API)
```python
sun = add_sun(elevation_deg=35, azimuth_deg=210, strength=4.0, color=(1.0, 0.85, 0.7))   # warm low sun
key = add_light("AREA", location=(4, -6, 5), energy=800, size=2.0, name="Light_Key", target=(0, 0, 1))
fill = add_light("POINT", location=(-5, 3, 3), energy=300, color=(0.8, 0.9, 1.0), name="Light_Fill")
spot = add_light("SPOT", location=(0, 0, 8), energy=1500, spot_angle_deg=40, name="Spot_Stage", target=(0, 0, 0))
```
Manual: `data = bpy.data.lights.new("Light_Key", 'AREA'); data.energy = 800; data.size = 2;
obj = bpy.data.objects.new("Light_Key", data); coll.objects.link(obj)`.
Energy units: SUN in W/m^2 (1-5 typical, 10+ harsh noon); POINT/SPOT/AREA in Watts (100-2000 indoors,
outdoors thousands). Sun direction: rotation_euler = (radians(90-elevation), 0, radians(azimuth)).

Time of day presets: dawn/dusk elevation 5-15, warm (1.0,0.6,0.4), strength 2-3, sky (0.5,0.35,0.4);
noon elevation 60-75, (1,1,0.98), strength 5; overcast: no sun, world strength 1.5 grey (0.6,0.62,0.65);
night: moon SUN elevation 30, (0.6,0.7,1.0) strength 0.3, world (0.01,0.015,0.03), emissive lamps.

## World / sky
```python
set_world(color=(0.45, 0.6, 0.85), strength=1.0)                  # flat sky color + ambient
set_world(hdri_path=r"C:/hdri/sky.exr", strength=1.0)              # only if a path was provided
```
Sky texture node (procedural sky, Cycles/EEVEE): `sky = nodes.new("ShaderNodeTexSky"); sky.sky_type = 'NISHITA';
sky.sun_elevation = math.radians(30); sky.sun_rotation = math.radians(200); links.new(sky.outputs["Color"], bg.inputs["Color"])`.

## Camera
```python
cam = add_camera(location=(28, -32, 16), target=(0, 0, 4), lens_mm=35, name="Camera_Main")
frame_camera_to_scene(cam, margin=1.15)               # fit everything visible (skips hidden objects)
cam2 = add_camera(location=(0, -40, 6), target=(0, 0, 4), lens_mm=50, name="Camera_Front", make_active=False)
ortho = add_camera(location=(0, -50, 20), target=(0, 0, 5), name="Camera_Ortho_Front", ortho=True, ortho_scale=45)
```
`bpy.context.scene.camera = cam` sets the active render camera. Lens: 24 mm wide environment, 35 mm
natural, 50-85 mm product/portrait. Aim slightly above the ground center for architecture.

## Render settings (the runner sets these for previews; only touch when the step is about rendering)
```python
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
scene.render.engine = 'CYCLES'          # 'BLENDER_EEVEE' (5.x, <=4.1) / 'BLENDER_EEVEE_NEXT' (4.2-4.x)
scene.cycles.samples = 128; scene.cycles.use_denoising = True
scene.render.film_transparent = True
scene.view_settings.view_transform = 'AgX'     # 4.0+ ('Filmic' in 3.x)
scene.view_settings.look = 'AgX - Medium High Contrast'
```
Never call `bpy.ops.render.render` or set `scene.render.filepath` inside a build step - the runner
renders the preview after validation.

## Compositor (Blender 5.x)
5.x replaced `scene.use_nodes` with a compositor node group: `tree = bpy.data.node_groups.new("Comp", 'CompositorNodeTree');
tree.interface.new_socket(name='Image', in_out='OUTPUT', socket_type='NodeSocketColor'); scene.compositing_node_group = tree`.
4.x and earlier: `scene.use_nodes = True; tree = scene.node_tree`. Avoid compositor work unless asked.
