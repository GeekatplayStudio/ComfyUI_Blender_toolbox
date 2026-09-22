# Materials and shader nodes

## Preferred: gap_helpers.make_material
```python
stone  = make_material("Mat_Granite", (0.4, 0.4, 0.38), roughness=0.85,
                       noise=dict(scale=10, detail=6, color_a=(0.3, 0.3, 0.28), color_b=(0.5, 0.5, 0.46), bump=0.25))
metal  = make_material("Mat_Brass", (0.8, 0.6, 0.25), metallic=1.0, roughness=0.35)
glass  = make_material("Mat_Window_Glass", (0.6, 0.75, 0.85), roughness=0.05, alpha=0.25, ior=1.5)
lamp   = make_material("Mat_Lamp_Warm", (1.0, 0.7, 0.4), emission_color=(1.0, 0.7, 0.4), emission_strength=8.0)
bark   = make_material("Mat_Bark", (0.25, 0.16, 0.09), roughness=0.9)
```
Colors are linear RGB 0-1. Metals: `metallic=1.0`, dark base colors look black - use the metal's
reflectance color (gold (1.0,0.77,0.34), copper (0.95,0.64,0.54), iron (0.56,0.57,0.58), aluminium
(0.91,0.92,0.92)). Oxidized/verdigris copper: (0.2,0.45,0.35) with metallic 0.6-0.8.

## Image textures
```python
mat = make_material("Mat_Brick", textures=dict(base_color=r"C:/tex/brick_albedo.png", roughness=r"C:/tex/brick_rough.png",
                                              normal=r"C:/tex/brick_normal.png", scale=2.0))
```
Only use paths that were given in the instruction or reference brief. Image meshes need UVs: Builder
meshes have none by default - add a simple box projection:
```python
def box_uv(obj, scale=2.0):
    me = obj.data
    uv = me.uv_layers.new(name="UVMap")
    for p in me.polygons:
        n = p.normal; axis = max(range(3), key=lambda i: abs(n[i])); a, b = [i for i in range(3) if i != axis]
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uv.data[li].uv = (co[a] / scale, co[b] / scale)
```

## Principled BSDF socket names (Blender 4.0 - 5.x)
`Base Color, Metallic, Roughness, IOR, Alpha, Normal, Subsurface Weight, Specular IOR Level,
Specular Tint, Anisotropic, Transmission Weight, Coat Weight, Coat Roughness, Sheen Weight,
Emission Color, Emission Strength`.
Blender 3.x used `Specular`, `Transmission`, `Emission` (color), `Subsurface`. `gap_helpers._set_input`
tries a list of names so helper code works on both. In hand-written node code use:
```python
sock = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
```

## Manual node trees
```python
mat = bpy.data.materials.new("Mat_Custom")
if hasattr(mat, "use_nodes"): mat.use_nodes = True     # attribute deprecated in 5.x but harmless
nodes, links = mat.node_tree.nodes, mat.node_tree.links
bsdf = nodes.get("Principled BSDF"); out = nodes.get("Material Output")
tex = nodes.new("ShaderNodeTexNoise"); tex.inputs["Scale"].default_value = 4.0
ramp = nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.1, 0.1, 0.1, 1); ramp.color_ramp.elements[1].color = (0.6, 0.55, 0.5, 1)
links.new(tex.outputs["Fac"], ramp.inputs["Fac"]); links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
```
Useful nodes: `ShaderNodeTexNoise`, `ShaderNodeTexVoronoi`, `ShaderNodeTexBrick`, `ShaderNodeTexWave`,
`ShaderNodeTexMusgrave` (removed in 4.1 - use Noise), `ShaderNodeBump`, `ShaderNodeNormalMap`,
`ShaderNodeMapping`, `ShaderNodeTexCoord` (outputs Object/Generated/UV), `ShaderNodeMixRGB`
(3.x) vs `ShaderNodeMix` with `data_type='RGBA'` (4.x+), `ShaderNodeValToRGB` (color ramp).

## Assigning materials and multiple slots
```python
obj.data.materials.append(mat)               # slot 0
obj.data.materials.append(mat2)              # slot 1
for p in obj.data.polygons: p.material_index = 1 if p.center.z > 3 else 0
```
`Builder.build(name, coll, [mat0, mat1, ...])` sets slots and per-face indices from the `mat=`
argument of each primitive.

## Transparency
Set `alpha < 1`. Blender 4.2+: `mat.surface_render_method = 'BLENDED'`; older: `mat.blend_method = 'BLEND'`.
`make_material` handles both.
