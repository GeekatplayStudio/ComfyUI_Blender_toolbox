# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

bl_info = {
    "name": "ComfyUI 360 HDRI Sync",
    "author": "ComfyUI-360-HDRI-Suite",
    "version": (1, 1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > ComfyUI",
    "description": "Sync 360 HDRI skies from ComfyUI to Blender",
    "warning": "",
    "category": "Import-Export",
}

import bpy
import socket
import threading
import os
import queue
import math

# Global variables for thread management
_SERVER_THREAD = None
_STOP_EVENT = threading.Event()
_ACTION_QUEUE = queue.Queue()
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8119 # Default port

def update_world_background(image_path):
    """Updates the Blender World Background with the provided image path."""
    
    # FAILSAFE: Check if this is actually a HEIGHTMAP command that got routed here by mistake
    # (This can happen if the addon wasn't fully reloaded and the old process_queue is running)
    if image_path.startswith("HEIGHTMAP:"):
        print(f"[ComfyUI-360] FAILSAFE: Redirecting to Landscape Creation...")
        parts = image_path.split("|TEXTURE:")
        height_path = parts[0].replace("HEIGHTMAP:", "").strip()
        texture_path = parts[1].strip() if len(parts) > 1 else None
        create_landscape_from_heightmap(height_path, texture_path)
        return

    print(f"[ComfyUI-360] Loading: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"[ComfyUI-360] Error: File not found: {image_path}")
        return

    try:
        # 1. Load the image
        filename = os.path.basename(image_path)
        
        # Check if image is already loaded to avoid duplicates
        img = bpy.data.images.get(filename)
        if img:
            if img.filepath != image_path:
                img.filepath = image_path
            img.reload()
        else:
            img = bpy.data.images.load(image_path)

        # 2. Get or Create World
        world = bpy.context.scene.world
        if not world:
            world = bpy.data.worlds.new("World")
            bpy.context.scene.world = world
        
        world.use_nodes = True
        nodes = world.node_tree.nodes
        links = world.node_tree.links

        # 3. Find or Create Environment Texture Node
        env_node = None
        for node in nodes:
            if node.type == 'TEX_ENVIRONMENT':
                env_node = node
                break
        
        if not env_node:
            env_node = nodes.new('ShaderNodeTexEnvironment')
            env_node.location = (-300, 0)
        
        # Assign image
        env_node.image = img
        
        # Ensure Linear Color Space for EXR/HDR to act as proper lighting/reflection map
        if img.file_format == 'OPEN_EXR' or img.filepath.lower().endswith('.exr') or img.filepath.lower().endswith('.hdr'):
            try:
                img.colorspace_settings.name = 'Linear'
            except TypeError:
                # Fallback for some Blender versions or configs where 'Linear' might be named differently (e.g. 'Linear Rec.709')
                try:
                    img.colorspace_settings.name = 'Linear Rec.709'
                except:
                    pass # Keep default if failing
            
            print(f"[ComfyUI-360] Set colorspace to Linear for HDR image.")

        # 4. Connect to Background Node
        bg_node = None
        for node in nodes:
            if node.type == 'BACKGROUND':
                bg_node = node
                break
        
        if not bg_node:
            bg_node = nodes.new('ShaderNodeBackground')
            bg_node.location = (0, 0)

        # Link Env -> Background
        if not any(l.from_node == env_node and l.to_node == bg_node for l in links):
            links.new(env_node.outputs['Color'], bg_node.inputs['Color'])

        # 5. Connect to Output
        out_node = None
        for node in nodes:
            if node.type == 'OUTPUT_WORLD':
                out_node = node
                break
        
        if not out_node:
            out_node = nodes.new('ShaderNodeOutputWorld')
            out_node.location = (200, 0)
            
        # Link Background -> Output
        if not any(l.from_node == bg_node and l.to_node == out_node for l in links):
            links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])

        # 6. Fix Mapping (Distortion at Poles)
        # If the image is not 2:1, we can try to adjust the mapping, but usually it's better to just warn.
        # However, we can add a Mapping node to allow rotation which is often needed.
        
        mapping_node = None
        tex_coord_node = None
        
        for node in nodes:
            if node.type == 'MAPPING':
                mapping_node = node
            if node.type == 'TEX_COORD':
                tex_coord_node = node
                
        if not mapping_node:
            mapping_node = nodes.new('ShaderNodeMapping')
            mapping_node.location = (-500, 0)
            
        if not tex_coord_node:
            tex_coord_node = nodes.new('ShaderNodeTexCoord')
            tex_coord_node.location = (-700, 0)
            
        # Link Coord -> Mapping -> Env
        if not any(l.from_node == tex_coord_node and l.to_node == mapping_node for l in links):
            links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
            
        if not any(l.from_node == mapping_node and l.to_node == env_node for l in links):
            links.new(mapping_node.outputs['Vector'], env_node.inputs['Vector'])

        print(f"[ComfyUI-360] Successfully set {filename} as World Background.")
        
        # Store last received file in scene property for UI feedback
        bpy.context.scene.comfy360_last_file = filename

        # Force update view and switch to Rendered mode
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        # Switch to Rendered mode so user sees the HDRI immediately
                        if space.shading.type != 'RENDERED':
                            space.shading.type = 'RENDERED'
                area.tag_redraw()
                
    except Exception as e:
        print(f"[ComfyUI-360] Failed to update world: {e}")

def create_landscape_from_heightmap(image_path, texture_path=None, use_pbr=False, roughness_path=None, normal_path=None):
    """Creates a landscape mesh from the provided heightmap image."""
    print(f"[ComfyUI-360] Creating Landscape from: {image_path}")
    if texture_path:
        print(f"[ComfyUI-360] Applying Texture: {texture_path}")
    if use_pbr:
        print(f"[ComfyUI-360] PBR Generation Enabled")
    if roughness_path:
        print(f"[ComfyUI-360] Applying Roughness Map: {roughness_path}")
    if normal_path:
        print(f"[ComfyUI-360] Applying Normal Map: {normal_path}")
    
    if not os.path.exists(image_path):
        print(f"[ComfyUI-360] Error: File not found: {image_path}")
        return

    try:
        # 1. Load Heightmap Image
        filename = os.path.basename(image_path)
        img = bpy.data.images.get(filename)
        if img:
            if img.filepath != image_path:
                img.filepath = image_path
            img.reload()
        else:
            img = bpy.data.images.load(image_path)
            
        # 2. Create Plane
        # Delete existing landscape if it exists to avoid clutter
        if "ComfyUI_Landscape" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["ComfyUI_Landscape"], do_unlink=True)
            
        # Calculate Aspect Ratio to avoid distortion
        width = img.size[0]
        height = img.size[1]
        aspect = width / height
        
        plane_width = 10.0
        plane_height = 10.0
        
        if aspect > 1:
            plane_height = 10.0 / aspect
        else:
            plane_width = 10.0 * aspect
            
        bpy.ops.mesh.primitive_plane_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))
        obj = bpy.context.active_object
        obj.name = "ComfyUI_Landscape"
        obj.scale = (plane_width, plane_height, 1)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        
        # Ensure UVs are correct (Reset maps the single face to 0-1)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT') # Ensure face is selected
        bpy.ops.uv.reset()
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # 3. Add Subdivision Surface Modifier
        subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf.subdivision_type = 'SIMPLE'
        subsurf.levels = 6 # Viewport
        subsurf.render_levels = 6 # Render
        
        # 4. Create Texture for Displacement
        tex_name = f"Tex_{filename}"
        
        # Remove existing texture to ensure clean settings
        if tex_name in bpy.data.textures:
            bpy.data.textures.remove(bpy.data.textures[tex_name])
            
        tex = bpy.data.textures.new(tex_name, type='IMAGE')
        tex.image = img
        
        # Force CLIP to prevent tiling artifacts. 
        tex.extension = 'CLIP' 
        
        print(f"[ComfyUI-360] Displacement Texture Extension set to: {tex.extension}")
        
        # 5. Add Displace Modifier
        disp = obj.modifiers.new(name="Displace", type='DISPLACE')
        disp.texture = tex
        disp.texture_coords = 'UV' # Explicitly use UV coordinates
        
        # Explicitly set UV layer if available
        if obj.data.uv_layers:
            disp.uv_layer = obj.data.uv_layers[0].name
            
        disp.strength = 2.0 # Default strength
        disp.mid_level = 0.0 # Ground level at black
        
        # 6. Shade Smooth
        bpy.ops.object.shade_smooth()
        
        # 7. Add Material
        mat_name = "Landscape_Mat"
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.use_nodes = True
            
        # Setup Material Nodes
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        # Output Node
        node_out = nodes.new(type='ShaderNodeOutputMaterial')
        node_out.location = (400, 0)
        
        # BSDF Node
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_bsdf.location = (0, 0)
        links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
        
        # Texture Logic
        node_tex = None
        if texture_path and os.path.exists(texture_path):
            tex_filename = os.path.basename(texture_path)
            tex_img = bpy.data.images.get(tex_filename)
            if tex_img:
                if tex_img.filepath != texture_path:
                    tex_img.filepath = texture_path
                tex_img.reload()
            else:
                tex_img = bpy.data.images.load(texture_path)
                
            node_tex = nodes.new(type='ShaderNodeTexImage')
            node_tex.location = (-400, 0)
            node_tex.image = tex_img
            node_tex.extension = 'EXTEND' # Fix tiling/repeating edges
            links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])
            print(f"[ComfyUI-360] Texture applied to material.")
            
        # PBR Logic
        if use_pbr or roughness_path or normal_path:
            print(f"[ComfyUI-360] Generating PBR Material Nodes...")
            
            # 1. Roughness
            if roughness_path and os.path.exists(roughness_path):
                # Use provided Roughness Map
                r_filename = os.path.basename(roughness_path)
                r_img = bpy.data.images.get(r_filename)
                if r_img:
                    if r_img.filepath != roughness_path:
                        r_img.filepath = roughness_path
                    r_img.reload()
                else:
                    r_img = bpy.data.images.load(roughness_path)
                
                # Set Non-Color
                try:
                    r_img.colorspace_settings.name = 'Non-Color'
                except:
                    pass

                node_r_tex = nodes.new(type='ShaderNodeTexImage')
                node_r_tex.location = (-400, 200)
                node_r_tex.image = r_img
                node_r_tex.extension = 'EXTEND'
                
                links.new(node_r_tex.outputs['Color'], node_bsdf.inputs['Roughness'])
                print(f"[ComfyUI-360] Roughness Map applied.")
                
            elif node_tex:
                # Auto-generate from Texture (Invert/Ramp)
                # Texture -> ColorRamp -> Roughness
                node_ramp = nodes.new(type='ShaderNodeValToRGB')
                node_ramp.location = (-200, 200)
                # Set ramp to be somewhat rough by default (0.4 to 0.9 range)
                node_ramp.color_ramp.elements[0].position = 0.0
                node_ramp.color_ramp.elements[0].color = (0.4, 0.4, 0.4, 1)
                node_ramp.color_ramp.elements[1].position = 1.0
                node_ramp.color_ramp.elements[1].color = (0.9, 0.9, 0.9, 1)
                
                links.new(node_tex.outputs['Color'], node_ramp.inputs['Fac'])
                links.new(node_ramp.outputs['Color'], node_bsdf.inputs['Roughness'])
            
            # 2. Normal / Bump
            if normal_path and os.path.exists(normal_path):
                # Use provided Normal Map
                n_filename = os.path.basename(normal_path)
                n_img = bpy.data.images.get(n_filename)
                if n_img:
                    if n_img.filepath != normal_path:
                        n_img.filepath = normal_path
                    n_img.reload()
                else:
                    n_img = bpy.data.images.load(normal_path)
                
                # Set Non-Color
                try:
                    n_img.colorspace_settings.name = 'Non-Color'
                except:
                    pass

                node_n_tex = nodes.new(type='ShaderNodeTexImage')
                node_n_tex.location = (-400, -200)
                node_n_tex.image = n_img
                node_n_tex.extension = 'EXTEND'
                
                node_normal_map = nodes.new(type='ShaderNodeNormalMap')
                node_normal_map.location = (-200, -200)
                node_normal_map.inputs['Strength'].default_value = 1.0
                
                links.new(node_n_tex.outputs['Color'], node_normal_map.inputs['Color'])
                links.new(node_normal_map.outputs['Normal'], node_bsdf.inputs['Normal'])
                print(f"[ComfyUI-360] Normal Map applied.")
                
            elif node_tex:
                # Auto-generate from Texture (Bump)
                # Texture -> Bump -> Normal
                node_bump = nodes.new(type='ShaderNodeBump')
                node_bump.location = (-200, -200)
                node_bump.inputs['Strength'].default_value = 0.3
                node_bump.inputs['Distance'].default_value = 0.1
                
                links.new(node_tex.outputs['Color'], node_bump.inputs['Height'])
                links.new(node_bump.outputs['Normal'], node_bsdf.inputs['Normal'])
                
        if not node_tex:
            # Default color if no texture
            node_bsdf.inputs['Base Color'].default_value = (0.2, 0.8, 0.2, 1) # Greenish
            
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)
        
        print(f"[ComfyUI-360] Landscape created successfully.")
        
        # Store last received file in scene property for UI feedback
        bpy.context.scene.comfy360_last_file = f"Landscape: {filename}"
        
    except Exception as e:
        print(f"[ComfyUI-360] Failed to create landscape: {e}")

def server_loop(stop_event, host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            s.listen()
            s.settimeout(1.0)
            print(f"[ComfyUI-360] Listener started on {host}:{port}")
            
            while not stop_event.is_set():
                try:
                    conn, addr = s.accept()
                    print(f"[ComfyUI-360] Connection accepted from {addr}")
                    with conn:
                        data = conn.recv(4096)
                        if data:
                            path = data.decode('utf-8').strip()
                            print(f"[ComfyUI-360] Received path: {path}")
                            _ACTION_QUEUE.put(path)
                        else:
                            print(f"[ComfyUI-360] Received empty data")
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[ComfyUI-360] Server error: {e}")
        except OSError as e:
            print(f"\n[ComfyUI-360] CRITICAL ERROR: Could not bind to port {port}")
            print(f"[ComfyUI-360] Error details: {e}")
            print(f"[ComfyUI-360] This usually means another instance of Blender is already running the listener.")
            print(f"[ComfyUI-360] Please close other Blender instances or stop the listener in them.\n")

def update_lighting(azimuth, elevation, intensity, color_hex):
    print(f"[ComfyUI-360] Updating Lighting: Az={azimuth}, El={elevation}, Int={intensity}, Col={color_hex}")
    
    # 1. Find or Create Sun
    sun = None
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun = obj
            break
            
    if not sun:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        sun = bpy.context.active_object
        sun.name = "ComfyUI_Sun"
        
    # 2. Set Rotation
    # Blender Sun points down (-Z) by default.
    # Elevation 90 = Straight down (Zenith) -> Rot X = 0
    # Elevation 0 = Horizon -> Rot X = 90
    # Azimuth 0 = North (-Y) -> Rot Z = 0
    
    sun.rotation_euler = (math.radians(90 - elevation), 0, math.radians(azimuth))
    
    # 3. Set Intensity
    sun.data.energy = intensity
    
    # 4. Set Color
    if color_hex.startswith("#"):
        color_hex = color_hex[1:]
    
    if len(color_hex) == 6:
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        sun.data.color = (r, g, b)
        
    print("[ComfyUI-360] Lighting updated.")

def process_queue():
    """Timer function to process the action queue on the main thread."""
    if not _ACTION_QUEUE.empty():
        print(f"[ComfyUI-360] DEBUG: Processing queue item...")
        
    while not _ACTION_QUEUE.empty():
        msg = _ACTION_QUEUE.get()
        print(f"[ComfyUI-360] Processing message.")
        
        if msg.startswith("LIGHTING:"):
            parts = msg.replace("LIGHTING:", "").split('|')
            azimuth = 0.0
            elevation = 45.0
            intensity = 1.0
            color = "#FFFFFF"
            
            for part in parts:
                if part.startswith("azimuth="):
                    try: azimuth = float(part.replace("azimuth=", ""))
                    except: pass
                elif part.startswith("elevation="):
                    try: elevation = float(part.replace("elevation=", ""))
                    except: pass
                elif part.startswith("intensity="):
                    try: intensity = float(part.replace("intensity=", ""))
                    except: pass
                elif part.startswith("color="):
                    color = part.replace("color=", "")
            
            # Update UI properties
            bpy.context.scene.comfy360_light_azimuth = azimuth
            bpy.context.scene.comfy360_light_elevation = elevation
            bpy.context.scene.comfy360_light_intensity = intensity
            bpy.context.scene.comfy360_light_color = color
            
            update_lighting(azimuth, elevation, intensity, color)
            
        elif msg.startswith("HEIGHTMAP:"):
            # Parse format: HEIGHTMAP:<path>|TEXTURE:<path>|PBR:<true/false>|ROUGHNESS:<path>|NORMAL:<path>
            
            parts = msg.split('|')
            height_path = ""
            texture_path = None
            use_pbr = False
            roughness_path = None
            normal_path = None
            
            for part in parts:
                if part.startswith("HEIGHTMAP:"):
                    height_path = part.replace("HEIGHTMAP:", "").strip()
                elif part.startswith("TEXTURE:"):
                    texture_path = part.replace("TEXTURE:", "").strip()
                elif part.startswith("PBR:"):
                    pbr_val = part.replace("PBR:", "").strip().lower()
                    use_pbr = (pbr_val == "true")
                elif part.startswith("ROUGHNESS:"):
                    roughness_path = part.replace("ROUGHNESS:", "").strip()
                elif part.startswith("NORMAL:"):
                    normal_path = part.replace("NORMAL:", "").strip()
            
            print(f"[ComfyUI-360] DEBUG: Parsed Heightmap: {height_path}")
            print(f"[ComfyUI-360] DEBUG: Parsed Texture: {texture_path}")
            print(f"[ComfyUI-360] DEBUG: Use PBR: {use_pbr}")
            print(f"[ComfyUI-360] DEBUG: Roughness: {roughness_path}")
            
            # Call function to create terrain
            create_terrain_from_heightmap(height_path, texture_path, use_pbr, roughness_path, normal_path)
            
        else:
            # Standard image path (HDRI or Preview)
            print(f"[ComfyUI-360] Handling standard image: {msg}")
            update_world_background(msg)
            
    return 1.0 # Run every 1.0 seconds


def create_terrain_from_heightmap(height_path, texture_path=None, use_pbr=False, roughness_path=None, normal_path=None):
    """
    Creates a 3D terrain mesh from a heightmap image using a high-density grid.
    Applies displacement, PBR texture materials, and sets up the scene.
    Solution to "segmented" terrain by using primitive_grid_add instead of plane.
    """
    print(f"[ComfyUI-360] Creating PBR Terrain...")

    if not os.path.exists(height_path):
        print(f"[ComfyUI-360] Error: Heightmap not found: {height_path}")
        return

    # 1. Clean up old terrain
    for name in ["ComfyTermain", "ComfyTerrain"]:
        if name in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    # 2. Add High-Res Grid (fixes "segments" issue by providing pre-subdivided geometry)
    # 256 subdivisions = ~65k faces base. + Subsurf level 2 = ~1M faces render.
    # explicit 'calc_uvs=True' ensures we have a UV map for displacement
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=256, y_subdivisions=256, size=10, location=(0, 0, 0), calc_uvs=True)
    obj = bpy.context.active_object
    obj.name = "ComfyTerrain"

    # 3. Add Subdivision Surface modifier (Smoothing)
    mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    
    # Use Simple subdivision to keep grid alignment but increase density
    mod.subdivision_type = 'SIMPLE'
    # Base 256x256 * 4^1 = 512x512 resolution in viewport
    mod.levels = 1 
    # Base 256x256 * 4^2 = 1024x1024 resolution in render
    mod.render_levels = 2

    # 4. Add Displace modifier
    disp = obj.modifiers.new(name="Displacement", type='DISPLACE')
    
    # Create texture for displacement
    tex_name = "ComfyHeightTex"
    if tex_name in bpy.data.textures:
        bpy.data.textures.remove(bpy.data.textures[tex_name])
        
    tex = bpy.data.textures.new(name=tex_name, type='IMAGE')
    try:
        img = bpy.data.images.load(height_path)
        img.colorspace_settings.name = 'Non-Color' # Important for data
        tex.extension = 'EXTEND' # Prevents border artifacts
        tex.use_interpolation = True # Smooth steps
        tex.image = img
    except Exception as e:
        print(f"Error loading heightmap image: {e}")
        return

    disp.texture = tex
    disp.mid_level = 0.0 # Black is bottom
    disp.strength = 2.0 # Height scale
    
    # CRITICAL FIX: Explicitly use UV coordinates for displacement.
    # Without this, it defaults to Local coords which may map only the center fraction of the mesh (causing edge stretching).
    disp.texture_coords = 'UV'
    if obj.data.uv_layers:
        disp.uv_layer = obj.data.uv_layers[0].name

    # 5. Setup Material (PBR)
    mat_name = "ComfyTerrainMat"
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])
        
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    obj.data.materials.append(mat)
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear() # Start fresh

    # Nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Texture Mapping (Common)
    coord = nodes.new('ShaderNodeTexCoord')
    coord.location = (-800, 0)
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-600, 0)
    links.new(coord.outputs['UV'], mapping.inputs['Vector'])

    # A. Texture (Color)
    if texture_path and os.path.exists(texture_path):
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-300, 200)
        try:
            t_img = bpy.data.images.load(texture_path)
            tex_node.image = t_img
            links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
            links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
        except: pass
    else:
        # Default Green
        bsdf.inputs['Base Color'].default_value = (0.05, 0.2, 0.05, 1)

    # B. Roughness
    if use_pbr and roughness_path and os.path.exists(roughness_path):
        r_node = nodes.new('ShaderNodeTexImage')
        r_node.location = (-300, -100)
        r_node.image = bpy.data.images.load(roughness_path)
        r_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping.outputs['Vector'], r_node.inputs['Vector'])
        links.new(r_node.outputs['Color'], bsdf.inputs['Roughness'])
    else:
        bsdf.inputs['Roughness'].default_value = 0.8

    # C. Normal Map
    if use_pbr and normal_path and os.path.exists(normal_path):
        n_node = nodes.new('ShaderNodeTexImage')
        n_node.location = (-600, -300)
        n_node.image = bpy.data.images.load(normal_path)
        n_node.image.colorspace_settings.name = 'Non-Color'
        
        norm_map = nodes.new('ShaderNodeNormalMap')
        norm_map.location = (-300, -300)
        norm_map.inputs['Strength'].default_value = 1.0
        
        links.new(mapping.outputs['Vector'], n_node.inputs['Vector'])
        links.new(n_node.outputs['Color'], norm_map.inputs['Color'])
        links.new(norm_map.outputs['Normal'], bsdf.inputs['Normal'])

    # Select Object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Smooth Shading
    bpy.ops.object.shade_smooth()


class COMFY360_OT_TestConnection(bpy.types.Operator):
    """Test the connection by printing to console"""
    bl_idname = "comfy360.test_connection"
    bl_label = "Test Connection"

    def execute(self, context):
        port = context.scene.comfy360_listener_port
        host = context.scene.comfy360_listener_ip
        print("[ComfyUI-360] DEBUG: Test Connection Triggered.")
        print(f"[ComfyUI-360] DEBUG: Host={host}, Port={port}")
        if _SERVER_THREAD and _SERVER_THREAD.is_alive():
             print("[ComfyUI-360] DEBUG: Server Thread is ALIVE.")
        else:
             print("[ComfyUI-360] DEBUG: Server Thread is DEAD/STOPPED.")
        return {'FINISHED'}

class COMFY360_OT_StartListener(bpy.types.Operator):
    """Start the ComfyUI Listener"""
    bl_idname = "comfy360.start_listener"
    bl_label = "Start Listener"

    def execute(self, context):
        global _SERVER_THREAD, _STOP_EVENT
        
        if _SERVER_THREAD and _SERVER_THREAD.is_alive():
            self.report({'INFO'}, "Listener already running")
            return {'FINISHED'}
        
        port = context.scene.comfy360_listener_port
        host = context.scene.comfy360_listener_ip
        _STOP_EVENT.clear()
        _SERVER_THREAD = threading.Thread(target=server_loop, args=(_STOP_EVENT, host, port), daemon=True)
        _SERVER_THREAD.start()
        
        context.scene.comfy360_is_running = True
        
        if not bpy.app.timers.is_registered(process_queue):
            bpy.app.timers.register(process_queue)
            
        self.report({'INFO'}, "ComfyUI Listener Started")
        return {'FINISHED'}

class COMFY360_OT_StopListener(bpy.types.Operator):
    """Stop the ComfyUI Listener"""
    bl_idname = "comfy360.stop_listener"
    bl_label = "Stop Listener"

    def execute(self, context):
        global _STOP_EVENT
        _STOP_EVENT.set()
        context.scene.comfy360_is_running = False
        self.report({'INFO'}, "ComfyUI Listener Stopping...")
        return {'FINISHED'}

class COMFY360_OT_ImportSky(bpy.types.Operator):
    """Manually Import a Sky Image"""
    bl_idname = "comfy360.import_sky"
    bl_label = "Import Sky File"
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.exr;*.hdr;*.png;*.jpg", options={'HIDDEN'})

    def execute(self, context):
        update_world_background(self.filepath)
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class COMFY360_OT_ApplyLighting(bpy.types.Operator):
    """Apply the current lighting settings from the UI"""
    bl_idname = "comfy360.apply_lighting"
    bl_label = "Apply Lighting"

    def execute(self, context):
        scene = context.scene
        update_lighting(
            scene.comfy360_light_azimuth,
            scene.comfy360_light_elevation,
            scene.comfy360_light_intensity,
            scene.comfy360_light_color
        )
        return {'FINISHED'}

class COMFY360_PT_Panel(bpy.types.Panel):
    """Creates a Panel in the View3D UI"""
    bl_label = "ComfyUI 360 Sync"
    bl_idname = "VIEW3D_PT_comfy360"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ComfyUI" # Shortened category name

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        is_running = scene.get("comfy360_is_running", False)
        port = scene.get("comfy360_listener_port", DEFAULT_PORT)
        host = scene.get("comfy360_listener_ip", DEFAULT_HOST)
        
        # Check actual thread state
        thread_alive = False
        if _SERVER_THREAD and _SERVER_THREAD.is_alive():
            thread_alive = True

        box = layout.box()
        box.label(text="Connection", icon='LINKED')
        
        if is_running:
            if thread_alive:
                row = box.row()
                row.label(text=f"Status: Listening on {host}:{port}...", icon='REC')
                box.operator("comfy360.stop_listener", text="Stop Listener", icon='PAUSE')
                box.operator("comfy360.test_connection", text="Debug: Test Connection", icon='CONSOLE')
                
                # Show last received file
                last_file = scene.get("comfy360_last_file", "None")
                if last_file != "None":
                    box.label(text=f"Last: {last_file}", icon='IMAGE_DATA')
            else:
                row = box.row()
                row.alert = True
                row.label(text="Error: Listener Stopped", icon='ERROR')
                row.label(text="Check Console for details")
                box.operator("comfy360.start_listener", text="Restart Listener", icon='PLAY')
        else:
            row = box.row()
            row.label(text="Status: Stopped", icon='X')
            box.prop(scene, "comfy360_listener_ip", text="Host")
            box.prop(scene, "comfy360_listener_port", text="Port")
            box.operator("comfy360.start_listener", text=f"Start Listener", icon='PLAY')
            box.operator("comfy360.test_connection", text="Debug: Test Connection", icon='CONSOLE')
            
        layout.separator()
        
        box2 = layout.box()
        box2.label(text="Lighting Settings", icon='LIGHT')
        box2.prop(scene, "comfy360_light_azimuth", text="Azimuth")
        box2.prop(scene, "comfy360_light_elevation", text="Elevation")
        box2.prop(scene, "comfy360_light_intensity", text="Intensity")
        box2.prop(scene, "comfy360_light_color", text="Color")
        box2.operator("comfy360.apply_lighting", text="Apply Lighting", icon='LIGHT_SUN')

        layout.separator()
        
        box3 = layout.box()
        box3.label(text="Manual Import", icon='IMPORT')
        box3.operator("comfy360.import_sky", text="Load Sky File...", icon='FILE_FOLDER')

classes = (
    COMFY360_OT_StartListener,
    COMFY360_OT_StopListener,
    COMFY360_OT_ImportSky,
    COMFY360_OT_TestConnection,
    COMFY360_OT_ApplyLighting,
    COMFY360_PT_Panel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register scene property
    bpy.types.Scene.comfy360_is_running = bpy.props.BoolProperty(
        name="ComfyUI Listener Running",
        default=False
    )
    bpy.types.Scene.comfy360_listener_ip = bpy.props.StringProperty(
        name="Listener Host",
        default=DEFAULT_HOST,
        description="IP to listen on (127.0.0.1 for local, 0.0.0.0 for all interfaces)"
    )
    bpy.types.Scene.comfy360_listener_port = bpy.props.IntProperty(
        name="Listener Port",
        default=DEFAULT_PORT,
        min=1024,
        max=65535,
        description="Port to listen for ComfyUI connections"
    )
    bpy.types.Scene.comfy360_last_file = bpy.props.StringProperty(
        name="Last Received File",
        default="None"
    )
    
    # Lighting Properties
    bpy.types.Scene.comfy360_light_azimuth = bpy.props.FloatProperty(name="Azimuth", default=0.0, min=0.0, max=360.0)
    bpy.types.Scene.comfy360_light_elevation = bpy.props.FloatProperty(name="Elevation", default=45.0, min=0.0, max=90.0)
    bpy.types.Scene.comfy360_light_intensity = bpy.props.FloatProperty(name="Intensity", default=1.0, min=0.0, max=10.0)
    bpy.types.Scene.comfy360_light_color = bpy.props.StringProperty(name="Color", default="#FFFFFF")
    
    print("[ComfyUI-360] Addon Registered Successfully")

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.comfy360_is_running
    del bpy.types.Scene.comfy360_listener_ip
    del bpy.types.Scene.comfy360_listener_port
    del bpy.types.Scene.comfy360_last_file
    del bpy.types.Scene.comfy360_light_azimuth
    del bpy.types.Scene.comfy360_light_elevation
    del bpy.types.Scene.comfy360_light_intensity
    del bpy.types.Scene.comfy360_light_color
    
    global _STOP_EVENT
    _STOP_EVENT.set()
    if bpy.app.timers.is_registered(process_queue):
        bpy.app.timers.unregister(process_queue)

if __name__ == "__main__":
    register()
