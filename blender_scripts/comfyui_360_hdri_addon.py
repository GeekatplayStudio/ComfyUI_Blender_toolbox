# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

bl_info = {
    "name": "ComfyUI 360 HDRI Sync",
    "author": "ComfyUI-360-HDRI-Suite",
    "version": (1, 1, 1),
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
import shutil

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
        if use_pbr or roughness_path or normal_path or metallic_path:
            print(f"[ComfyUI-360] Generating PBR Material Nodes...")
            
            # 1. Metallic
            if metallic_path and os.path.exists(metallic_path):
                # Use provided Metallic Map
                m_filename = os.path.basename(metallic_path)
                m_img = bpy.data.images.get(m_filename)
                if m_img:
                    if m_img.filepath != metallic_path:
                        m_img.filepath = metallic_path
                    m_img.reload()
                else:
                    m_img = bpy.data.images.load(metallic_path)
                
                # Set Non-Color
                try: m_img.colorspace_settings.name = 'Non-Color'
                except: pass

                node_m_tex = nodes.new(type='ShaderNodeTexImage')
                node_m_tex.location = (-400, 400)
                node_m_tex.image = m_img
                node_m_tex.extension = 'EXTEND'
                
                links.new(node_m_tex.outputs['Color'], node_bsdf.inputs['Metallic'])
                print(f"[ComfyUI-360] Metallic Map applied.")
            
            # 2. Roughness
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
                
                # Add Color Ramp to fix "too shiny" low roughness values (User feedback)
                # Remap 0.0 (Perfectly Shiny) -> Min Roughness
                # Remap 1.0 (Matte) -> Max Roughness
                node_r_ramp = nodes.new(type='ShaderNodeValToRGB')
                node_r_ramp.location = (-200, 200)
                
                # Element 0: Black input (0.0) -> Output Min Value
                node_r_ramp.color_ramp.elements[0].position = 0.0
                node_r_ramp.color_ramp.elements[0].color = (roughness_min, roughness_min, roughness_min, 1) 
                
                # Element 1: White input (1.0) -> Output Max Value
                node_r_ramp.color_ramp.elements[1].position = 1.0
                node_r_ramp.color_ramp.elements[1].color = (roughness_max, roughness_max, roughness_max, 1)
                
                links.new(node_r_tex.outputs['Color'], node_r_ramp.inputs['Fac'])
                links.new(node_r_ramp.outputs['Color'], node_bsdf.inputs['Roughness'])
                print(f"[ComfyUI-360] Roughness Map applied with Correction Range [{roughness_min}, {roughness_max}].")
                
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
            
            # 3. Normal / Bump
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
            metallic_path = None
            alpha_path = None
            ior_path = None
            roughness_min = 0.0
            roughness_max = 1.0
            size_x = 10.0
            size_y = 10.0
            
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
                elif part.startswith("METALLIC:"):
                    metallic_path = part.replace("METALLIC:", "").strip()
                elif part.startswith("ALPHA:"):
                    alpha_path = part.replace("ALPHA:", "").strip()
                elif part.startswith("IOR:"):
                    ior_path = part.replace("IOR:", "").strip()
                elif part.startswith("ROUGHNESS_MIN:"):
                    try: roughness_min = float(part.replace("ROUGHNESS_MIN:", ""))
                    except: pass
                elif part.startswith("ROUGHNESS_MAX:"):
                    try: roughness_max = float(part.replace("ROUGHNESS_MAX:", ""))
                    except: pass
                elif part.startswith("SIZE_X:"):
                    try: size_x = float(part.replace("SIZE_X:", ""))
                    except: pass
                elif part.startswith("SIZE_Y:"):
                    try: size_y = float(part.replace("SIZE_Y:", ""))
                    except: pass
            
            print(f"[ComfyUI-360] DEBUG: Parsed Heightmap: {height_path}")
            print(f"[ComfyUI-360] DEBUG: Parsed Texture: {texture_path}")
            print(f"[ComfyUI-360] DEBUG: Use PBR: {use_pbr}")
            print(f"[ComfyUI-360] DEBUG: Roughness: {roughness_path} (Range: {roughness_min}-{roughness_max})")
            print(f"[ComfyUI-360] DEBUG: Metallic: {metallic_path}")
            print(f"[ComfyUI-360] DEBUG: Alpha: {alpha_path}")
            print(f"[ComfyUI-360] DEBUG: IOR: {ior_path}")
            print(f"[ComfyUI-360] DEBUG: Size: {size_x} x {size_y} m")
            
            # Call function to create terrain
            create_terrain_from_heightmap(height_path, texture_path, use_pbr, roughness_path, normal_path, metallic_path, alpha_path, ior_path, roughness_min, roughness_max, size_x, size_y)

        elif msg.startswith("TEXTURE_UPDATE:"):
            # Format: TEXTURE_UPDATE:ALBEDO:<path>|ROUGHNESS:<path>|...
            print(f"[ComfyUI-360] Processing Texture Update for Active Object...")
            
            parts = msg.replace("TEXTURE_UPDATE:", "").split('|')
            pbr_data = {}
            for part in parts:
                if ":" in part:
                    type_name, path = part.split(":", 1)
                    pbr_data[type_name] = path.strip()
            
            update_active_object_material(pbr_data)
            
        elif msg.startswith("MODEL:"):
            model_path = msg.replace("MODEL:", "").strip()
            print(f"[ComfyUI-360] Importing Model: {model_path}")
            
            # Import GLTF/GLB
            if os.path.exists(model_path):
                # Clear existing ComfyUI_Landscape or other imported objects if desired?
                # For now, let's keep it additive or maybe clear previous selection.
                # Or create a dedicated collection?
                
                # Let's try to just import.
                try:
                    # Note: bpy.ops.import_scene.gltf might need the addon enabled, but it's usually standard in 2.8+
                    bpy.ops.import_scene.gltf(filepath=model_path)
                    
                    # Get imported objects (they are selected by default after import)
                    imported_obs = bpy.context.selected_objects
                    if imported_obs:
                        print(f"[ComfyUI-360] Imported {len(imported_obs)} objects.")
                        # Rename main parent or group? 
                        # Optionally Frame Selected
                        for area in bpy.context.screen.areas:
                            if area.type == 'VIEW_3D':
                                for region in area.regions:
                                    if region.type == 'WINDOW':
                                        with bpy.context.temp_override(area=area, region=region):
                                            bpy.ops.view3d.view_selected(use_all_regions=False)
                                        break
                                break
                                
                except Exception as e:
                    print(f"[ComfyUI-360] Error importing GLB: {e}")
                    # Fallback for OBJ
                    if model_path.lower().endswith(".obj"):
                         try: bpy.ops.import_scene.obj(filepath=model_path)
                         except: pass
            else:
                print(f"[ComfyUI-360] Model file not found: {model_path}")
            
        else:
            # Standard image path (HDRI or Preview)
            print(f"[ComfyUI-360] Handling standard image: {msg}")
            update_world_background(msg)
            
    return 1.0 # Run every 1.0 seconds


def create_terrain_from_heightmap(height_path, texture_path=None, use_pbr=False, roughness_path=None, normal_path=None, metallic_path=None, alpha_path=None, ior_path=None, roughness_min=0.0, roughness_max=1.0, size_x=10.0, size_y=10.0):
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
            
    # Remove existing material to ensure clean state
    mat_name = "ComfyTerrainMat"
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name])

    # 2. Add High-Res Grid (fixes "segments" issue by providing pre-subdivided geometry)
    # 256 subdivisions = ~65k faces base. + Subsurf level 2 = ~1M faces render.
    # explicit 'calc_uvs=True' ensures we have a UV map for displacement
    # size=1 creates a 1x1m unit grid, which we then scale
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=256, y_subdivisions=256, size=1, location=(0, 0, 0), calc_uvs=True)
    obj = bpy.context.active_object
    obj.name = "ComfyTerrain"
    
    # Apply Scale
    obj.scale = (size_x, size_y, 1.0)
    # Apply Scale so that Displacement happens in local Z correctly (if applied in object mode)
    # Actually, keep scale on object so user can reset it? 
    # Or apply it so 1 unit of displacement = 1 meter?
    # If we scale object X/Y, Z is 1. Displacement happens in local Z.
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

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
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    # Enable Alpha Hashed for transparency if needed
    mat.blend_method = 'HASHED'
    
    # Check for shadow_method (Removed in Blender 4.2+)
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'HASHED'
    
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
    coord.location = (-1000, 0)
    mapping = nodes.new('ShaderNodeMapping')
    mapping.location = (-800, 0)
    links.new(coord.outputs['UV'], mapping.inputs['Vector'])

    # A. Texture (Color)
    if texture_path and os.path.exists(texture_path):
        tex_node = nodes.new('ShaderNodeTexImage')
        tex_node.location = (-300, 300)
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

    # C. Metallic
    if use_pbr and metallic_path and os.path.exists(metallic_path):
        m_node = nodes.new('ShaderNodeTexImage')
        m_node.location = (-300, 100)
        m_node.image = bpy.data.images.load(metallic_path)
        m_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping.outputs['Vector'], m_node.inputs['Vector'])
        links.new(m_node.outputs['Color'], bsdf.inputs['Metallic'])
        
    # D. Alpha (Opacity)
    if use_pbr and alpha_path and os.path.exists(alpha_path):
        a_node = nodes.new('ShaderNodeTexImage')
        a_node.location = (-300, -500)
        a_node.image = bpy.data.images.load(alpha_path)
        a_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping.outputs['Vector'], a_node.inputs['Vector'])
        links.new(a_node.outputs['Color'], bsdf.inputs['Alpha'])
        
    # E. IOR (Transmission)
    # Mapping IOR map to Transmission Weight probably (usually IOR is constant, but if passed as map could be transmission)
    # Or specifically IOR value? Principled BSDF has IOR input.
    if use_pbr and ior_path and os.path.exists(ior_path):
        i_node = nodes.new('ShaderNodeTexImage')
        i_node.location = (-300, -700)
        i_node.image = bpy.data.images.load(ior_path)
        i_node.image.colorspace_settings.name = 'Non-Color'
        links.new(mapping.outputs['Vector'], i_node.inputs['Vector'])
        # Linking to IOR
        links.new(i_node.outputs['Color'], bsdf.inputs['IOR'])
        # Also maybe Transmission Weight if it implies glass?
        # Let's assume if IOR is passed, we might want some Transmission? 
        # For now, just map to IOR.
        # bsdf.inputs['Transmission Weight'].default_value = 1.0 

    # F. Normal Map
    if use_pbr and normal_path and os.path.exists(normal_path):
        n_node = nodes.new('ShaderNodeTexImage')
        n_node.location = (-600, -300)
        try:
            n_node.image = bpy.data.images.load(normal_path)
            n_node.image.colorspace_settings.name = 'Non-Color'
            
            norm_map = nodes.new('ShaderNodeNormalMap')
            norm_map.location = (-300, -300)
            norm_map.inputs['Strength'].default_value = 1.0
            
            links.new(mapping.outputs['Vector'], n_node.inputs['Vector'])
            links.new(n_node.outputs['Color'], norm_map.inputs['Color'])
            links.new(norm_map.outputs['Normal'], bsdf.inputs['Normal'])
        except Exception as e:
            print(f"[ComfyUI-360] Error loading Normal Map: {e}")

    # G. Depth Map (Displacement inside Material)
    # The main displacement is done via the Modifier, but we can also add it to the material output for Cycles optimization
    if height_path and os.path.exists(height_path):
        d_node = nodes.new('ShaderNodeTexImage')
        d_node.location = (-600, -600)
        try:
            d_node.image = bpy.data.images.load(height_path)
            d_node.image.colorspace_settings.name = 'Non-Color'
            
            disp_node = nodes.new('ShaderNodeDisplacement')
            disp_node.location = (100, -200)
            disp_node.inputs['Midlevel'].default_value = 0.0
            disp_node.inputs['Scale'].default_value = 0.2 # Material displacement usually needs smaller scale than modifier
            
            links.new(mapping.outputs['Vector'], d_node.inputs['Vector'])
            links.new(d_node.outputs['Color'], disp_node.inputs['Height'])
            links.new(disp_node.outputs['Displacement'], output.inputs['Displacement'])
        except Exception as e:
            print(f"[ComfyUI-360] Error loading Displacement Map: {e}")

    # Select Object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Smooth Shading
    bpy.ops.object.shade_smooth()

def update_active_object_material(pbr_data):
    """
    Updates the active object's active material with the provided PBR textures.
    If no material exists, it creates one.
    Does NOT affect geometry.
    """
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        print(f"[ComfyUI-360] Warning: No active mesh object selected for texture update.")
        return

    print(f"[ComfyUI-360] Updating material for object: {obj.name}")
    
    # Get active material
    mat = obj.active_material
    if not mat:
        # Create new if none
        mat = bpy.data.materials.new(name=f"{obj.name}_Mat")
        mat.use_nodes = True
        obj.data.materials.append(mat)
    
    if not mat.use_nodes:
        mat.use_nodes = True
        
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Find Principled BSDF
    bsdf = None
    for n in nodes:
        if n.type == 'BSDF_PRINCIPLED':
            bsdf = n
            break
            
    if not bsdf:
        # If no BSDF, clear and create basic setup
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # ---------------------------------------------------------
    # SCALE FIX: FORCE UV -> MAPPING -> TEXTURE CHAIN
    # ---------------------------------------------------------
    
    # 1. Create centralized Coord + Mapping nodes (Fresh start for consistency)
    # We try to reuse existing matching nodes if they look like our "standard" ones to avoid clutter
    
    tex_coord_node = None
    mapping_node = None

    # Search for an existing Mapping node at (-800, 0) - rough heuristic
    for n in nodes:
        if n.type == 'MAPPING':
             mapping_node = n
             break
    
    if not mapping_node:
        mapping_node = nodes.new('ShaderNodeMapping')
        mapping_node.location = (-800, 0)
        mapping_node.label = "ComfyUI_Mapping"

    # Search for TexCoord
    for n in nodes:
        if n.type == 'TEX_COORD':
            tex_coord_node = n
            break
            
    if not tex_coord_node:
        tex_coord_node = nodes.new('ShaderNodeTexCoord')
        tex_coord_node.location = (-1000, 0)
    
    # Force Link: UV -> Mapping
    # This is critical for fixing "patchy" textures (projected weirdly)
    try:
        links.new(tex_coord_node.outputs['UV'], mapping_node.inputs['Vector'])
    except:
        pass # In case inputs are named differently in older versions

    # Helper to load and connect image with EXPLICIT vector linking
    def connect_texture(path, input_socket_name, is_non_color=False, location=(-300, 300)):
        if not path or not os.path.exists(path):
            return

        # Load Image
        try: 
            img = bpy.data.images.load(path)
            if is_non_color:
                img.colorspace_settings.name = 'Non-Color'
        except: 
            return

        # Find Logic
        tex_node = None
        
        # 1. Check if socket already has a texture
        if input_socket_name in bsdf.inputs and bsdf.inputs[input_socket_name].is_linked:
            link = bsdf.inputs[input_socket_name].links[0]
            if link.from_node.type == 'TEX_IMAGE':
                tex_node = link.from_node
        
        if not tex_node:
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.location = location
            bsdf_input = bsdf.inputs.get(input_socket_name)
            if bsdf_input:
                links.new(tex_node.outputs['Color'], bsdf_input)
        
        # CRITICAL: Force Vector Link to our Mapping Node
        try:
            links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
        except: pass

        tex_node.image = img
        print(f"[ComfyUI-360] Updated {input_socket_name} with {os.path.basename(path)}")
    
    # Connect Textures
    if "ALBEDO" in pbr_data:
        connect_texture(pbr_data["ALBEDO"], 'Base Color', is_non_color=False, location=(-300, 300))
        
    if "ROUGHNESS" in pbr_data:
        connect_texture(pbr_data["ROUGHNESS"], 'Roughness', is_non_color=True, location=(-300, 0))
        
    if "METALLIC" in pbr_data:
        connect_texture(pbr_data["METALLIC"], 'Metallic', is_non_color=True, location=(-300, -150))
        
    if "ALPHA" in pbr_data:
        connect_texture(pbr_data["ALPHA"], 'Alpha', is_non_color=True, location=(-300, -300))
        # Ensure blend mode is set
        mat.blend_method = 'HASHED'
        if hasattr(mat, 'shadow_method'): mat.shadow_method = 'HASHED'

    if "EMISSION" in pbr_data:
        # Try 'Emission Color' (4.0) first, then 'Emission'
        target_socket = 'Emission Color' if 'Emission Color' in bsdf.inputs else 'Emission'
        if target_socket in bsdf.inputs:
             connect_texture(pbr_data["EMISSION"], target_socket, is_non_color=False, location=(-300, -450))

    if "NORMAL" in pbr_data:
        # specific logic for Normal Map node
        path = pbr_data["NORMAL"]
        if path and os.path.exists(path):
            try:
                img = bpy.data.images.load(path)
                img.colorspace_settings.name = 'Non-Color'
                
                # Check for Normal Map node
                norm_node = None
                if 'Normal' in bsdf.inputs and bsdf.inputs['Normal'].is_linked:
                    link = bsdf.inputs['Normal'].links[0]
                    if link.from_node.type == 'NORMAL_MAP':
                        norm_node = link.from_node
                
                if not norm_node:
                    norm_node = nodes.new('ShaderNodeNormalMap')
                    norm_node.location = (-150, -200)
                    links.new(norm_node.outputs['Normal'], bsdf.inputs['Normal'])
                
                # Check input to Normal Map node
                tex_node = None
                if norm_node.inputs['Color'].is_linked:
                    link = norm_node.inputs['Color'].links[0]
                    if link.from_node.type == 'TEX_IMAGE':
                         tex_node = link.from_node
                
                if not tex_node:
                    tex_node = nodes.new('ShaderNodeTexImage')
                    tex_node.location = (-450, -200)
                    links.new(tex_node.outputs['Color'], norm_node.inputs['Color'])
                
                # CRITICAL: Force Vector Link
                links.new(mapping_node.outputs['Vector'], tex_node.inputs['Vector'])
                
                tex_node.image = img
                print(f"[ComfyUI-360] Updated Normal Map")
            except: pass

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

class COMFY360_OT_SendTextures(bpy.types.Operator):
    """Send active object textures to ComfyUI Buffer"""
    bl_idname = "comfy360.send_textures"
    bl_label = "Send PBR to ComfyUI"
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}
            
        # Collect materials
        materials = []
        for slot in obj.material_slots:
            if slot.material and slot.material.use_nodes:
                if slot.material not in materials:
                    materials.append(slot.material)
        
        # Fallback to active material if no slots (somehow)
        if not materials and obj.active_material and obj.active_material.use_nodes:
            materials.append(obj.active_material)
            
        if not materials:
            self.report({'ERROR'}, "No materials with nodes found")
            return {'CANCELLED'}
        
        # Get path from Preferences (Global)
        addon_name = __package__ if __package__ else __name__
        try:
            prefs = context.preferences.addons[addon_name].preferences
            base_path = prefs.comfy360_export_path
        except:
            base_path = ""

        # Deduced ComfyUI Output Path logic
        export_path = ""
        
        # 1. Direct Manual Path (High Priority)
        # If user set a path manually in preferences, use it.
        if base_path and os.path.exists(base_path):
             # Heuristic: If they pointed to ComfyUI root or output folder specifically, 
             # we might want to organize it, but if they pointed to a random folder "Textures", 
             # we should just use it.
             
             # Case A: User pointed to a generic root (contains 'custom_nodes' or 'ComfyUI' in path)
             is_root_like = "ComfyUI" in base_path or "custom_nodes" in os.listdir(base_path)
             
             if is_root_like and "output" in os.listdir(base_path):
                  export_path = os.path.join(base_path, "output", "blender")
             elif os.path.basename(base_path).lower() == "output":
                  export_path = os.path.join(base_path, "blender")
             else:
                  # Case B: Custom folder (e.g. D:\Textures) -> Use directly
                  export_path = base_path

        # 2. Try to deduce relative to this script (if no manual path or invalid)
        if not export_path:
             # Script is in .../custom_nodes/Suite/blender_scripts/addon.py
             try:
                 addon_dir = os.path.dirname(os.path.abspath(__file__))
                 # Go up to ComfyUI/
                 # .../blender_scripts/.. /.. /.. 
                 possible_root = os.path.abspath(os.path.join(addon_dir, "..", "..", ".."))
                 possible_output = os.path.join(possible_root, "output")
                 if os.path.exists(possible_output) and os.path.isdir(possible_output):
                      export_path = os.path.join(possible_output, "blender")
             except:
                 pass
        
        # 3. Fallback to Temp
        if not export_path:
             import tempfile
             export_path = os.path.join(tempfile.gettempdir(), "comfy_blender_export")
             self.report({'WARNING'}, f"Using default temp path: {export_path}")

        # Create model specific timestamped folder
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_name = "".join([c for c in obj.name if c.isalnum() or c in (' ', '_', '-')]).strip()
        final_folder_name = f"{safe_name}_{timestamp}"
        
        version_export_path = os.path.join(export_path, final_folder_name)

        if not os.path.exists(version_export_path):
             try:
                 os.makedirs(version_export_path)
             except Exception as e:
                 self.report({'ERROR'}, f"Cannot create path: {e}")
                 return {'CANCELLED'}

        # NOTE: We do NOT clean up old files anymore, as we are creating a unique folder for each export.
        
        target_map = {
            'Base Color': 'blender_albedo.png',
            'Roughness': 'blender_roughness.png',
            'Normal': 'blender_normal.png',
            'Metallic': 'blender_metallic.png',
            'Alpha': 'blender_alpha.png',
            'Emission': 'blender_emission.png',
            'Emission Color': 'blender_emission.png',
            'Specular IOR Level': 'blender_specular.png',
            'Specular': 'blender_specular.png'
        }
        
        found_any_global = False
        
        # Iterate over all found materials
        for idx, mat in enumerate(materials):
            nodes = mat.node_tree.nodes
            bsdf = None
            for n in nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    bsdf = n
                    break
            
            if not bsdf:
                continue
                
            # Prefix for this material index
            prefix = f"{idx}_"
            
            for input_name, base_filename in target_map.items():
                if input_name in bsdf.inputs:
                    socket_in = bsdf.inputs[input_name]
                    if socket_in.is_linked:
                        link = socket_in.links[0]
                        node = link.from_node
                        if node.type == 'TEX_IMAGE':
                            image = node.image
                            if image:
                                # Prepend index
                                filename = prefix + base_filename
                                save_path = os.path.join(version_export_path, filename)
                                try:
                                    copied = False
                                    if image.source == 'FILE':
                                        src = bpy.path.abspath(image.filepath)
                                        if os.path.exists(src) and os.path.isfile(src):
                                            shutil.copy(src, save_path)
                                            copied = True
                                    if not copied:
                                        prev = image.filepath_raw
                                        try:
                                            image.filepath_raw = save_path
                                            image.file_format = 'PNG'
                                            image.save()
                                        finally:
                                            image.filepath_raw = prev
                                    found_any_global = True
                                    print(f"[ComfyUI-360] Saved {filename} for material {mat.name}")
                                except Exception as e:
                                    print(f"Error saving {filename}: {e}")

        if found_any_global:
            self.report({'INFO'}, f"Textures exported to {version_export_path}")
            # Also copy to clipboard or print prominently so user can find it
            print(f"\n=======================================================")
            print(f"[ComfyUI-360] EXPORT COMPLETE")
            print(f"PATH: {version_export_path}")
            print(f"=======================================================\n")
            
            # Write "Link File" to temp for ComfyUI to discover auto-magically
            try:
                import tempfile
                link_file = os.path.join(tempfile.gettempdir(), "comfy_360_last_export.txt")
                with open(link_file, "w") as f:
                    f.write(version_export_path)
                print(f"[ComfyUI-360] Link file updated: {link_file}")
            except Exception as e:
                print(f"[ComfyUI-360] Checksum write failed: {e}")
                
        else:
            self.report({'WARNING'}, "No texture nodes found connected to Principled BSDF")

        return {'FINISHED'}

class COMFY360_OT_CleanAndRig(bpy.types.Operator):
    """Clean and Rig the active object (GUI version of clean_and_rig.py)"""
    bl_idname = "comfy360.clean_and_rig"
    bl_label = "Clean & Auto-Prep Mesh"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Props are now read from Scene to allow UI control before execution
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
             self.report({'ERROR'}, "Select a mesh first!")
             return {'CANCELLED'}
        
        # Use simple scene properties instead of operator props for the button
        scene = context.scene
        voxel_size = getattr(scene, "comfy360_clean_voxel_size", 0.05)
        decimate_ratio = getattr(scene, "comfy360_clean_decimate_ratio", 0.1)

        self.report({'INFO'}, f"Processing {obj.name} (Voxel: {voxel_size}, Decimate: {decimate_ratio})...")
        
        # 1. Remesh
        mod_remesh = obj.modifiers.new(name="AutoRemesh", type='REMESH')
        mod_remesh.mode = 'VOXEL'
        mod_remesh.voxel_size = voxel_size
        bpy.ops.object.modifier_apply(modifier="AutoRemesh")
        
        # 2. Decimate
        mod_dec = obj.modifiers.new(name="AutoDecimate", type='DECIMATE')
        mod_dec.ratio = decimate_ratio
        bpy.ops.object.modifier_apply(modifier="AutoDecimate")
        
        # 3. Smooth
        bpy.ops.object.shade_smooth()
        
        # 4. Center
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        obj.location = (0, 0, 0)
        
        self.report({'INFO'}, "Cleanup Complete!")
        return {'FINISHED'}

class COMFY360_OT_QuickRig(bpy.types.Operator):
    """Adds a basic armature and binds it to the active mesh"""
    bl_idname = "comfy360.quick_rig"
    bl_label = "Auto-Rig (Basic)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
             self.report({'ERROR'}, "Select a mesh first!")
             return {'CANCELLED'}
             
        # Center mesh first
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        obj.location = (0, 0, 0)
        
        # Calculate bounds for scaling
        min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
        max_z = max((obj.matrix_world @ v.co).z for v in obj.data.vertices)
        mesh_height = max_z - min_z
        
        # Ensure feet at 0
        obj.location.z -= min_z
        bpy.ops.object.transform_apply(location=True)

        self.report({'INFO'}, "Generating Armature...")
        
        armature = None
        # Try Rigify first
        try:
            import addon_utils
            addon_utils.enable("rigify")
            bpy.ops.object.armature_basic_human_metarig_add()
            armature = context.active_object
            armature.name = "AutoRig_Skeleton"
            
            # Simple scaling (Height / 1.75m)
            scale = mesh_height / 1.75
            armature.scale = (scale, scale, scale)
            bpy.ops.object.transform_apply(scale=True)
            self.report({'INFO'}, f"Used Rigify Basic Human (Scale: {scale:.2f})")
        except Exception:
            # Fallback
            bpy.ops.object.armature_add(enter_editmode=False, location=(0,0,0))
            armature = context.active_object
            armature.name = "AutoRig_Simple"
            armature.scale = (1, 1, mesh_height)
            bpy.ops.object.transform_apply(scale=True)
            self.report({'WARNING'}, "Rigify not found, using simple Box armature.")

        # Bind
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        armature.select_set(True)
        context.view_layer.objects.active = armature
        
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        
        self.report({'INFO'}, "Rigging Complete! (Check weights)")
        return {'FINISHED'}

class COMFY360_OT_ExportForExternal(bpy.types.Operator):
    """Prepares and exports the mesh as FBX for Mixamo/AccuRig"""
    bl_idname = "comfy360.export_external"
    bl_label = "Export for Mixamo/AccuRig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
             self.report({'ERROR'}, "Select a mesh first!")
             return {'CANCELLED'}
        
        # 1. Ensure clean and centered (Simplified inline logic)
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        obj.location = (0, 0, 0)
        
        # 2. Define path
        # Use simple temp folder or user's Documents
        import platform
        if platform.system() == "Windows":
             base_path = os.path.join(os.environ['USERPROFILE'], 'Documents')
        else:
             base_path = os.path.expanduser('~/Documents')
             
        export_path = os.path.join(base_path, "Cleaned_For_Rigging.fbx")
        
        # 3. Export FBX
        # Settings optimized for Mixamo/AccuRig (Y-Up is standard, but some prefer Z-up. FBX defaults usually work)
        try:
             bpy.ops.export_scene.fbx(
                 filepath=export_path,
                 use_selection=True,
                 axis_forward='-Z',
                 axis_up='Y',
                 # Apply Scale is crucial
                 apply_scale_options='FBX_SCALE_ALL' 
             )
             self.report({'INFO'}, f"Exported to {export_path}")
             
             # 4. Open Explorer
             if platform.system() == "Windows":
                  os.startfile(os.path.dirname(export_path))
             elif platform.system() == "Darwin":
                  subprocess.Popen(["open", os.path.dirname(export_path)])
             else:
                  subprocess.Popen(["xdg-open", os.path.dirname(export_path)])
                  
        except Exception as e:
             self.report({'ERROR'}, f"Export failed: {e}")
             return {'CANCELLED'}
             
        return {'FINISHED'}

class COMFY360_OT_SendMeshData(bpy.types.Operator):
    """Send active mesh and UV layout to ComfyUI for texture generation"""
    bl_idname = "comfy360.send_mesh_data"
    bl_label = "Send Mesh & UVs to ComfyUI"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object is not a mesh")
            return {'CANCELLED'}

        # Get path from Preferences (Global)
        addon_name = __package__ if __package__ else __name__
        try:
            prefs = context.preferences.addons[addon_name].preferences
            base_path = prefs.comfy360_export_path
        except:
            base_path = ""

        # Deduced ComfyUI Output Path logic (Duplicated for safety)
        export_path = ""
        if base_path and os.path.exists(base_path):
             is_root_like = "ComfyUI" in base_path or "custom_nodes" in os.listdir(base_path)
             if is_root_like and "output" in os.listdir(base_path):
                  export_path = os.path.join(base_path, "output", "blender")
             elif os.path.basename(base_path).lower() == "output":
                  export_path = os.path.join(base_path, "blender")
             else:
                  export_path = base_path

        if not export_path:
             try:
                 addon_dir = os.path.dirname(os.path.abspath(__file__))
                 possible_root = os.path.abspath(os.path.join(addon_dir, "..", "..", ".."))
                 possible_output = os.path.join(possible_root, "output")
                 if os.path.exists(possible_output) and os.path.isdir(possible_output):
                      export_path = os.path.join(possible_output, "blender")
             except: pass
        
        if not export_path:
             import tempfile
             export_path = os.path.join(tempfile.gettempdir(), "comfy_blender_export")

        # Create unique folder
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_name = "".join([c for c in obj.name if c.isalnum() or c in (' ', '_', '-')]).strip()
        final_folder_name = f"{safe_name}_DATA_{timestamp}"
        
        version_export_path = os.path.join(export_path, final_folder_name)
        if not os.path.exists(version_export_path):
             os.makedirs(version_export_path)

        # 1. Export Mesh (FBX)
        fbx_path = os.path.join(version_export_path, "mesh.fbx")
        try:
             # Store original selection
             original_select = obj.select_get()
             
             bpy.ops.object.select_all(action='DESELECT')
             obj.select_set(True)
             bpy.context.view_layer.objects.active = obj
             
             bpy.ops.export_scene.fbx(
                 filepath=fbx_path,
                 use_selection=True,
                 axis_forward='-Z',
                 axis_up='Y',
                 apply_scale_options='FBX_SCALE_ALL' 
             )
        except Exception as e:
             self.report({'ERROR'}, f"FBX Export failed: {e}")

        # 2. Export UV Layout
        uv_path = os.path.join(version_export_path, "uv_layout.png")
        try:
            # Must be in Edit Mode and Select All for export_layout
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            
            # Check if UV map exists
            if obj.data.uv_layers.active:
                bpy.ops.uv.export_layout(filepath=uv_path, size=(2048, 2048), opacity=0.25)
                self.report({'INFO'}, "UV Layout Exported")
            else:
                 self.report({'WARNING'}, "No UV Map found on object")
                 
            bpy.ops.object.mode_set(mode='OBJECT')
        except Exception as e:
            self.report({'ERROR'}, f"UV Export failed: {e}")
            bpy.ops.object.mode_set(mode='OBJECT')

        # 3. Create Link File (Auto-Discovery)
        import tempfile
        link_file = os.path.join(tempfile.gettempdir(), "comfyui_blender_latest.txt")
        try:
            with open(link_file, 'w') as f:
                f.write(version_export_path)
        except: pass

        # =========================================================================
        # 4. EXPORT CURRENT TEXTURES (Added functionality)
        # =========================================================================
        # We also export the Albedo/Diffuse texture if available so user has color context.
        # This fixes "exported texture is B&W/wireframe" confusion.
        
        target_map = {
            'Base Color': 'blender_albedo.png', # Primary desire
            # We can add others if needed, but Albedo is usually what they want for "Color"
        }
        
        # Helper to find BSDF
        materials = []
        if obj.active_material and obj.active_material.use_nodes:
             materials.append(obj.active_material)
        elif obj.material_slots:
             for slot in obj.material_slots:
                  if slot.material and slot.material.use_nodes:
                       materials.append(slot.material)

        import shutil
        found_albedo = False

        for mat in materials:
             if found_albedo: break # Just get the first one for "Reference" to avoid clutter? Or all?
             # Let's just get the active material's albedo
             
             nodes = mat.node_tree.nodes
             bsdf = None
             for n in nodes:
                  if n.type == 'BSDF_PRINCIPLED':
                       bsdf = n
                       break
             
             if not bsdf: continue
             
             if 'Base Color' in bsdf.inputs and bsdf.inputs['Base Color'].is_linked:
                  link = bsdf.inputs['Base Color'].links[0]
                  if link.from_node.type == 'TEX_IMAGE':
                       image = link.from_node.image
                       if image:
                            save_path = os.path.join(version_export_path, "blender_albedo.png")
                            try:
                                 copied = False
                                 if image.source == 'FILE':
                                      src = bpy.path.abspath(image.filepath)
                                      if os.path.exists(src) and os.path.isfile(src):
                                           shutil.copy(src, save_path)
                                           copied = True
                                 if not copied:
                                      image.save_render(save_path) # Try save_render for packing
                                 found_albedo = True
                                 self.report({'INFO'}, "Included Albedo Texture")
                            except Exception as e:
                                 print(f"Error saving Albedo: {e}")

        # 5. Also write a small JSON metadata file with transform info (Position/Scale)
        # This answers "read info... on texture location positioning"
        import json
        info = {
            "name": obj.name,
            "location": list(obj.location),
            "rotation": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "uv_maps": [uv.name for uv in obj.data.uv_layers]
        }
        with open(os.path.join(version_export_path, "mesh_info.json"), 'w') as f:
            json.dump(info, f, indent=2)

        self.report({'INFO'}, f"Sent Mesh, UVs & Color to {final_folder_name}")
        
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

        layout.separator()
        
        box4 = layout.box()
        box4.label(text="PBR Sync", icon='SHADING_RENDERED')
        
        # Use Preferences for global path persistence
        try:
             # Robust addon name resolution
             addon_name = __package__ if __package__ else __name__
             prefs = context.preferences.addons[addon_name].preferences
             box4.prop(prefs, "comfy360_export_path", text="Buffer Path")
        except Exception as e:
             box4.label(text="Error accessing preferences")
             print(f"[ComfyUI-360] Prefs Error: {e}")
             
        box4.operator("comfy360.send_mesh_data", text="1. Send Mesh & UVs", icon='UV_DATA')
        box4.operator("comfy360.send_textures", text="2. Send Active Textures (Back)", icon='EXPORT')


        layout.separator()
        
        box5 = layout.box()
        box5.label(text="Auto-Rigger Tools", icon='ARMATURE_DATA')
        box5.prop(scene, "comfy360_clean_voxel_size", text="Voxel Size")
        box5.prop(scene, "comfy360_clean_decimate_ratio", text="Decimate Ratio")
        box5.operator("comfy360.clean_and_rig", text="Clean Active Mesh", icon='MOD_REMESH')
        
        box5.label(text="Quick Rigging", icon='POSE_HLT')
        box5.operator("comfy360.quick_rig", text="Add Basic Humanoid Rig", icon='ARMATURE_DATA')
        box5.operator("comfy360.export_external", text="Prep for Mixamo/AccuRig", icon='FILE_3D')
        
        # Check for Voxel Heat Diffuse Skinning
        try:
            import addon_utils
            is_enabled, _ = addon_utils.check("voxel_heat_diffuse_skinning")
            if is_enabled:
                 box5.operator("object.voxel_heat_diffuse_skinning", text="Apply Voxel Skinning (Paid Addon)", icon='mod_skin')
        except: pass

class ComfyUI360Prefs(bpy.types.AddonPreferences):
    bl_idname = __package__ if __package__ else __name__

    comfy360_export_path: bpy.props.StringProperty(
        name="Texture Buffer Path",
        description="Path where textures are saved for ComfyUI to pick up (e.g. ComfyUI/input/from_blender)",
        subtype='DIR_PATH',
        default=""
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "comfy360_export_path")

classes = (
    COMFY360_OT_StartListener,
    COMFY360_OT_StopListener,
    COMFY360_OT_ImportSky,
    COMFY360_OT_TestConnection,
    COMFY360_OT_ApplyLighting,
    COMFY360_OT_SendTextures,
    COMFY360_OT_SendMeshData,
    COMFY360_OT_CleanAndRig,
    COMFY360_OT_QuickRig,
    COMFY360_OT_ExportForExternal,
    COMFY360_PT_Panel,
    ComfyUI360Prefs,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register scene property (Keep for backward compatibility/runtime vars)
    bpy.types.Scene.comfy360_export_path = bpy.props.StringProperty(
        name="Texture Buffer Path (Deprecated)",
        description="Deprecated: Use Addon Preferences",
        subtype='DIR_PATH'
    )
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
    
    # Auto-Rigger Properties
    bpy.types.Scene.comfy360_clean_voxel_size = bpy.props.FloatProperty(
        name="Voxel Size", 
        description="Smaller = More Detail, Larger = Blockier/Less Polys", 
        default=0.05, 
        min=0.001, max=0.5, step=0.01, precision=3
    )
    bpy.types.Scene.comfy360_clean_decimate_ratio = bpy.props.FloatProperty(
        name="Decimate Ratio", 
        description="1.0 = Keep all polys, 0.1 = Keep 10%", 
        default=0.1, 
        min=0.001, max=1.0, step=0.05
    )
    
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
    del bpy.types.Scene.comfy360_export_path
    
    # Auto-Rigger
    del bpy.types.Scene.comfy360_clean_voxel_size
    del bpy.types.Scene.comfy360_clean_decimate_ratio
    
    global _STOP_EVENT
    _STOP_EVENT.set()
    if bpy.app.timers.is_registered(process_queue):
        bpy.app.timers.unregister(process_queue)

if __name__ == "__main__":
    register()
