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

# Global variables for thread management
_SERVER_THREAD = None
_STOP_EVENT = threading.Event()
_ACTION_QUEUE = queue.Queue()
HOST = '127.0.0.1'
PORT = 8119 # Changed port to avoid zombie threads from previous sessions

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

def create_landscape_from_heightmap(image_path, texture_path=None):
    """Creates a landscape mesh from the provided heightmap image."""
    print(f"[ComfyUI-360] Creating Landscape from: {image_path}")
    if texture_path:
        print(f"[ComfyUI-360] Applying Texture: {texture_path}")
    
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
        
        # 3. Add Subdivision Surface Modifier
        subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf.subdivision_type = 'SIMPLE'
        subsurf.levels = 6 # Viewport
        subsurf.render_levels = 6 # Render
        
        # 4. Create Texture for Displacement
        tex_name = f"Tex_{filename}"
        tex = bpy.data.textures.get(tex_name)
        if not tex:
            tex = bpy.data.textures.new(tex_name, type='IMAGE')
        tex.image = img
        tex.extension = 'EXTEND' # Fix tiling/repeating edges
        
        # 5. Add Displace Modifier
        disp = obj.modifiers.new(name="Displace", type='DISPLACE')
        disp.texture = tex
        disp.texture_coords = 'UV' # Explicitly use UV coordinates
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
        else:
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

def server_loop(stop_event):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, PORT))
            s.listen()
            s.settimeout(1.0)
            print(f"[ComfyUI-360] Listener started on {HOST}:{PORT}")
            
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
            print(f"\n[ComfyUI-360] CRITICAL ERROR: Could not bind to port {PORT}")
            print(f"[ComfyUI-360] Error details: {e}")
            print(f"[ComfyUI-360] This usually means another instance of Blender is already running the listener.")
            print(f"[ComfyUI-360] Please close other Blender instances or stop the listener in them.\n")

def process_queue():
    """Timer function to process the action queue on the main thread."""
    if not _ACTION_QUEUE.empty():
        print(f"[ComfyUI-360] DEBUG: Processing queue item...")
        
    while not _ACTION_QUEUE.empty():
        msg = _ACTION_QUEUE.get()
        print(f"[ComfyUI-360] DEBUG: Processing message: '{msg}'")
        
        if msg.startswith("HEIGHTMAP:"):
            # Parse format: HEIGHTMAP:<path>|TEXTURE:<path>
            parts = msg.split("|TEXTURE:")
            height_path = parts[0].replace("HEIGHTMAP:", "").strip()
            texture_path = parts[1].strip() if len(parts) > 1 else None
            
            print(f"[ComfyUI-360] DEBUG: Parsed Heightmap: {height_path}")
            print(f"[ComfyUI-360] DEBUG: Parsed Texture: {texture_path}")
            
            create_landscape_from_heightmap(height_path, texture_path)
        else:
            # Default to HDRI update for backward compatibility or plain paths
            update_world_background(msg)
            
    return 1.0 # Run every 1.0 seconds

class COMFY360_OT_TestConnection(bpy.types.Operator):
    """Test the connection by printing to console"""
    bl_idname = "comfy360.test_connection"
    bl_label = "Test Connection"

    def execute(self, context):
        print("[ComfyUI-360] DEBUG: Test Connection Triggered.")
        print(f"[ComfyUI-360] DEBUG: Host={HOST}, Port={PORT}")
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
        
        _STOP_EVENT.clear()
        _SERVER_THREAD = threading.Thread(target=server_loop, args=(_STOP_EVENT,), daemon=True)
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

class COMFY360_PT_Panel(bpy.types.Panel):
    """Creates a Panel in the View3D UI"""
    bl_label = "ComfyUI 360 Sync"
    bl_idname = "VIEW3D_PT_comfy360"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "ComfyUI" # Shortened category name

    def draw(self, context):
        layout = self.layout
        is_running = context.scene.get("comfy360_is_running", False)
        
        # Check actual thread state
        thread_alive = False
        if _SERVER_THREAD and _SERVER_THREAD.is_alive():
            thread_alive = True

        box = layout.box()
        box.label(text="Connection", icon='LINKED')
        
        if is_running:
            if thread_alive:
                row = box.row()
                row.label(text=f"Status: Listening on {PORT}...", icon='REC')
                box.operator("comfy360.stop_listener", text="Stop Listener", icon='PAUSE')
                box.operator("comfy360.test_connection", text="Debug: Test Connection", icon='CONSOLE')
                
                # Show last received file
                last_file = context.scene.get("comfy360_last_file", "None")
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
            box.operator("comfy360.start_listener", text=f"Start Listener (Port {PORT})", icon='PLAY')
            box.operator("comfy360.test_connection", text="Debug: Test Connection", icon='CONSOLE')
            
        layout.separator()
        
        box2 = layout.box()
        box2.label(text="Manual Import", icon='IMPORT')
        box2.operator("comfy360.import_sky", text="Load Sky File...", icon='FILE_FOLDER')

classes = (
    COMFY360_OT_StartListener,
    COMFY360_OT_StopListener,
    COMFY360_OT_ImportSky,
    COMFY360_OT_TestConnection,
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
    bpy.types.Scene.comfy360_last_file = bpy.props.StringProperty(
        name="Last Received File",
        default="None"
    )
    print("[ComfyUI-360] Addon Registered Successfully")

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.comfy360_is_running
    del bpy.types.Scene.comfy360_last_file
    
    global _STOP_EVENT
    _STOP_EVENT.set()
    if bpy.app.timers.is_registered(process_queue):
        bpy.app.timers.unregister(process_queue)

if __name__ == "__main__":
    register()
