# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox

import bpy
import socket
import threading
import os

# Configuration
HOST = '127.0.0.1'
PORT = 8118

def update_world_background(image_path):
    """Updates the Blender World Background with the provided image path."""
    print(f"Received request to load: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    # Ensure we are in the main thread for Blender API calls
    # (Blender 2.8+ is thread-safe for some things, but it's safer to use a timer or queue if this crashes. 
    # For simple cases, direct call often works if triggered carefully, but let's be safe).
    
    # We will schedule the update to run on the main thread
    bpy.app.timers.register(lambda: _apply_image(image_path))

def _apply_image(image_path):
    try:
        # 1. Load the image
        filename = os.path.basename(image_path)
        
        # Check if image is already loaded to avoid duplicates
        img = bpy.data.images.get(filename)
        if img:
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
        links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])

        print(f"Successfully set {filename} as World Background.")
    except Exception as e:
        print(f"Failed to update world: {e}")
    
    return None # Unregister timer

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"Listening for ComfyUI on {HOST}:{PORT}...")

    while True:
        client, addr = server.accept()
        try:
            data = client.recv(4096).decode('utf-8')
            if data:
                update_world_background(data.strip())
        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client.close()

# Run server in a separate thread so it doesn't block Blender UI
thread = threading.Thread(target=start_server, daemon=True)
thread.start()
print("ComfyUI-Blender-Toolbox Listener Started.")
