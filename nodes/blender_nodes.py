import socket
import os
import folder_paths
import numpy as np
import imageio
import torch

class SendToBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ), # Pass through
                "ui_images": ("IMAGE", ), # From SaveFakeHDRI output
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "send"
    OUTPUT_NODE = True
    CATEGORY = "360_HDRI"

    def send(self, images, ui_images, prompt=None, extra_pnginfo=None):
        # The SaveFakeHDRI node returns a dictionary with "ui": {"images": [...]}
        # But here we need the actual file path.
        # Since standard ComfyUI nodes don't pass file paths easily between nodes,
        # we have to rely on the fact that the user just saved it.
        
        # However, to make this robust, we should probably modify SaveFakeHDRI to return the path,
        # OR we can just reconstruct the path if we know the settings.
        
        # BETTER APPROACH:
        # Let's modify SaveFakeHDRI to return the full path as a string.
        # But since I can't easily change the previous node's return type without breaking compatibility 
        # (though I just wrote it, so I can), let's update SaveFakeHDRI to return (IMAGE, STRING).
        
        # For now, assuming we update SaveFakeHDRI, let's write this node to accept a STRING path.
        pass

# Re-writing the class to accept a path string
class PreviewInBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_to_blender"
    OUTPUT_NODE = True
    CATEGORY = "360_HDRI"

    def send_to_blender(self, file_path):
        host = '127.0.0.1'
        port = 8119 # Changed port to avoid zombie threads from previous sessions
        
        print(f"Sending {file_path} to Blender...")
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                print(f"DEBUG: Connecting to {host}:{port}...")
                s.connect((host, port))
                print(f"DEBUG: Connected. Sending data...")
                s.sendall(file_path.encode('utf-8'))
                print(f"Successfully sent path to Blender: {file_path}")
        except ConnectionRefusedError:
            print(f"CONNECTION ERROR: Could not connect to Blender at {host}:{port}.")
            print("1. Make sure Blender is running.")
            print("2. Make sure the 'ComfyUI 360 HDRI' addon is enabled.")
            print("3. Click 'Start Listener' in the ComfyUI tab in Blender's N-Panel.")
        except Exception as e:
            print(f"Error sending to Blender: {e}")

        return {}

class PreviewHeightmapInBlender:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
            "optional": {
                "texture": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_heightmap"
    OUTPUT_NODE = True
    CATEGORY = "360_HDRI"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def send_heightmap(self, images, texture=None):
        # Save to temp dir
        output_dir = folder_paths.get_temp_directory()
        filename = "ComfyUI_Heightmap_Temp.png"
        filepath = os.path.join(output_dir, filename)
        
        # Convert tensor to numpy
        img_np = images[0].cpu().numpy() # [H, W, C]
        
        # Convert to 0-255 uint8
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        
        imageio.imwrite(filepath, img_np)
        
        # Handle Texture
        texture_filepath = ""
        if texture is not None:
            texture_filename = "ComfyUI_Texture_Temp.png"
            texture_filepath = os.path.join(output_dir, texture_filename)
            
            tex_np = texture[0].cpu().numpy()
            tex_np = (np.clip(tex_np, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(texture_filepath, tex_np)

        # Send to Blender
        host = '127.0.0.1'
        port = 8119 # Changed port to avoid zombie threads from previous sessions
        
        # Message format: HEIGHTMAP:<height_path>|TEXTURE:<texture_path>
        message = f"HEIGHTMAP:{filepath}"
        if texture_filepath:
            message += f"|TEXTURE:{texture_filepath}"
        
        print(f"DEBUG: [PreviewHeightmapInBlender] Heightmap saved to: {filepath}")
        if texture_filepath:
            print(f"DEBUG: [PreviewHeightmapInBlender] Texture saved to: {texture_filepath}")
        
        print(f"DEBUG: [PreviewHeightmapInBlender] Connecting to {host}:{port}...")
        print(f"DEBUG: [PreviewHeightmapInBlender] Sending message: '{message}'")
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5) # Increased timeout
                s.connect((host, port))
                s.sendall(message.encode('utf-8'))
                print(f"DEBUG: [PreviewHeightmapInBlender] Successfully sent data to Blender.")
        except ConnectionRefusedError:
            print(f"CONNECTION ERROR: Could not connect to Blender at {host}:{port}.")
            print("DEBUG: Is the Blender listener running? Check the Blender console.")
        except Exception as e:
            print(f"Error sending to Blender: {e}")
            import traceback
            traceback.print_exc()

        return {}
