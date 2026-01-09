# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

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
                "blender_ip_address": ("STRING", {"default": "127.0.0.1"}),
                "blender_listen_port": ("INT", {"default": 8119, "min": 1024, "max": 65535, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_to_blender"
    OUTPUT_NODE = True
    CATEGORY = "360_HDRI"

    def send_to_blender(self, file_path, blender_ip_address="127.0.0.1", blender_listen_port=8119):
        host = blender_ip_address
        port = blender_listen_port
        
        print(f"Sending {file_path} to Blender...")
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect((host, port))
                s.sendall(file_path.encode('utf-8'))
                print(f"Successfully sent path to Blender: {file_path}")
        except (ConnectionRefusedError, OSError) as e:
            # Check for WinError 10061 specifically if it's an OSError
            if isinstance(e, ConnectionRefusedError) or (hasattr(e, 'winerror') and e.winerror == 10061):
                print(f"CONNECTION ERROR: Could not connect to Blender at {host}:{port}.")
                print("1. Make sure Blender is running.")
                print("2. Make sure the 'ComfyUI 360 HDRI' addon is enabled.")
                print("3. Click 'Start Listener' in the ComfyUI tab in Blender's N-Panel.")
            else:
                print(f"Error sending to Blender: {e}")
        except Exception as e:
            print(f"Uncaught Error sending to Blender: {e}")

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
                "generate_pbr": ("BOOLEAN", {"default": True}),
                "use_texture_as_heightmap": ("BOOLEAN", {"default": False}),
                "blender_ip_address": ("STRING", {"default": "127.0.0.1"}),
                "blender_listen_port": ("INT", {"default": 8119, "min": 1024, "max": 65535, "step": 1}),
            },
            "optional": {
                "texture": ("IMAGE", ),
                "roughness_map": ("IMAGE", ),
                "normal_map": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_heightmap"
    OUTPUT_NODE = True
    CATEGORY = "360_HDRI"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def send_heightmap(self, images, generate_pbr=True, use_texture_as_heightmap=False, texture=None, roughness_map=None, normal_map=None, blender_ip_address="127.0.0.1", blender_listen_port=8119):
        # Save to temp dir
        output_dir = folder_paths.get_temp_directory()
        filename = "ComfyUI_Heightmap_Temp.png"
        filepath = os.path.join(output_dir, filename)
        
        # Decide source for heightmap
        if use_texture_as_heightmap and texture is not None:
            # Use texture as heightmap (Grayscale conversion)
            img_np = texture[0].cpu().numpy()
            # RGB to Grayscale using standard luminance weights
            # Y = 0.299 R + 0.587 G + 0.114 B
            if img_np.shape[2] == 3:
                img_gray = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
                # Expand back to 3 channels for simple saving or keep as single
                img_np = np.stack((img_gray,)*3, axis=-1)
            elif img_np.shape[2] == 4: # Handle RGBA
                img_gray = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
                img_alpha = img_np[..., 3]
                img_np = np.stack((img_gray, img_gray, img_gray, img_alpha), axis=-1)
        else:
            # Use provided heightmap images
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

        # Handle Roughness
        roughness_filepath = ""
        if roughness_map is not None:
            roughness_filename = "ComfyUI_Roughness_Temp.png"
            roughness_filepath = os.path.join(output_dir, roughness_filename)
            
            r_np = roughness_map[0].cpu().numpy()
            r_np = (np.clip(r_np, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(roughness_filepath, r_np)

        # Handle Normal
        normal_filepath = ""
        if normal_map is not None:
            normal_filename = "ComfyUI_Normal_Temp.png"
            normal_filepath = os.path.join(output_dir, normal_filename)
            
            n_np = normal_map[0].cpu().numpy()
            n_np = (np.clip(n_np, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(normal_filepath, n_np)

        # Send to Blender
        host = blender_ip_address
        port = blender_listen_port
        
        # Message format: HEIGHTMAP:<height_path>|TEXTURE:<texture_path>|PBR:<true/false>|ROUGHNESS:<path>|NORMAL:<path>
        message = f"HEIGHTMAP:{filepath}"
        if texture_filepath:
            message += f"|TEXTURE:{texture_filepath}"
        
        if generate_pbr:
            message += "|PBR:true"
        else:
            message += "|PBR:false"
            
        if roughness_filepath:
            message += f"|ROUGHNESS:{roughness_filepath}"
        if normal_filepath:
            message += f"|NORMAL:{normal_filepath}"
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5) # Increased timeout
                s.connect((host, port))
                s.sendall(message.encode('utf-8'))
        except ConnectionRefusedError:
            print(f"CONNECTION ERROR: Could not connect to Blender at {host}:{port}.")
            print("Is the Blender listener running? Check the Blender console.")
        except Exception as e:
            print(f"Error sending to Blender: {e}")
            import traceback
            traceback.print_exc()

        return {}

class SyncLightingToBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "azimuth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0}),
                "elevation": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 90.0}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "color_hex": ("STRING", {"default": "#FFFFFF"}),
                "blender_ip_address": ("STRING", {"default": "127.0.0.1"}),
                "blender_listen_port": ("INT", {"default": 8119, "min": 1024, "max": 65535, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "sync_lighting"
    OUTPUT_NODE = True
    CATEGORY = "360_HDRI"

    def sync_lighting(self, azimuth, elevation, intensity, color_hex, blender_ip_address="127.0.0.1", blender_listen_port=8119):
        host = blender_ip_address
        port = blender_listen_port
        
        # Message format: LIGHTING:azimuth=<val>|elevation=<val>|intensity=<val>|color=<hex>
        message = f"LIGHTING:azimuth={azimuth}|elevation={elevation}|intensity={intensity}|color={color_hex}"
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((host, port))
                s.sendall(message.encode('utf-8'))
        except Exception as e:
            print(f"Error sending to Blender: {e}")
            
        return {}
