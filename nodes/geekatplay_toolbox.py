import torch
import comfy.utils
import comfy.model_management
import math
import time
import sys
import os
import folder_paths
import random

# Define wildcard type since comfy.utils.ANY is not always available
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_type = AnyType("*")

# For Pauser api
from server import PromptServer
from aiohttp import web

# Global registry for Pauser nodes
WAITING_NODES = {}

def setup_pauser_routes():
    try:
        routes = PromptServer.instance.routes
        # Check if route already exists to avoid dupes on reload
        for route in routes:
            if route.method == "POST" and route.path == "/geekatplay/continue":
                return
        
        async def continue_node(request):
            data = await request.json()
            node_id = data.get("node_id")
            if node_id in WAITING_NODES:
                WAITING_NODES[node_id].set() # Signal event
                del WAITING_NODES[node_id]
                return web.Response(text="Continued")
            return web.Response(text="Node not waiting", status=404)

        PromptServer.instance.app.router.add_post("/geekatplay/continue", continue_node)
    except Exception as e:
        print(f"[Geekatplay Toolbox] Error setting up routes: {e}")

setup_pauser_routes()

class GapSmartResizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_target": (["SD 1.5", "SDXL", "Flux", "Custom"], {"default": "SDXL"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9", "Original"], {"default": "1:1"}),
                "processing_method": (["Scale (Lanczos)", "Stretch", "Center Crop", "Pad"], {"default": "Scale (Lanczos)"}),
            },
            "optional": {
                "custom_width": ("INT", {"default": 1024, "min": 64}),
                "custom_height": ("INT", {"default": 1024, "min": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "Geekatplay Studio/3D Toolbox"

    def resize(self, image, model_target, aspect_ratio, processing_method, custom_width=1024, custom_height=1024):
        # 1. Determine Target Pixel Count
        target_pixels = 0
        if model_target == "SD 1.5":
            target_pixels = 512 * 512 # 262144
        elif model_target == "SDXL":
            target_pixels = 1024 * 1024 # 1048576
        elif model_target == "Flux":
            target_pixels = 1024 * 1024 * 2 # approx 2MP? PRD says 2,000,000+
            # Flux is flexible, but ~1MP to 2MP is good. Let's use 2MP for now as per PRD.
            target_pixels = 2000000 
        else:
            # Custom uses input width/height directly, so logic allows skipping
            pass

        # 2. Determine Ratio
        w_in, h_in = image.shape[2], image.shape[1]
        ratio = 1.0
        
        if aspect_ratio == "Original":
            ratio = w_in / h_in
        elif aspect_ratio == "1:1": ratio = 1.0
        elif aspect_ratio == "16:9": ratio = 16/9
        elif aspect_ratio == "9:16": ratio = 9/16
        elif aspect_ratio == "4:3": ratio = 4/3
        elif aspect_ratio == "3:4": ratio = 3/4
        elif aspect_ratio == "3:2": ratio = 3/2
        elif aspect_ratio == "2:3": ratio = 2/3
        elif aspect_ratio == "21:9": ratio = 21/9
        
        # 3. Calculate Target dims
        if model_target == "Custom":
            target_w = custom_width
            target_h = custom_height
        else:
            # H = sqrt(Pixels / Ratio)
            # W = H * Ratio
            target_h = math.sqrt(target_pixels / ratio)
            target_w = target_h * ratio
        
        # Round to 8
        target_w = int(round(target_w / 8) * 8)
        target_h = int(round(target_h / 8) * 8)
        
        # 4. Process
        output = image
        
        if processing_method == "Scale (Lanczos)":
            output = comfy.utils.common_upscale(image.permute(0,3,1,2), target_w, target_h, "lanczos", "disabled").permute(0,2,3,1)
            
        elif processing_method == "Stretch":
            # Bicubic just to be safe, but "Stretch" implies direct resize without keeping aspect? 
            # commons_upscale does that if crop is disabled.
            output = comfy.utils.common_upscale(image.permute(0,3,1,2), target_w, target_h, "bicubic", "disabled").permute(0,2,3,1)

        elif processing_method == "Center Crop":
             # Provide logic to center crop to target dims
             # First scale minimal side to cover target
             scale_w = target_w / w_in
             scale_h = target_h / h_in
             scale = max(scale_w, scale_h)
             
             temp_w = int(w_in * scale)
             temp_h = int(h_in * scale)
             
             resized = comfy.utils.common_upscale(image.permute(0,3,1,2), temp_w, temp_h, "lanczos", "disabled").permute(0,2,3,1)
             
             # Center crop
             start_x = (temp_w - target_w) // 2
             start_y = (temp_h - target_h) // 2
             output = resized[:, start_y:start_y+target_h, start_x:start_x+target_w, :]

        elif processing_method == "Pad":
             # Scale to fit inside
             scale_w = target_w / w_in
             scale_h = target_h / h_in
             scale = min(scale_w, scale_h)
             
             temp_w = int(w_in * scale)
             temp_h = int(h_in * scale)
             
             resized = comfy.utils.common_upscale(image.permute(0,3,1,2), temp_w, temp_h, "lanczos", "disabled").permute(0,2,3,1)
             
             # Create black canvas
             canvas = torch.zeros((image.shape[0], target_h, target_w, 3), dtype=torch.float32)
             
             start_x = (target_w - temp_w) // 2
             start_y = (target_h - temp_h) // 2
             
             canvas[:, start_y:start_y+temp_h, start_x:start_x+temp_w, :] = resized
             output = canvas

        return (output, target_w, target_h)

class GapStringViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    CATEGORY = "Geekatplay Studio/3D Toolbox"
    OUTPUT_NODE = True 

    def show(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}

class GapPauser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": (any_type,), 
            },
             "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "pause"
    CATEGORY = "Geekatplay Studio/3D Toolbox"
    
    def pause(self, any_input, unique_id):
        # Create event
        import threading
        event = threading.Event()
        WAITING_NODES[unique_id] = event
        
        print(f"[GapPauser] Node {unique_id} waiting...")
        event.wait() # Blocks execution
        print(f"[GapPauser] Node {unique_id} continuing...")
        
        return (any_input,)

class GapLogicSwitch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_a": (any_type,),
                "input_b": (any_type,),
                "boolean_switch": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("selected_output",)
    FUNCTION = "switch"
    CATEGORY = "Geekatplay Studio/3D Toolbox"
    
    def switch(self, input_a, input_b, boolean_switch):
        if boolean_switch:
            return (input_a,)
        else:
            return (input_b,)

class GapGroupManager:
    # Requires JS to be useful, but logic needed here?
    # Actually, mostly front-end. The backend node might just be a dummy to hold the widget.
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }
    
    RETURN_TYPES = ()
    FUNCTION = "nop"
    CATEGORY = "Geekatplay Studio/3D Toolbox"
    
    def nop(self):
        return ()

class GapVRAMPurge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_in": (any_type,),
            }
        }
    
    RETURN_TYPES = (any_type,)
    FUNCTION = "purge"
    CATEGORY = "Geekatplay Studio/3D Toolbox"
    
    def purge(self, any_in):
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        return (any_in,)

class GapVisualComparator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image_a",)
    FUNCTION = "compare"
    CATEGORY = "Geekatplay Studio/3D Toolbox"
    OUTPUT_NODE = True # Important for UI to receive images

    def compare(self, image_a, image_b):
        import numpy as np
        from PIL import Image

        results = []
        for img_tensor in [image_a, image_b]:
            # Take first from batch
            t = img_tensor[0]
            i = 255. * t.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Save to temp
            filename = f"gap_compare_{random.randint(1, 100000000)}.png"
            subfolder = "gap_temp"
            full_output_folder = os.path.join(folder_paths.get_temp_directory(), subfolder)
            
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)
            
            img.save(os.path.join(full_output_folder, filename))
            
            results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp"
            })

        return {"ui": {"comparison_data": results}, "result": (image_a,)}
