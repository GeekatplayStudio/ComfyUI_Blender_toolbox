import requests
import json
import time
import os
import folder_paths
import torch
import numpy as np
from PIL import Image
from io import BytesIO

# Configuration helper
TRIPO_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tripo_config.json")

def load_tripo_api_key():
    if os.path.exists(TRIPO_CONFIG_PATH):
        try:
            with open(TRIPO_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                return config.get("tripo_api_key", "")
        except:
            pass
    return ""

def save_tripo_api_key(key):
    config = {}
    if os.path.exists(TRIPO_CONFIG_PATH):
        try:
            with open(TRIPO_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except:
            pass
    config["tripo_api_key"] = key
    with open(TRIPO_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

def resolve_tripo_key(input_value):
    """Resolves API Key from input or config. Handles masking."""
    if input_value == "****************":
        return load_tripo_api_key()
    if input_value and input_value.strip():
        # User provided a new key, save it
        save_tripo_api_key(input_value)
        return input_value
    # Fallback to loading if empty
    return load_tripo_api_key()

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class TripoAPI:
    BASE_URL = "https://api.tripo3d.ai/v2/openapi"

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}"
        }

    def upload_image(self, image_tensor):
        """Uploads a PIL image or Tensor to Tripo and returns the image_token."""
        if isinstance(image_tensor, torch.Tensor):
            pil_image = tensor2pil(image_tensor)
        else:
            pil_image = image_tensor

        # Save to buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Try the endpoint from the documentation example
        # Example used: https://api.tripo3d.ai/v2/openapi/upload/sts
        url = f"{self.BASE_URL}/upload/sts"
        
        files = {"file": ("image.png", buffer, "image/png")}
        
        response = requests.post(url, headers=self.headers, files=files)
        
        if response.status_code != 200:
            # Fallback to upload/sts if 404 or similar, though unlikely for server-side
            print(f"Tripo Upload Failed: {response.text}. Status: {response.status_code}")
            raise Exception(f"Tripo API Error: {response.text}")
            
        data = response.json()
        if data["code"] != 0:
            raise Exception(f"Tripo API Error: {data}")
            
        return data["data"]["image_token"]

    def create_task(self, payload):
        url = f"{self.BASE_URL}/task"
        headers = self.headers.copy()
        headers["Content-Type"] = "application/json"
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Tripo Task Creation Failed: {response.text}")
            
        data = response.json()
        if data["code"] != 0:
            raise Exception(f"Tripo API Error: {data}")
            
        return data["data"]["task_id"]

    def get_task(self, task_id):
        url = f"{self.BASE_URL}/task/{task_id}"
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Tripo Get Task Failed: {response.text}")
        
        data = response.json()
        return data["data"]

    def poll_task(self, task_id, timeout=600):
        start_time = time.time()
        while time.time() - start_time < timeout:
            task_data = self.get_task(task_id)
            status = task_data["status"]
            
            if status == "success":
                return task_data
            elif status == "failed":
                raise Exception(f"Tripo Task Failed: {task_data}")
            elif status == "cancelled":
                raise Exception("Tripo Task Cancelled")
                
            time.sleep(2) # Poll every 2 seconds
            
        raise Exception("Tripo Task Timeout")

class Geekatplay_Tripo_ModelGen:
    @classmethod
    def INPUT_TYPES(s):
        # API Key is now mostly handled optionally or via connection
        return {
            "required": {
                "image_front": ("IMAGE",),
                "face_limit": ("INT", {"default": 10000, "min": 500, "max": 20000, "label": "Max Polygon Count (Detail)"}),
                "texture": ("BOOLEAN", {"default": True, "label": "Generate Texture"}),
                "pbr": ("BOOLEAN", {"default": True, "label": "PBR Materials"}),
                "quad": ("BOOLEAN", {"default": False, "label": "Quad Mesh (FBX Only)"}),
                "auto_size": ("BOOLEAN", {"default": False, "label": "Auto Size (Real World)"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "label": "Tripo API Key (Optional if Manager Connected)"}),
                "image_left": ("IMAGE",),
                "image_back": ("IMAGE",),
                "image_right": ("IMAGE",),
                "model_version": (["v2.5-20250123", "v3.0-20250812"], {"default": "v2.5-20250123"}),
                "texture_alignment": (["original_image", "geometry"], {"default": "original_image"}),
                "texture_quality": (["standard", "detailed"], {"default": "standard"}),
            }
        }


    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "task_id")
    FUNCTION = "generate_model"
    CATEGORY = "Geekatplay/Tripo3D"

    def generate_model(self, api_key, image_front, face_limit, texture, pbr, quad, auto_size,
                      image_left=None, image_back=None, image_right=None, model_version="v2.5-20250123",
                      texture_alignment="original_image", texture_quality="standard"):
        
        # Priority: 1. Input/Connection 2. Legacy config
        key_to_use = ""
        if api_key and api_key.strip():
             # Check if it is a masked value (legacy config check)
             if api_key == "****************":
                 key_to_use = load_tripo_api_key()
             else:
                 key_to_use = api_key
        else:
             # Try legacy load
             key_to_use = load_tripo_api_key()

        if not key_to_use:
            raise Exception("Tripo API Key is required. Connect an API Key Manager or enter it manually.")
            
        tripo = TripoAPI(key_to_use)
        
        # Determine mode
        # If any additional view is present, it's multiview
        is_multiview = any([image_left is not None, image_back is not None, image_right is not None])
        
        files_config = []
        
        if is_multiview:
            # Multiview expects 4 items: [front, left, back, right]
            # Omit file_token for missing views
            
            # Front (Required)
            token_front = tripo.upload_image(image_front)
            files_config.append({"type": "png", "file_token": token_front})
            
            # Left
            if image_left is not None:
                token_left = tripo.upload_image(image_left)
                files_config.append({"type": "png", "file_token": token_left})
            else:
                files_config.append({})
                
            # Back
            if image_back is not None:
                token_back = tripo.upload_image(image_back)
                files_config.append({"type": "png", "file_token": token_back})
            else:
                files_config.append({})
                
            # Right
            if image_right is not None:
                token_right = tripo.upload_image(image_right)
                files_config.append({"type": "png", "file_token": token_right})
            else:
                files_config.append({})
                
            payload = {
                "type": "multiview_to_model",
                "files": files_config,
                "model_version": model_version,
                "face_limit": face_limit,
                "texture": texture,
                "pbr": pbr,
                "quad": quad,
                "auto_size": auto_size,
                "texture_alignment": texture_alignment,
                "texture_quality": texture_quality
            }
            
        else:
            # Image to Model (Single View)
            token_front = tripo.upload_image(image_front)
            payload = {
                "type": "image_to_model",
                "file": {"type": "png", "file_token": token_front},
                "model_version": model_version,
                "face_limit": face_limit,
                "texture": texture,
                "pbr": pbr,
                "quad": quad,
                "auto_size": auto_size,
                "texture_alignment": texture_alignment,
                "texture_quality": texture_quality
            }

        # Submit Task
        task_id = tripo.create_task(payload)
        print(f"Tripo Task Submitted: {task_id}")
        
        # Poll for completion
        result = tripo.poll_task(task_id)
        
        # Download Result
        output = result.get("output", {})
        model_url = output.get("model") or output.get("pbr_model") or output.get("base_model") or output.get("original_model")

        if not model_url:
            raise Exception(f"Tripo Error: No model URL found in output. Available keys: {list(output.keys())}")
        
        # Save to Output Directory
        output_dir = folder_paths.get_output_directory()
        tripo_dir = os.path.join(output_dir, "tripo_models")
        if not os.path.exists(tripo_dir):
            os.makedirs(tripo_dir)
            
        # Determine extension
        ext = ".fbx" if quad else ".glb"
        filename = f"tripo_{task_id}{ext}"
        filepath = os.path.join(tripo_dir, filename)
        
        print(f"Downloading model to {filepath}...")
        model_resp = requests.get(model_url)
        with open(filepath, "wb") as f:
            f.write(model_resp.content)
            
        return (filepath, task_id)

class Geekatplay_Tripo_AnimateRig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_task_id": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
                "animation_preset": (["preset:idle", "preset:walk", "preset:run", "preset:jump", "preset:dance"], {"default": "preset:walk"}),
                "rig_type": (["biped", "quadruped", "auto"], {"default": "biped"}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "label": "Tripo API Key"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("rigged_model_path", "task_id")
    FUNCTION = "animate_model"
    CATEGORY = "Geekatplay/Tripo3D"

    def animate_model(self, api_key, original_task_id, animation_preset, rig_type):
        # Priority: 1. Input/Connection 2. Legacy config
        key_to_use = ""
        if api_key and api_key.strip():
             # Check if it is a masked value (legacy config check)
             if api_key == "****************":
                 key_to_use = load_tripo_api_key()
             else:
                 key_to_use = api_key
        else:
             # Try legacy load
             key_to_use = load_tripo_api_key()
            
        if not key_to_use:
            raise Exception("Tripo API Key is required. Connect an API Key Manager or enter it manually.")

        if not original_task_id:
            raise Exception("Original Task ID is required. Connect to ModelGen node.")
            
        tripo = TripoAPI(key_to_use)
        
        # Step 1: Pre-Rig Check (Optional but good for validation, but we'll skip to Rig for speed unless it fails)
        # Actually Tripo split animate into Rig + Retarget under "animate" if we use the NEW v2 API.
        
        # Wait, the docs say: "The old animate interface is split into... animate = prerigcheck + rigging + retarget"
        # So we likely need to chain them or use a wrapper?
        # Actually, let's just use `animate_rig` then `animate_retarget`.
        
        # 1. Rig
        # If rig_type is auto, we might need pre-rig check. For now assume biped or user selection.
        rig_payload = {
            "type": "animate_rig",
            "original_model_task_id": original_task_id,
            "out_format": "glb"
        }
        if rig_type != "auto":
             rig_payload["rig_type"] = rig_type

        print(f"Submitting Rig Task for {original_task_id}...")
        rig_task_id = tripo.create_task(rig_payload)
        rig_result = tripo.poll_task(rig_task_id)
        
        # Result of rig is a model. Now we Retarget (Apply Animation)
        rigged_model_task_id = rig_task_id # The result of rigging is the input for retargeting?
        # "original_model_task_id: The task_id of a rig task." -> YES.
        
        # 2. Retarget
        retarget_payload = {
            "type": "animate_retarget",
            "original_model_task_id": rigged_model_task_id,
            "out_format": "glb",
            "animation": animation_preset
        }
        
        print(f"Submitting Retarget Task (Animation: {animation_preset})...")
        anim_task_id = tripo.create_task(retarget_payload)
        anim_result = tripo.poll_task(anim_task_id)
        
        # Download Final Model
        output = anim_result.get("output", {})
        model_url = output.get("model") or output.get("pbr_model")

        if not model_url:
            raise Exception(f"Tripo Error: No animated model URL found. Available keys: {list(output.keys())}")
        
        # Save
        output_dir = folder_paths.get_output_directory()
        tripo_dir = os.path.join(output_dir, "tripo_models")
        if not os.path.exists(tripo_dir):
            os.makedirs(tripo_dir)
            
        filename = f"tripo_anim_{anim_task_id}.glb"
        filepath = os.path.join(tripo_dir, filename)
        
        print(f"Downloading animated model to {filepath}...")
        model_resp = requests.get(model_url)
        with open(filepath, "wb") as f:
            f.write(model_resp.content)
            
        return (filepath, anim_task_id)

NODE_CLASS_MAPPINGS = {
    "Geekatplay_Tripo_ModelGen": Geekatplay_Tripo_ModelGen,
    "Geekatplay_Tripo_AnimateRig": Geekatplay_Tripo_AnimateRig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Geekatplay_Tripo_ModelGen": "Tripo3D Model Generator (Geekatplay)",
    "Geekatplay_Tripo_AnimateRig": "Tripo3D Animator (Geekatplay)"
}
