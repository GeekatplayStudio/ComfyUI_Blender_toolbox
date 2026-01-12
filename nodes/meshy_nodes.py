import requests
import json
import time
import os
import folder_paths
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def image_to_data_uri(image_tensor):
    pil_image = tensor2pil(image_tensor)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def download_meshy_model(result, task_id):
    # Determine model URL (GLB preferred)
    model_urls = result.get("model_urls", {})
    model_url = model_urls.get("glb")
    # Fallback to fbx or obj if glb not available
    if not model_url:
        model_url = model_urls.get("fbx") or model_urls.get("obj")

    if not model_url:
        raise Exception(f"Meshy Error: No supported model URL (GLB/FBX/OBJ) found. Available keys: {list(model_urls.keys())}")

    output_dir = folder_paths.get_output_directory()
    meshy_dir = os.path.join(output_dir, "meshy_models")
    if not os.path.exists(meshy_dir):
        os.makedirs(meshy_dir)
    
    ext = model_url.split('.')[-1].split('?')[0] # simplistic extension extraction
    if len(ext) > 4: ext = "glb" # default if parsing fails

    filename = f"meshy_{task_id}.{ext}"
    filepath = os.path.join(meshy_dir, filename)
    
    print(f"Downloading model to {filepath}...")
    resp = requests.get(model_url)
    with open(filepath, "wb") as f:
        f.write(resp.content)
        
    return (filepath, task_id)

class MeshyAPI:
    BASE_URL = "https://api.meshy.ai"

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def create_text_to_3d_task(self, prompt, art_style, negative_prompt, mode="preview"):
        url = f"{self.BASE_URL}/v2/text-to-3d"
        payload = {
            "mode": mode,
            "prompt": prompt,
            "art_style": art_style,
            "negative_prompt": negative_prompt
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        self._check_error(response)
        return response.json()["result"]

    def create_image_to_3d_task(self, image_uri, enable_pbr=True):
        url = f"{self.BASE_URL}/v1/image-to-3d"
        payload = {
            "image_url": image_uri,
            "enable_pbr": enable_pbr
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        self._check_error(response)
        return response.json()["result"]

    def get_task(self, task_id, task_type="text-to-3d"):
        # task_type: 'text-to-3d' (v2) or 'image-to-3d' (v1)
        version = "v2" if task_type == "text-to-3d" else "v1"
        url = f"{self.BASE_URL}/{version}/{task_type}/{task_id}"
        
        response = requests.get(url, headers=self.headers)
        self._check_error(response)
        return response.json()["result"]

    def poll_task(self, task_id, task_type="text-to-3d", timeout=600):
        start_time = time.time()
        while time.time() - start_time < timeout:
            task_data = self.get_task(task_id, task_type)
            status = task_data["status"]
            
            if status == "SUCCEEDED":
                return task_data
            elif status in ["FAILED", "EXPIRED"]:
                raise Exception(f"Meshy Task Failed: {task_data.get('task_error', 'Unknown Error')}")
                
            time.sleep(2) # Poll every 2 seconds
            
        raise Exception("Meshy Task Timeout")

    def _check_error(self, response):
        if response.status_code != 200 and response.status_code != 202:
            try:
                err = response.json()
                msg = err.get("message", response.text)
            except:
                msg = response.text
            raise Exception(f"Meshy API Error ({response.status_code}): {msg}")

class Geekatplay_Meshy_TextTo3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a futuristic sci-fi helmet"}),
                "mode": (["preview", "refine"], {"default": "preview"}),
                "art_style": (["realistic", "cartoon", "low-poly", "voxel"], {"default": "realistic"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "ugly, blurry, low quality"}),
                "api_key": ("STRING", {"multiline": False, "default": "", "label": "Meshy API Key"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Geekatplay/Meshy"

    def generate(self, prompt, mode, art_style, negative_prompt, api_key):
        if not api_key:
            raise Exception("Meshy API Key is required. Connect the Geekatplay API Key Manager.")
            
        meshy = MeshyAPI(api_key)
        
        print(f"Submitting Meshy Text-to-3D Task: {prompt[:30]}...")
        task_id = meshy.create_text_to_3d_task(prompt, art_style, negative_prompt, mode)
        print(f"Task ID: {task_id}")
        
        result = meshy.poll_task(task_id, "text-to-3d")
        
        return download_meshy_model(result, task_id)

class Geekatplay_Meshy_ImageTo3D:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_pbr": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": "", "label": "Meshy API Key"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Geekatplay/Meshy"

    def generate(self, image, enable_pbr, api_key):
        if not api_key:
            raise Exception("Meshy API Key is required.")
            
        meshy = MeshyAPI(api_key)
        
        # Convert image to data URI (Meshy supports image_url as data uri in v1 usually, giving it a try)
        # Note: If Meshy demands a public URL, this might fail without a cloud uploader.
        # However, for this implementation request, we attempt data URI.
        image_uri = image_to_data_uri(image)
        
        print(f"Submitting Meshy Image-to-3D Task...")
        task_id = meshy.create_image_to_3d_task(image_uri, enable_pbr)
        print(f"Task ID: {task_id}")
        
        result = meshy.poll_task(task_id, "image-to-3d")
        
        return download_meshy_model(result, task_id)

NODE_CLASS_MAPPINGS = {
    "Geekatplay_Meshy_TextTo3D": Geekatplay_Meshy_TextTo3D,
    "Geekatplay_Meshy_ImageTo3D": Geekatplay_Meshy_ImageTo3D
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Geekatplay_Meshy_TextTo3D": "Meshy Text to 3D (Geekatplay)",
    "Geekatplay_Meshy_ImageTo3D": "Meshy Image to 3D (Geekatplay)"
}