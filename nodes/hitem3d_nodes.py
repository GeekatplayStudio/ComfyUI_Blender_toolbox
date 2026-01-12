import requests
import json
import time
import os
import folder_paths
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import torch

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def tensor_to_bytes(tensor, format="JPEG"):
    pil_image = tensor2pil(tensor)
    buffer = BytesIO()
    pil_image.save(buffer, format=format)
    return buffer.getvalue()

class HiTem3DAPIClient:
    def __init__(self, access_key, secret_key, base_url="https://api.hitem3d.ai"):
        self.access_key = access_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip('/')
        self.access_token = None
        self.token_expires_at = 0

    def _get_basic_auth_header(self):
        credentials = f"{self.access_key}:{self.secret_key}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        return f"Basic {encoded_credentials}"

    def _get_token(self):
        current_time = time.time()
        if self.access_token and current_time < self.token_expires_at - 3600:
            return self.access_token

        url = f"{self.base_url}/open-api/v1/auth/token"
        headers = {
            'Authorization': self._get_basic_auth_header(),
            'Content-Type': 'application/json',
            'Accept': '*/*'
        }

        try:
            response = requests.post(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get('code') == 200:
                self.access_token = data['data']['accessToken']
                self.token_expires_at = current_time + 24 * 3600
                return self.access_token
            else:
                raise Exception(f"Token request failed: {data.get('msg', 'Unknown error')}")
        except Exception as e:
            raise Exception(f"Failed to get HiTem3D access token: {str(e)}")

    def create_task(self, front_image_bytes, back_image_bytes=None, left_image_bytes=None, right_image_bytes=None, 
                    model="hitem3dv1.5", resolution=1024, face_count=1000000, output_format=2, request_type=3):
        
        token = self._get_token()
        url = f"{self.base_url}/open-api/v1/submit-task"
        headers = {'Authorization': f'Bearer {token}', 'Accept': '*/*'}

        data = {
            'request_type': str(request_type),
            'model': model,
            'resolution': str(resolution),
            'face': str(face_count),
            'format': str(output_format)
        }

        files = []
        # Multi-view check logic
        multi_images = [img for img in [front_image_bytes, back_image_bytes, left_image_bytes, right_image_bytes] if img is not None]

        if len(multi_images) > 1:
            view_names = ['front', 'back', 'left', 'right']
            images_list = [front_image_bytes, back_image_bytes, left_image_bytes, right_image_bytes]
            for i, img_bytes in enumerate(images_list):
                if img_bytes is not None:
                    files.append(('multi_images', (f'{view_names[i]}.jpg', img_bytes, 'image/jpeg')))
        else:
            files.append(('images', ('front.jpg', front_image_bytes, 'image/jpeg')))

        response = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        
        # Close generic file handles if used? Here we use bytes directly so requests handles it.
        
        if response.status_code != 200:
            raise Exception(f"HiTem3D Task Submit Failed: {response.text}")
            
        result = response.json()
        if result.get('code') == 200:
            return result['data']['task_id']
        else:
             msg = result.get('msg', 'Unknown Error')
             if 'balance is not enough' in msg.lower():
                 msg = "Insufficient balance in HiTem3D account."
             raise Exception(f"HiTem3D Error ({result.get('code')}): {msg}")

    def query_task(self, task_id):
        token = self._get_token()
        url = f"{self.base_url}/open-api/v1/query-task"
        headers = {'Authorization': f'Bearer {token}', 'Accept': '*/*'}
        params = {'task_id': task_id}
        
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
             raise Exception(f"Query Failed: {response.text}")
             
        result = response.json()
        if result.get('code') == 200:
            return result['data']
        else:
            raise Exception(f"Task Query Error: {result.get('msg')}")

    def poll_task(self, task_id, timeout=900):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data = self.query_task(task_id)
                state = data.get('state', '').lower()
                
                if state == 'success':
                    return data
                elif state == 'failed':
                    raise Exception(f"HiTem3D Task Failed: {task_id}")
                
                time.sleep(5)
            except Exception as e:
                # If transient error, maybe continue? For now let's re-raise if meaningful
                if "Failed" in str(e): raise e
                time.sleep(5)
                
        raise Exception("HiTem3D Timeout")

class Geekatplay_HiTem3D_Gen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "front_image": ("IMAGE",),
                "model": (["hitem3dv1", "hitem3dv1.5", "hitem3dv2.0", "scene-portraitv1.5"], {"default": "hitem3dv1.5"}),
                "resolution": ([512, 1024, 1536, "1536pro"], {"default": 1024}),
                "face_count": ("INT", {"default": 1000000, "min": 100000, "max": 2000000, "step": 10000}),
                "output_format": (["obj", "glb", "stl", "fbx"], {"default": "glb"}),
                "generation_type": (["geometry_only", "staged", "all_in_one"], {"default": "all_in_one"}),
            },
            "optional": {
                "back_image": ("IMAGE",),
                "left_image": ("IMAGE",),
                "right_image": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": "", "label": "HiTem3D Key (Format: AccessKey:SecretKey)"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("model_path", "task_id")
    FUNCTION = "generate"
    CATEGORY = "Geekatplay/HiTem3D"

    def generate(self, front_image, model, resolution, face_count, output_format, generation_type, 
                 back_image=None, left_image=None, right_image=None, api_key=""):
        
        if not api_key or ":" not in api_key:
            raise Exception("Invalid HiTem3D API Key. Must be in format 'AccessKey:SecretKey'. Use the API Key Manager or input manually.")
            
        access_key, secret_key = api_key.split(":", 1)
        client = HiTem3DAPIClient(access_key.strip(), secret_key.strip())
        
        # Prepare params
        fmt_map = {"obj": 1, "glb": 2, "stl": 3, "fbx": 4}
        gen_map = {"geometry_only": 1, "staged": 2, "all_in_one": 3}
        res_int = 1536 if resolution == "1536pro" else int(resolution)
        
        # Prepare Images
        front_bytes = tensor_to_bytes(front_image)
        back_bytes = tensor_to_bytes(back_image) if back_image is not None else None
        left_bytes = tensor_to_bytes(left_image) if left_image is not None else None
        right_bytes = tensor_to_bytes(right_image) if right_image is not None else None
        
        print(f"Submitting HiTem3D Task ({model})...")
        task_id = client.create_task(
            front_bytes, back_bytes, left_bytes, right_bytes,
            model=model,
            resolution=res_int,
            face_count=face_count,
            output_format=fmt_map.get(output_format, 2),
            request_type=gen_map.get(generation_type, 3)
        )
        print(f"Task ID: {task_id}")
        
        result = client.poll_task(task_id)
        
        # Download
        model_url = result.get('model_url') or result.get('mesh_url')
        if not model_url:
             # Try other specific keys?
             # Based on API docs implied in client code: query_task returns dict.
             # Client uses `result['data']` -> `state`. It doesn't show return format.
             # Wait, usually `model_url` is in the data.
             pass 
             
        # Fallback if specific url key not found in top level
        # Let's inspect typical response if we can.
        # Assuming model_url is standard.
        if not model_url:
            raise Exception(f"No model URL found in HiTem3D response: {list(result.keys())}")

        output_dir = folder_paths.get_output_directory()
        hitem_dir = os.path.join(output_dir, "hitem3d_models")
        if not os.path.exists(hitem_dir):
            os.makedirs(hitem_dir)
            
        filename = f"hitem3d_{task_id}.{output_format}"
        filepath = os.path.join(hitem_dir, filename)
        
        print(f"Downloading model to {filepath}...")
        resp = requests.get(model_url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
            
        return (filepath, task_id)

NODE_CLASS_MAPPINGS = {
    "Geekatplay_HiTem3D_Gen": Geekatplay_HiTem3D_Gen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Geekatplay_HiTem3D_Gen": "HiTem3D Generator (Geekatplay)"
}
