import os
import json
import base64
from itertools import cycle
from server import PromptServer
from aiohttp import web

# Path to the encrypted key store
KEY_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "geekatplay_keystore.enc")
OBFUSCATION_KEY = "GeekatplayStudio_Secure_Key_Salt_2026"

def xor_crypt_string(data, key=OBFUSCATION_KEY, encode=False):
    """Simple XOR obfuscation to prevent cleartext storage."""
    # Ensure key length matches data if possible or cycle
    if encode:
        # XOR then Base64
        try:
            xored = ''.join(chr(ord(x) ^ ord(y)) for (x, y) in zip(data, cycle(key)))
            return base64.b64encode(xored.encode('utf-8')).decode('utf-8')
        except Exception as e:
            print(f"Encryption error: {e}")
            return ""
    else:
        # Base64 decode then XOR
        try:
            if not data: return ""
            decoded_b64 = base64.b64decode(data).decode('utf-8')
            return ''.join(chr(ord(x) ^ ord(y)) for (x, y) in zip(decoded_b64, cycle(key)))
        except Exception as e:
            print(f"Decryption error: {e}")
            return ""

def load_keys():
    if not os.path.exists(KEY_STORE_PATH):
        return {}
    try:
        with open(KEY_STORE_PATH, 'r') as f:
            encrypted_data = f.read()
            json_str = xor_crypt_string(encrypted_data, encode=False)
            if not json_str: return {}
            return json.loads(json_str)
    except Exception as e:
        print(f"Error loading keystore: {e}")
        return {}

def save_key(name, value):
    keys = load_keys()
    keys[name] = value
    json_str = json.dumps(keys)
    encrypted_data = xor_crypt_string(json_str, encode=True)
    with open(KEY_STORE_PATH, 'w') as f:
        f.write(encrypted_data)

def delete_key(name):
    keys = load_keys()
    if name in keys:
        del keys[name]
        json_str = json.dumps(keys)
        encrypted_data = xor_crypt_string(json_str, encode=True)
        with open(KEY_STORE_PATH, 'w') as f:
            f.write(encrypted_data)

# Register API routes for the JS frontend
if hasattr(PromptServer, "instance"):
    routes = PromptServer.instance.routes
    
    @routes.post("/geekatplay/save_key")
    async def save_key_endpoint(request):
        try:
            json_data = await request.json()
            key_name = json_data.get("name")
            key_value = json_data.get("value")
            
            if not key_name or not key_value:
                return web.json_response({"status": "error", "message": "Missing name or value"}, status=400)
                
            save_key(key_name, key_value)
            return web.json_response({"status": "success"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.post("/geekatplay/delete_key")
    async def delete_key_endpoint(request):
        try:
            json_data = await request.json()
            key_name = json_data.get("name")
            
            if not key_name:
                return web.json_response({"status": "error", "message": "Missing name"}, status=400)
                
            delete_key(key_name)
            return web.json_response({"status": "success"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    @routes.get("/geekatplay/list_keys")
    async def list_keys_endpoint(request):
        keys = load_keys()
        return web.json_response({"keys": list(keys.keys())})

class Geekatplay_ApiKey_Manager:
    @classmethod
    def INPUT_TYPES(s):
        # Load keys to populate dropdown (server-side initial render)
        keys = load_keys()
        # Always include "None" to prevent validation errors if workflow has "None" saved
        # but keys exist in the backend.
        key_names = ["None"] + sorted(list(keys.keys()))
            
        return {
            "required": {
                "service_name": (key_names, ),
                # Mode switch is kept for backward compatibility and fallback
                "mode": (["Read/Select", "SAVE New Key", "DELETE Selected"], {"default": "Read/Select"}),
            },
            "optional": {
                # These remain for the fallback manual entry if JS fails or for automation
                "new_key_name": ("STRING", {"default": "MyService"}),
                "new_key_value": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "manage_keys"
    CATEGORY = "Geekatplay Studio/Utils"
    OUTPUT_NODE = True

    def manage_keys(self, service_name, mode, new_key_name="MyService", new_key_value=""):
        keys = load_keys()
        
        if mode == "SAVE New Key":
            if new_key_name and new_key_value:
                save_key(new_key_name, new_key_value.strip())
                print(f"[Geekatplay KeyManager] Saved key for '{new_key_name}'. Please refresh browser to see in dropdown.")
                return (new_key_value.strip(),)
            else:
                print(f"[Geekatplay KeyManager] Error: Name and Value required to save.")
                return ("",)
                
        elif mode == "DELETE Selected":
            if service_name in keys:
                delete_key(service_name)
                print(f"[Geekatplay KeyManager] Deleted key for '{service_name}'. Please refresh browser.")
                return ("",)
            else:
                return ("",)
                
        else: # Read/Select
            # Primary: Try to get from keystore
            if service_name in keys:
                return (keys[service_name],)
            
            # Fallback: If user entered a key in new_key_value but didn't save, use it (Pass-through mode)
            # This helps when setting up first time and forgetting to switch mode
            if new_key_value and new_key_value.strip():
                 print(f"[Geekatplay KeyManager] Warning: Using direct key value because '{service_name}' was not found in keystore. Recommended: Switch mode to 'SAVE' to store it securely.")
                 return (new_key_value.strip(),)
                 
            return ("",)

    @classmethod
    def IS_CHANGED(s, service_name, mode, new_key_name, new_key_value):
        # Force re-execution if mode is write/delete
        if mode != "Read/Select":
            return float("nan")
        return service_name

NODE_CLASS_MAPPINGS = {
    "Geekatplay_ApiKey_Manager": Geekatplay_ApiKey_Manager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Geekatplay_ApiKey_Manager": "API Key Manager (Geekatplay)"
}
