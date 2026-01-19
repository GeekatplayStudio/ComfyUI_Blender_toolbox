# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox

import socket
import os
import tempfile
import folder_paths
import numpy as np
import imageio
import torch
from PIL import Image, ImageFilter

# Shared Constants
DEFAULT_BLENDER_IP = "127.0.0.1"
DEFAULT_BLENDER_PORT = 8119

def send_socket_message(host, port, message, timeout=5):
    """Helper to send a message to the Blender socket server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, port))
            s.sendall(message.encode('utf-8'))
        return True
    except (ConnectionRefusedError, OSError) as e:
        if isinstance(e, ConnectionRefusedError) or (hasattr(e, 'winerror') and e.winerror == 10061):
             print(f"[ComfyUI-360] Connection Refused: {host}:{port}. Is Blender running with the addon listener started?")
        else:
             print(f"[ComfyUI-360] Socket Error: {e}")
        return False
    except Exception as e:
        print(f"[ComfyUI-360] Unexpected Transmission Error: {e}")
        return False

class SendToBlender:
    """Deprecated / Placeholder Node"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "send"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/Legacy"

    def send(self, images):
        print("[ComfyUI-360] Warning: 'SendToBlender' is deprecated. Please use 'Preview In Blender' nodes.")
        return {}

class PreviewInBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),
                "blender_ip_address": ("STRING", {"default": DEFAULT_BLENDER_IP}),
                "blender_listen_port": ("INT", {"default": DEFAULT_BLENDER_PORT, "min": 1024, "max": 65535, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_to_blender"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def send_to_blender(self, file_path, blender_ip_address=DEFAULT_BLENDER_IP, blender_listen_port=DEFAULT_BLENDER_PORT):
        print(f"[ComfyUI-360] Sending Image Path: {file_path}")
        if send_socket_message(blender_ip_address, blender_listen_port, file_path):
            print(f"[ComfyUI-360] Success.")
        return {}

class PreviewModelInBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"forceInput": True}),
                "blender_ip_address": ("STRING", {"default": DEFAULT_BLENDER_IP}),
                "blender_listen_port": ("INT", {"default": DEFAULT_BLENDER_PORT, "min": 1024, "max": 65535, "step": 1}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "send_model"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/360 HDRI"
    
    def send_model(self, model_path, blender_ip_address=DEFAULT_BLENDER_IP, blender_listen_port=DEFAULT_BLENDER_PORT):
         message = f"MODEL:{model_path}"
         print(f"[ComfyUI-360] Sending Model Path: {model_path}")
         
         if send_socket_message(blender_ip_address, blender_listen_port, message):
             print(f"[ComfyUI-360] Success.")
         else:
             print(f"[ComfyUI-360] Failed to send model.")
            
         return {}

class PreviewTextureOnMesh:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "albedo_map": ("IMAGE", ),
                "blender_ip_address": ("STRING", {"default": DEFAULT_BLENDER_IP}),
                "blender_listen_port": ("INT", {"default": DEFAULT_BLENDER_PORT, "min": 1024, "max": 65535, "step": 1}),
            },
            "optional": {
                "roughness_map": ("IMAGE", ),
                "normal_map": ("IMAGE", ),
                "metallic_map": ("IMAGE", ),
                "alpha_map": ("IMAGE", ),
                "emission_map": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_texture"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def send_texture(self, albedo_map, blender_ip_address=DEFAULT_BLENDER_IP, blender_listen_port=DEFAULT_BLENDER_PORT, roughness_map=None, normal_map=None, metallic_map=None, alpha_map=None, emission_map=None):
        # Save to temp dir
        output_dir = folder_paths.get_temp_directory()
        
        # Message Builder
        message_parts = ["TEXTURE_UPDATE:"]
        
        # Helper to save and append
        def process_map(image_tensor, type_name, filename_suffix):
            if image_tensor is not None:
                filename = f"ComfyUI_{filename_suffix}_Temp.png"
                filepath = os.path.join(output_dir, filename)
                
                try:
                    # Convert - Handle Batch (Take First)
                    if image_tensor.shape[0] > 1:
                        print(f"[ComfyUI-360] Warning: Batch size > 1 for {type_name}. Using first image only.")
                        
                    img_np = image_tensor[0].cpu().numpy()
                    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                    imageio.imwrite(filepath, img_np)
                    
                    message_parts.append(f"{type_name}:{filepath}")
                except Exception as e:
                    print(f"[ComfyUI-360] Error saving {type_name}: {e}")

        process_map(albedo_map, "ALBEDO", "Albedo")
        process_map(roughness_map, "ROUGHNESS", "Roughness")
        process_map(normal_map, "NORMAL", "Normal")
        process_map(metallic_map, "METALLIC", "Metallic")
        process_map(alpha_map, "ALPHA", "Alpha")
        process_map(emission_map, "EMISSION", "Emission")
        
        # Reconstruct message
        full_message = "|".join(message_parts)
        
        # Send
        print(f"[ComfyUI-360] Sending Texture Update...")
        send_socket_message(blender_ip_address, blender_listen_port, full_message)

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
                "auto_level_height": ("BOOLEAN", {"default": True}),
                "height_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "smoothing_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "blender_ip_address": ("STRING", {"default": DEFAULT_BLENDER_IP}),
                "blender_listen_port": ("INT", {"default": DEFAULT_BLENDER_PORT, "min": 1024, "max": 65535, "step": 1}),
                "edge_falloff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "rotation": (["0", "90", "180", "270"], ),
                "roughness_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "roughness_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mesh_scale_1px_m": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 100.0, "step": 0.001, "tooltip": "Physical size of 1 pixel in meters. Controls the scale of the terrain in Blender."}),
            },
            "optional": {
                "albedo_map": ("IMAGE", ),
                "roughness_map": ("IMAGE", ),
                "normal_map": ("IMAGE", ),
                "metallic_map": ("IMAGE", ),
                "depth_map": ("IMAGE", ),
                "alpha_map": ("IMAGE", ),
                "ior_map": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("heightmap", )
    FUNCTION = "send_heightmap"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/360 HDRI"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def send_heightmap(self, images, generate_pbr=True, use_texture_as_heightmap=False, auto_level_height=True, height_gamma=1.0, smoothing_amount=1.0, edge_falloff=0.0, rotation="0", roughness_min=0.0, roughness_max=1.0, mesh_scale_1px_m=0.01, albedo_map=None, roughness_map=None, normal_map=None, metallic_map=None, depth_map=None, alpha_map=None, ior_map=None, blender_ip_address=DEFAULT_BLENDER_IP, blender_listen_port=DEFAULT_BLENDER_PORT):
        # Resolve inputs
        texture = albedo_map 

        # Save to temp dir
        output_dir = folder_paths.get_temp_directory()
        filename = "ComfyUI_Heightmap_Temp.png"
        filepath = os.path.join(output_dir, filename)
        
        # Decide source for heightmap
        img_np = None
        
        if depth_map is not None:
             # Explicit depth input overrides other logic
             img_np = depth_map[0].cpu().numpy()
        elif use_texture_as_heightmap and texture is not None:
            # Use texture as heightmap (Grayscale conversion)
            img_np = texture[0].cpu().numpy()
            # RGB to Grayscale using standard luminance weights
            if img_np.shape[2] >= 3:
                img_gray = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
                img_np = img_gray 
            
            # --- ANALYSIS & CORRECTION ---
            # 1. Auto-Levels (Normalize)
            if auto_level_height:
                h_min = np.min(img_np)
                h_max = np.max(img_np)
                if h_max > h_min: # Prevent divide by zero
                    img_np = (img_np - h_min) / (h_max - h_min)
            
            # 2. Gamma Correction
            if height_gamma != 1.0:
               img_np = np.power(img_np, 1.0 / height_gamma)
            
            # Expand back to 3 channels for saving logic
            img_np = np.stack((img_np,)*3, axis=-1)
                
        else:
            # Use provided heightmap images
            img_np = images[0].cpu().numpy() # [H, W, C]
        
        # Convert to 0-255 uint8
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        # Apply Smoothing
        if smoothing_amount > 0:
            try:
                if img_np.ndim == 2:
                     pil_img = Image.fromarray(img_np, mode='L')
                else:
                     pil_img = Image.fromarray(img_np)
                pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=smoothing_amount))
                img_np = np.array(pil_img)
            except Exception as e:
                print(f"[ComfyUI-360] Warning: Smoothing failed: {e}")

        # Apply Rotation
        if rotation != "0":
            k = int(int(rotation) / 90)
            img_np = np.rot90(img_np, k=k, axes=(0, 1))

        # Apply Edge Falloff
        if edge_falloff > 0.0:
            H, W = img_np.shape[:2]
            x = np.linspace(0, 1, W)
            ramp_x = np.minimum(x/edge_falloff, (1.0-x)/edge_falloff)
            ramp_x = np.clip(ramp_x, 0, 1)
            y = np.linspace(0, 1, H)
            ramp_y = np.minimum(y/edge_falloff, (1.0-y)/edge_falloff)
            ramp_y = np.clip(ramp_y, 0, 1)
            mask = np.outer(ramp_y, ramp_x)
            
            if img_np.ndim == 3:
                 img_np = (img_np.astype(np.float32) * mask[..., None]).astype(np.uint8)
            else:
                 img_np = (img_np.astype(np.float32) * mask).astype(np.uint8)

        # Output Image Logic
        out_float = img_np.astype(np.float32) / 255.0
        if out_float.ndim == 2:
            out_float = np.stack((out_float,)*3, axis=-1)
        
        out_tensor = torch.from_numpy(out_float).unsqueeze(0)
        
        try:
            imageio.imwrite(filepath, img_np)
        except Exception as e:
            print(f"[ComfyUI-360] Error saving heightmap: {e}")

        # Helper for textures
        def save_tex(tensor, suffix):
            if tensor is None: return ""
            fname = f"ComfyUI_{suffix}_Temp.png"
            fpath = os.path.join(output_dir, fname)
            try:
                t_np = tensor[0].cpu().numpy()
                t_np = (np.clip(t_np, 0, 1) * 255).astype(np.uint8)
                imageio.imwrite(fpath, t_np)
                return fpath
            except Exception as e:
                print(f"[ComfyUI-360] Error saving {suffix}: {e}")
                return ""

        texture_filepath = save_tex(texture, "Texture")
        roughness_filepath = save_tex(roughness_map, "Roughness")
        normal_filepath = save_tex(normal_map, "Normal")
        metallic_filepath = save_tex(metallic_map, "Metallic")
        alpha_filepath = save_tex(alpha_map, "Alpha")
        ior_filepath = save_tex(ior_map, "IOR")

        # Calculate Physical Dimensions
        H, W = img_np.shape[:2]
        phys_w = W * mesh_scale_1px_m
        phys_h = H * mesh_scale_1px_m
        
        # Send to Blender
        # Message format: HEIGHTMAP:<height_path>|...|SIZE_X:<val>|SIZE_Y:<val>
        parts = [f"HEIGHTMAP:{filepath}"]
        if texture_filepath: parts.append(f"TEXTURE:{texture_filepath}")
        parts.append(f"PBR:{'true' if generate_pbr else 'false'}")
        
        if roughness_filepath: parts.append(f"ROUGHNESS:{roughness_filepath}")
        if normal_filepath: parts.append(f"NORMAL:{normal_filepath}")
        if metallic_filepath: parts.append(f"METALLIC:{metallic_filepath}")
        if alpha_filepath: parts.append(f"ALPHA:{alpha_filepath}")
        if ior_filepath: parts.append(f"IOR:{ior_filepath}")
            
        parts.append(f"ROUGHNESS_MIN:{roughness_min:.3f}")
        parts.append(f"ROUGHNESS_MAX:{roughness_max:.3f}")
        parts.append(f"SIZE_X:{phys_w:.4f}")
        parts.append(f"SIZE_Y:{phys_h:.4f}")
        
        message = "|".join(parts)
        
        send_socket_message(blender_ip_address, blender_listen_port, message)

        return (out_tensor, )

class PreviewMeshInBlender:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH", ),
                "blender_ip_address": ("STRING", {"default": DEFAULT_BLENDER_IP}),
                "blender_listen_port": ("INT", {"default": DEFAULT_BLENDER_PORT, "min": 1024, "max": 65535, "step": 1}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "send_mesh"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/360 HDRI"
    
    def send_mesh(self, mesh, blender_ip_address=DEFAULT_BLENDER_IP, blender_listen_port=DEFAULT_BLENDER_PORT):
        # 1. Convert MESH to GLB
        import trimesh
        
        output_dir = folder_paths.get_temp_directory()
        filename = "ComfyUI_Mesh_Temp.glb"
        filepath = os.path.join(output_dir, filename)
        
        final_mesh = None
        
        # --- Type Detection Strategy ---
        try:
            # Case A: Trimesh Object
            if isinstance(mesh, trimesh.Trimesh):
                final_mesh = mesh
            
            # Case B: List of meshes (batch) -> Take first
            elif isinstance(mesh, list) and len(mesh) > 0:
                 if isinstance(mesh[0], trimesh.Trimesh):
                     final_mesh = mesh[0]
                 
                 # Case B2: List of tuples [(verts, faces)] (Common batch format)
                 elif isinstance(mesh[0], tuple) and len(mesh[0]) >= 2:
                     verts = mesh[0][0]
                     faces = mesh[0][1]
                     
                     if isinstance(verts, torch.Tensor): verts = verts.cpu().numpy()
                     if isinstance(faces, torch.Tensor): faces = faces.cpu().numpy()
                     
                     if verts.ndim > 2: verts = verts.reshape(-1, 3)
                     if faces.ndim > 2: faces = faces.reshape(-1, 3)
                     
                     final_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, validate=False)

                 # Case C: Tensor tuple (verts, faces) - unpacked list [verts, faces]
                 elif isinstance(mesh[0], torch.Tensor) or isinstance(mesh[0], np.ndarray):
                     # Assume mesh = [vertices, faces]
                     if len(mesh) >= 2:
                         verts, faces = mesh[0], mesh[1]
                         if isinstance(verts, torch.Tensor): verts = verts.cpu().numpy()
                         if isinstance(faces, torch.Tensor): faces = faces.cpu().numpy()
                         
                         if verts.ndim > 2: verts = verts.reshape(-1, 3)
                         if faces.ndim > 2: faces = faces.reshape(-1, 3)
                         
                         final_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, validate=False)
            
            # Case D: Object with .vertices .faces (SimpleNamespace or custom class)
            elif hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                 verts = mesh.vertices
                 faces = mesh.faces
                 if isinstance(verts, torch.Tensor): verts = verts.cpu().numpy()
                 if isinstance(faces, torch.Tensor): faces = faces.cpu().numpy()
                 final_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, validate=False)

            if final_mesh is None:
                 print(f"[ComfyUI-360] Error: Unknown MESH format: {type(mesh)}. Try converting to trimesh first.")
                 return {}
                 
            # 2. Export
            final_mesh.export(filepath)
            print(f"[ComfyUI-360] Saved Temp Mesh to {filepath}")
            
            # 3. Send
            message = f"MODEL:{filepath}"
            send_socket_message(blender_ip_address, blender_listen_port, message)
            
        except Exception as e:
            print(f"[ComfyUI-360] Error processing mesh: {e}")
            
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
                "blender_ip_address": ("STRING", {"default": DEFAULT_BLENDER_IP}),
                "blender_listen_port": ("INT", {"default": DEFAULT_BLENDER_PORT, "min": 1024, "max": 65535, "step": 1}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "sync_lighting"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def sync_lighting(self, azimuth, elevation, intensity, color_hex, blender_ip_address=DEFAULT_BLENDER_IP, blender_listen_port=DEFAULT_BLENDER_PORT):
        # Message format: LIGHTING:azimuth=<val>|elevation=<val>|intensity=<val>|color=<hex>
        message = f"LIGHTING:azimuth={azimuth}|elevation={elevation}|intensity={intensity}|color={color_hex}"
        send_socket_message(blender_ip_address, blender_listen_port, message)
        return {}

class LoadBlenderPBR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_name": ("STRING", {"default": "from_blender"}),
                "load_mode": (["Single Folder", "Latest Subfolder"], {"default": "Single Folder"}),
            },
            "optional": {
                "force_reload": ("INT", {"default": 0, "min": 0, "max": 10000}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("Albedo", "Roughness", "Normal", "Metallic", "Alpha", "Emission", "Specular", "UV Layout", "Directory Path")
    FUNCTION = "load_textures"
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def load_textures(self, folder_name="from_blender", load_mode="Single Folder", force_reload=0):
        # Determine path
        # Check if folder_name is essentially an absolute path
        if os.path.isabs(folder_name) and os.path.exists(folder_name):
            target_dir = folder_name
        else:
            input_dir = folder_paths.get_input_directory()
            target_dir = os.path.join(input_dir, folder_name)
        
        target_dir = target_dir.strip('"').strip("'")
        
        # Latest Subfolder Logic
        if load_mode == "Latest Subfolder" and os.path.exists(target_dir):
             try:
                 subdirs = [os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
                 if subdirs:
                     latest_subdir = max(subdirs, key=os.path.getmtime)
                     print(f"[ComfyUI-360] Found latest subfolder -> {latest_subdir}")
                     target_dir = latest_subdir
             except Exception as e:
                 print(f"[ComfyUI-360] Error finding subfolder: {e}")

        # 3. Fallback / Auto-Discovery Logic
        if not os.path.exists(target_dir) or (folder_name == "from_blender" and load_mode == "Latest Subfolder"):
            link_file = os.path.join(tempfile.gettempdir(), "comfy_360_last_export.txt")
            if os.path.exists(link_file):
                try:
                    with open(link_file, "r") as f:
                        linked_path = f.read().strip()
                    if os.path.exists(linked_path):
                         print(f"[ComfyUI-360] Auto-Discovery: Found link from Blender -> {linked_path}")
                         target_dir = linked_path
                except: pass

        if not os.path.exists(target_dir):
            print(f"[Warn] Buffer directory not found: {target_dir}")
            # Check for common temp locations used by our addon as backup
            temp_loc = os.path.join(tempfile.gettempdir(), "comfy_blender_export")
            if os.path.exists(temp_loc):
                target_dir = temp_loc 
                if load_mode == "Latest Subfolder":
                     # Re-check latest inside temp
                     subdirs = [os.path.join(target_dir, d) for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
                     if subdirs:
                         target_dir = max(subdirs, key=os.path.getmtime)

        if not os.path.exists(target_dir):
            print(f"[Error] Could not find any texture buffer at {target_dir}")
            black = torch.zeros((1, 64, 64, 3))
            return (black, black, black, black, black, black, black, black, "")

        # Helper to load single image
        def load_one(fpath, is_alpha=False, target_size=None):
             if not os.path.exists(fpath): return None
             try:
                 i = Image.open(fpath)
                 from PIL import ImageOps
                 i = ImageOps.exif_transpose(i)
                 
                 if is_alpha:
                     # For Alpha/UV layout, ensure 4 channels
                     if i.mode != 'RGBA': i = i.convert('RGBA')
                     # Create Mask from alpha channel if needed? Comfy uses [batch, H, W, channels]
                     # If image is just transparency, 'RGB' would be white/black, 'RGBA' keeps alpha.
                 else:
                     i = i.convert("RGB")
                 
                 if target_size and i.size != target_size:
                     i = i.resize(target_size, Image.LANCZOS)
                     
                 image = np.array(i).astype(np.float32) / 255.0
                 return torch.from_numpy(image)[None,]
             except Exception as e:
                 print(f"Error loading {fpath}: {e}")
                 return None

        # Determine batch mode
        import re
        max_idx = -1
        
        # List files once
        files = []
        try:
            files = os.listdir(target_dir)
            pattern = re.compile(r"^(\d+)_blender_.*\.png$")
            
            for f in files:
                m = pattern.match(f)
                if m:
                    idx = int(m.group(1))
                    if idx > max_idx: max_idx = idx
        except: pass

        print(f"[ComfyUI-360] Max Index found: {max_idx}")

        def load_batch(base_name, is_alpha=False):
             # 1. Check for UV Layout specifically (no index usually, or simple name)
             if base_name == "uv_layout.png":
                 fpath = os.path.join(target_dir, base_name)
                 if os.path.exists(fpath):
                      return load_one(fpath, is_alpha) or torch.zeros((1, 64, 64, 4 if is_alpha else 3))

             # Single mode (no indices found)
             if max_idx == -1:
                 fpath = os.path.join(target_dir, base_name)
                 if os.path.exists(fpath):
                     res = load_one(fpath, is_alpha)
                     if res is not None: return res
                 return torch.zeros((1, 64, 64, 4 if is_alpha else 3))

             # Multi mode
             first_size = None # (W, H)
             
             # First pass: find first valid image size
             for i in range(max_idx + 1):
                 fname = f"{i}_{base_name}"
                 fpath = os.path.join(target_dir, fname)
                 if os.path.exists(fpath):
                      try:
                          with Image.open(fpath) as img:
                              first_size = img.size
                              break
                      except: pass
             
             if first_size is None: first_size = (64, 64)

             # Second pass: load and stack
             tensors = []
             for i in range(max_idx + 1):
                 fname = f"{i}_{base_name}"
                 fpath = os.path.join(target_dir, fname)
                 
                 res = load_one(fpath, is_alpha, target_size=first_size)
                 if res is None:
                     res = torch.zeros((1, first_size[1], first_size[0], 4 if is_alpha else 3))
                 tensors.append(res)
            
             if not tensors: return torch.zeros((1, 64, 64, 4 if is_alpha else 3))
                  
             return torch.cat(tensors, dim=0)

        albedo = load_batch("blender_albedo.png")
        roughness = load_batch("blender_roughness.png")
        normal = load_batch("blender_normal.png")
        metallic = load_batch("blender_metallic.png")
        alpha = load_batch("blender_alpha.png", is_alpha=True)
        emission = load_batch("blender_emission.png")
        specular = load_batch("blender_specular.png")
        uv_layout = load_batch("uv_layout.png", is_alpha=True)
        
        return (albedo, roughness, normal, metallic, alpha, emission, specular, uv_layout, target_dir)


class SaveAndSendPBRToBlender:
    def __init__(self):
        self.type = "pbr_sender"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "albedo_map": ("IMAGE", ),
                "directory_path": ("STRING", {"forceInput": True, "default": ""}), 
                "blender_ip_address": ("STRING", {"default": DEFAULT_BLENDER_IP}),
                "blender_listen_port": ("INT", {"default": DEFAULT_BLENDER_PORT, "min": 1024, "max": 65535, "step": 1}),
            },
            "optional": {
                "roughness_map": ("IMAGE", ),
                "normal_map": ("IMAGE", ),
                "metallic_map": ("IMAGE", ),
                "alpha_map": ("IMAGE", ),
                "emission_map": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_and_send"
    OUTPUT_NODE = True
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def save_and_send(self, albedo_map, directory_path, blender_ip_address=DEFAULT_BLENDER_IP, blender_listen_port=DEFAULT_BLENDER_PORT, roughness_map=None, normal_map=None, metallic_map=None, alpha_map=None, emission_map=None):
        # Determine output folder
        output_dir = ""
        is_temp = False
        
        if directory_path and os.path.exists(directory_path) and os.path.isdir(directory_path):
             output_dir = directory_path
        else:
             output_dir = folder_paths.get_temp_directory()
             is_temp = True
             print(f"[ComfyUI-360] Reference Directory invalid, using Temp: {output_dir}")

        # Message Builder
        message_parts = ["TEXTURE_UPDATE:"]
        
        # Helper to save and append
        def process_map(image_tensor, type_name, filename_suffix, standard_name):
            if image_tensor is not None:
                if is_temp:
                    filename = f"ComfyUI_{filename_suffix}_Temp.png"
                else:
                    filename = standard_name

                filepath = os.path.join(output_dir, filename)
                
                # Convert
                try:
                    if image_tensor.ndim == 4:
                         img_np = image_tensor[0].cpu().numpy()
                    else:
                         img_np = image_tensor.cpu().numpy()
                         
                    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                    imageio.imwrite(filepath, img_np)
                    message_parts.append(f"{type_name}:{filepath}")
                except Exception as e:
                    print(f"[ComfyUI-360] Error saving {type_name}: {e}")

        process_map(albedo_map, "ALBEDO", "Albedo", "comfy_albedo.png")
        process_map(roughness_map, "ROUGHNESS", "Roughness", "comfy_roughness.png")
        process_map(normal_map, "NORMAL", "Normal", "comfy_normal.png")
        process_map(metallic_map, "METALLIC", "Metallic", "comfy_metallic.png")
        process_map(alpha_map, "ALPHA", "Alpha", "comfy_alpha.png")
        process_map(emission_map, "EMISSION", "Emission", "comfy_emission.png")
        
        # Reconstruct message
        full_message = "|".join(message_parts)
        send_socket_message(blender_ip_address, blender_listen_port, full_message)

        return {}

# Mappings
NODE_CLASS_MAPPINGS = {
    "SendToBlender": SendToBlender,
    "PreviewInBlender": PreviewInBlender,
    "PreviewModelInBlender": PreviewModelInBlender,
    "PreviewTextureOnMesh": PreviewTextureOnMesh,
    "PreviewHeightmapInBlender": PreviewHeightmapInBlender,
    "PreviewMeshInBlender": PreviewMeshInBlender,
    "SyncLightingToBlender": SyncLightingToBlender,
    "LoadBlenderPBR": LoadBlenderPBR,
    "SaveAndSendPBRToBlender": SaveAndSendPBRToBlender
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SendToBlender": "Send Image to Blender (Legacy)",
    "PreviewInBlender": "Preview Image in Blender",
    "PreviewModelInBlender": "Preview Model in Blender",
    "PreviewTextureOnMesh": "Preview Texture on Mesh",
    "PreviewHeightmapInBlender": "Preview Heightmap in Blender",
    "PreviewMeshInBlender": "Send/Preview Mesh in Blender",
    "SyncLightingToBlender": "Sync Lighting to Blender",
    "LoadBlenderPBR": "Load Blender PBR Textures",
    "SaveAndSendPBRToBlender": "Save and Send PBR to Blender"
}
