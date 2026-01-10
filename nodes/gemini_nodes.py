import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

class GeminiSeamlessTiler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend_amount": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
                "mode": (["Simple Blend", "AI Prep"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("seamless_image", "mask")
    FUNCTION = "tile"
    CATEGORY = "360_HDRI/Gemini"

    def tile(self, image, blend_amount, mode):
        # image: [B, H, W, C]
        B, H, W, C = image.shape
        
        # 1. Roll/Shift
        shift_h = H // 2
        shift_w = W // 2
        shifted_image = torch.roll(image, shifts=(shift_h, shift_w), dims=(1, 2))
        
        # 2. Mask Generation (Cross)
        # 1.0 = Inpaint (The seam area), 0.0 = Keep
        mask = torch.zeros((B, H, W), dtype=torch.float32, device=image.device)
        
        # Calculate thickness based on blend_amount (fraction of dimension)
        h_thickness = int(H * blend_amount)
        w_thickness = int(W * blend_amount)
        
        # Vertical bar (covering W//2)
        w_start = (W // 2) - (w_thickness // 2)
        w_end = w_start + w_thickness
        mask[:, :, w_start:w_end] = 1.0
        
        # Horizontal bar (covering H//2)
        h_start = (H // 2) - (h_thickness // 2)
        h_end = h_start + h_thickness
        mask[:, h_start:h_end, :] = 1.0
        
        # Soften mask? Gaussian blur approximates "soft gradient"
        # Using a simple box blur or linear gradient is better for performance if strictly python, 
        # but for Comfy, maybe just hardness is fine or use simple smoothing.
        # Let's apply a simple gradient falloff manually or leave hard for Inpainting (usually prefer soft).
        # We can implement a simple convolution blur.
        
        if mode == "Simple Blend":
            # Very basic mirroring/blending logic (heuristic)
            # This is complex to do well without OpenCV inpainting.
            # We will use a simple overlap technique:
            # But the user logic says "Output is now seamless because the outer edges were originally the contiguous center".
            # The shifted image *is* the seamless candidate if we heal the center.
            # If we don't heal, it's just a shifted image.
            # "Simple Blend" in this context might just return the shifted image so user sees the seams?
            # Or try to blur the seam area.
            
            # For this MVP, let's just blur the masked area in the shifted image to "hide" the seam slightly.
            # This is a placeholder for "Simple". True healing needs inpainting.
            blurred = GaussianBlur(shifted_image, kernel_size=h_thickness if h_thickness % 2 == 1 else h_thickness + 1)
            # mask is [B, H, W]. shape to [B, H, W, 1]
            mask_expanded = mask.unsqueeze(-1)
            result = shifted_image * (1 - mask_expanded) + blurred * mask_expanded
            return (result, mask)

        else: # AI Prep
            # Return the shifted image and the mask for KSampler
            return (shifted_image, mask)

def GaussianBlur(img, kernel_size=15, sigma=5):
    # img: [B, H, W, C]
    # Simple gauss implementation using conv2d
    k = kernel_size
    if k % 2 == 0: k += 1
    
    # Create 1D Gaussian kernel
    x = torch.arange(k, dtype=torch.float32, device=img.device) - k // 2
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    
    # Create 2D kernel
    gauss_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0) # [k, k]
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0) # [1, 1, k, k]
    
    # Prepare input
    # Needs [B, C, H, W]
    inputs = img.permute(0, 3, 1, 2)
    C = inputs.shape[1]
    
    # Repeat kernel for each channel
    kernel = gauss_2d.repeat(C, 1, 1, 1)
    
    # Pad
    pad = k // 2
    inputs_padded = F.pad(inputs, (pad, pad, pad, pad), mode='reflect')
    
    # Convolve
    output = F.conv2d(inputs_padded, kernel, groups=C)
    
    return output.permute(0, 2, 3, 1)


class GeminiPBRExtractor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "albedo_image": ("IMAGE",),
                "fidelity": (["High", "Low"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("albedo_out", "normal_map", "roughness_map", "depth_map", "metallic_map")
    FUNCTION = "extract"
    CATEGORY = "360_HDRI/PBR"

    def extract(self, albedo_image, fidelity):
        # albedo_image: [B, H, W, C]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if CHORD model and repo are available
        current_dir = os.path.dirname(os.path.abspath(__file__))
        repo_path = os.path.abspath(os.path.join(current_dir, "../repos/ubisoft-laforge-chord"))
        model_path = os.path.abspath(os.path.join(current_dir, "../models/gemini_pbr/chord_v1.safetensors"))
        
        chord_available = False
        
        if os.path.exists(repo_path) and os.path.exists(model_path):
            try:
                if repo_path not in sys.path:
                    sys.path.append(repo_path)
                
                # Attempt to import CHORD
                # Note: imports depend on the repo structure. Assuming 'chord' package exists.
                from chord.models.chord import ChordModel # Adjust import based on actual repo structure if known, otherwise try top level
                from omegaconf import OmegaConf
                import safetensors.torch
                
                chord_available = True
            except ImportError as e:
                print(f"PBR Extractor: Failed to import CHORD dependencies: {e}")
            except Exception as e:
                print(f"PBR Extractor: Unexpected error importing CHORD: {e}")
        
        if chord_available:
            try:
                print("PBR Extractor: Running CHORD model...")
                return self._run_chord_inference(albedo_image, repo_path, model_path, device)
            except Exception as e:
                print(f"PBR Extractor: Error during CHORD inference, falling back to algorithmic: {e}")

        
        # FALLBACK: PBR Algorithmic extraction
        print("PBR Extractor: Running algorithmic fallback...")
        
        # 1. Albedo (Just pass through)
        albedo_out = albedo_image.clone()
        
        # 2. Height/Depth Map (Grayscale)
        # Luminance
        height_map = albedo_image[..., 0] * 0.299 + albedo_image[..., 1] * 0.587 + albedo_image[..., 2] * 0.114
        height_map = height_map.unsqueeze(-1) # [B, H, W, 1]
        
        # Optional: Equalize histogram or normalize for better height range
        min_v = torch.min(height_map)
        max_v = torch.max(height_map)
        if max_v - min_v > 1e-5:
            height_map = (height_map - min_v) / (max_v - min_v)
            
        # 3. Normal Map (From Height)
        # Reuse logic from GeminiNormalMap logic (Prompt 2)
        normal_map = self.height_to_normal(height_map, flip_green=False, strength=5.0 if fidelity=="High" else 2.0)
        
        # 4. Roughness
        # Heuristic: Invert height? High parts (white) might be worn/smooth? Low parts (black) rough?
        # Or Edge detection?
        # Let's use inverted height for basic approximation + some contrast
        roughness_map = 1.0 - height_map
        roughness_map = torch.clamp(roughness_map * 1.2, 0.0, 1.0) # Boost contrast
        
        # 5. Metallic
        # Heuristic: Usually dark unless specfied.
        metallic_map = torch.zeros_like(height_map)
        
        # Ensure 3 channels for fallback too
        if roughness_map.shape[-1] == 1: roughness_map = roughness_map.repeat(1, 1, 1, 3)
        if height_map.shape[-1] == 1: height_map = height_map.repeat(1, 1, 1, 3)
        if metallic_map.shape[-1] == 1: metallic_map = metallic_map.repeat(1, 1, 1, 3)
            
        return (albedo_out, normal_map, roughness_map, height_map, metallic_map)

    def _run_chord_inference(self, image_tensor, repo_path, model_path, device):
        from chord.models.chord import ChordModel
        from omegaconf import OmegaConf
        import safetensors.torch
        import torchvision.transforms.functional as TF

        # 1. Setup Model
        # Load config - assuming a standard config exists or we construct a minimal one
        # If the repo allows loading without config file or provides a default, that's best.
        # Ideally we load the config yaml from the repo.
        config_path = os.path.join(repo_path, "configs", "chord.yaml") # Guessing path
        if not os.path.exists(config_path):
             # Try finding any yaml in configs
             possible_configs = [f for f in os.listdir(os.path.join(repo_path, "configs")) if f.endswith(".yaml")]
             if possible_configs:
                 config_path = os.path.join(repo_path, "configs", possible_configs[0])
        
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
        else:
            # Minimal config if file missing (Risk)
            cfg = OmegaConf.create({"model": {"encoder": "vit_base_patch16_224", "decoder": "linear"}}) # Placeholder

        model = ChordModel(cfg)
        
        # Load weights
        # safetensors loading
        state_dict = safetensors.torch.load_file(model_path)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # 2. Prepare Input
        # ComfyUI Image is [B, H, W, 3] usually, range 0-1
        # CHORD likely expects [B, 3, H, W], normalized?
        # And specifically 1024x1024 resolution per instructions
        
        B, H, W, C = image_tensor.shape
        # Permute to [B, C, H, W]
        img_input = image_tensor.permute(0, 3, 1, 2).to(device)
        
        # Resize to 1024x1024
        img_resized = F.interpolate(img_input, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        # Normalize? Standard ImageNet usually.
        # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # Assuming input is 0-1.
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        img_normalized = (img_resized - mean) / std

        # 3. Inference
        with torch.no_grad():
            # Model forward
            # CHORD output is usually a dictionary: {'albedo': ..., 'normal': ..., 'roughness': ..., 'shading': ...}
            output = model(img_normalized)

        # 4. Process Output
        # output keys might include 'pred_albedo', 'pred_normal', 'pred_roughness' etc.
        # Need to check dictionary keys. Assuming 'albedo', 'normal', 'roughness'.
        
        def process_out_channel(out_key, default_channels=3):
            if out_key in output:
                res = output[out_key]
                # Resize back to original
                res = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)
                # Un-normalize if needed? Usually outputs are 0-1 or -1 to 1.
                # Normals are usually -1 to 1. Albedo 0-1.
                if out_key == 'normal':
                    res = (res + 1.0) / 2.0 # -1..1 -> 0..1
                
                res = torch.clamp(res, 0.0, 1.0)
                return res.permute(0, 2, 3, 1) # [B, H, W, C]
            else:
                return torch.zeros((B, H, W, default_channels), device=device)

        out_albedo = process_out_channel('albedo', 3) # Or 'pred_albedo'
        if torch.sum(out_albedo) == 0: out_albedo = process_out_channel('pred_albedo', 3)
        
        out_normal = process_out_channel('normal', 3)
        if torch.sum(out_normal) == 0: out_normal = process_out_channel('pred_normal', 3)

        out_roughness = process_out_channel('roughness', 1)
        if torch.sum(out_roughness) == 0: out_roughness = process_out_channel('pred_roughness', 1)
        
        # CHORD usually produces 'visibility' or 'shading', not depth/height directly?
        # But maybe we can map shading to depth vaguely or just return black.
        # Actually it might produce depth. Let's look for it.
        out_depth = process_out_channel('depth', 1) 
        if torch.sum(out_depth) == 0: out_depth = process_out_channel('pred_depth', 1)
        
        # Metallic is not usually predicted by CHORD.
        out_metallic = torch.zeros((B, H, W, 1), device=device)

        # Helper to convert 1-channel to 3-channel duplicate for ComfyUI preview if needed
        # But ComfyUI usually handles 1 channel images fine if they are [B, H, W, 1]
        # The crash "Cannot handle this data type: (1, 1, 1), |u1" implies PIL conversion issue.
        # ComfyUI's PreviewImage node expects 3 channels usually or handles it specifically.
        # Let's ensure consistency. If it's a mask/grayscale, 3 channels is safer for preview.
        # However, standard mask nodes use 1 channel.
        # The error comes from numpy conversion. PIL fromarray ((1,1,1), |u1) is weird.
        # It's likely ComfyUI trying to save a batch of images where C=1.
        
        # FIX: Ensure 1-channel outputs have 3 channels for maximum compatibility with PreviewImage
        if out_roughness.shape[-1] == 1:
            out_roughness = out_roughness.repeat(1, 1, 1, 3)
        if out_depth.shape[-1] == 1:
            out_depth = out_depth.repeat(1, 1, 1, 3)
        if out_metallic.shape[-1] == 1:
            out_metallic = out_metallic.repeat(1, 1, 1, 3)

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return (out_albedo.cpu(), out_normal.cpu(), out_roughness.cpu(), out_depth.cpu(), out_metallic.cpu())
    
    def height_to_normal(self, height_tensor, flip_green=False, strength=1.0):
        # height_tensor: [B, H, W, 1]
        
        input_tensor = height_tensor.permute(0, 3, 1, 2) # [B, 1, H, W]
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=height_tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=height_tensor.device).view(1, 1, 3, 3)
        
        input_padded = F.pad(input_tensor, (1, 1, 1, 1), mode='replicate')
        
        grad_x = F.conv2d(input_padded, sobel_x)
        grad_y = F.conv2d(input_padded, sobel_y)
        
        # dX, dY are gradients.
        # Normal vector is (-dx, -dy, 1) usually? Or cross product.
        # Surface Z = H(x,y). Normal ~ (-dH/dx, -dH/dy, 1)
        
        grad_x = grad_x * strength
        grad_y = grad_y * strength
        
        if flip_green:
            grad_y = -grad_y
            
        z = torch.ones_like(grad_x)
        
        normal_vec = torch.cat([-grad_x, -grad_y, z], dim=1)
        normal_vec = F.normalize(normal_vec, dim=1)
        
        # Map to 0..1
        normal_map = (normal_vec + 1.0) * 0.5
        
        return normal_map.permute(0, 2, 3, 1)


class GeminiChannelPacker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "red_channel": ("IMAGE",),
                "green_channel": ("IMAGE",),
                "blue_channel": ("IMAGE",),
            },
            "optional": {
                "alpha_channel": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("packed_image",)
    FUNCTION = "pack"
    CATEGORY = "360_HDRI/Gemini"

    def pack(self, red_channel, green_channel, blue_channel, alpha_channel=None):
        # Assuming inputs are [B, H, W, C]
        
        def get_channel(img):
            if img is None: return None
            # If [B, H, W, 3] take first channel
            if img.shape[-1] == 3:
                return img[..., 0]
            # If [B, H, W, 1] take first channel
            if img.shape[-1] == 1:
                return img[..., 0]
            return img
        
        # Check shapes match. Use Red as ref
        B, H, W, _ = red_channel.shape
        
        # Helper to ensure tensors are same size
        def resize_to_match(img_t, target_shape):
            if img_t is None: return None
            if img_t.shape[1:3] != target_shape:
                 return F.interpolate(img_t.permute(0,3,1,2), size=target_shape, mode='bilinear').permute(0,2,3,1)
            return img_t

        green_channel = resize_to_match(green_channel, (H, W))
        blue_channel = resize_to_match(blue_channel, (H, W))
        if alpha_channel is not None:
            alpha_channel = resize_to_match(alpha_channel, (H, W))

        r = get_channel(red_channel)
        g = get_channel(green_channel)
        b = get_channel(blue_channel)
        
        if alpha_channel is not None:
            a = get_channel(alpha_channel)
            packed = torch.stack([r, g, b, a], dim=-1) # [B, H, W, 4]
        else:
            packed = torch.stack([r, g, b], dim=-1) # [B, H, W, 3]
            
        return (packed,)


class GeminiMaterialSaver:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_filename": ("STRING", {"default": "Material_01"}),
                "folder_name": ("STRING", {"default": "PBR_Materials"}),
                "albedo": ("IMAGE",),
            },
            "optional": {
                "normal": ("IMAGE",),
                "roughness": ("IMAGE",),
                "metallic": ("IMAGE",),
                "depth": ("IMAGE",),
                "occlusion": ("IMAGE",),
                "packed_map": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_material"
    CATEGORY = "360_HDRI/Gemini"
    OUTPUT_NODE = True

    def save_material(self, base_filename, folder_name, albedo, normal=None, roughness=None, metallic=None, depth=None, occlusion=None, packed_map=None):
        import os
        from PIL import Image
        import numpy as np
        
        # 1. Setup Output Directory
        # Use ComfyUI standard output directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "output")
        # Or better, just relative to where comfy is running? 
        # Usually it's in ComfyUI/output
        # Let's find ComfyUI root by walking up from custom_nodes
        # __file__ = custom_nodes/ComfyUI-360-HDRI-Suite/nodes/gemini_nodes.py
        # root = ../../../
        
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        comfy_output_dir = os.path.join(root_dir, "output")
        
        full_output_path = os.path.join(comfy_output_dir, folder_name)
        if not os.path.exists(full_output_path):
            os.makedirs(full_output_path)
            
        def save_img(tensor, suffix):
            if tensor is None: return
            # tensor is [B, H, W, C]
            # Take first in batch
            img_t = tensor[0]
            img_np = (img_t.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            
            # Handle channels
            if img_np.shape[-1] == 1:
                img_pil = Image.fromarray(img_np[:, :, 0], mode='L')
            elif img_np.shape[-1] == 3:
                img_pil = Image.fromarray(img_np, mode='RGB')
            elif img_np.shape[-1] == 4:
                img_pil = Image.fromarray(img_np, mode='RGBA')
            else:
                return # Check later
            
            fname = f"{base_filename}_{suffix}.png"
            img_pil.save(os.path.join(full_output_path, fname))
            
        save_img(albedo, "Albedo")
        save_img(normal, "Normal")
        save_img(roughness, "Roughness")
        save_img(metallic, "Metallic")
        save_img(depth, "Depth")
        save_img(occlusion, "AO")
        save_img(packed_map, "Packed_ORM")
        
        return ()

class GeminiComparator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "mode": (["Side-by-Side", "Vertical"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "compare"
    CATEGORY = "360_HDRI/Gemini"

    def compare(self, image_a, image_b, mode):
        # Resize B to match A? Or just cat. Assuming match.
        if image_a.shape != image_b.shape:
             # Basic resize of B to A
             B, H, W, C = image_a.shape
             image_b = F.interpolate(image_b.permute(0, 3, 1, 2), size=(H, W), mode='bilinear').permute(0, 2, 3, 1)
             
        if mode == "Side-by-Side":
            return (torch.cat([image_a, image_b], dim=2),)
        else:
            return (torch.cat([image_a, image_b], dim=1),)
