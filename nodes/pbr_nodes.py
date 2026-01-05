import torch
import numpy as np

class SimplePBRGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "roughness_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "normal_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("roughness", "metallic", "normal")
    FUNCTION = "generate"
    CATEGORY = "360_HDRI/PBR"

    def generate(self, images, roughness_intensity, normal_strength):
        # images is [B, H, W, C]
        
        # 1. Grayscale for processing
        # Luminance: 0.299 R + 0.587 G + 0.114 B
        gray = images[..., 0] * 0.299 + images[..., 1] * 0.587 + images[..., 2] * 0.114
        gray = gray.unsqueeze(-1) # [B, H, W, 1]
        
        # 2. Roughness
        # Simple heuristic: Invert gray? Or just use gray?
        # Usually, darker parts of a texture might be smoother (cracks) or rougher depending on material.
        # Let's just output the grayscale adjusted by intensity.
        # We can mix the grayscale with the intensity value.
        # Let's assume brighter = rougher for now (like concrete), but allow intensity to scale it.
        roughness = gray * roughness_intensity
        # Clamp
        roughness = torch.clamp(roughness, 0.0, 1.0)
        
        # 3. Metallic
        # Hard to guess from just color. Let's output a flat map based on some heuristic or just black (dielectric)
        # For now, let's just return a dark map (non-metal)
        metallic = torch.full_like(gray, 0.0) # Non-metal by default
        
        # 4. Normal Map
        # Use Sobel operator
        # We need to permute to [B, C, H, W] for conv2d
        input_tensor = gray.permute(0, 3, 1, 2)
        
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        
        # Pad to keep size
        input_padded = torch.nn.functional.pad(input_tensor, (1, 1, 1, 1), mode='replicate')
        
        grad_x = torch.nn.functional.conv2d(input_padded, sobel_x)
        grad_y = torch.nn.functional.conv2d(input_padded, sobel_y)
        
        # Normal Z is usually 1.0 (or scaled)
        # Normal map = (x, y, z) normalized -> mapped to 0..1
        # Strength factor
        grad_x = grad_x * normal_strength
        grad_y = grad_y * normal_strength
        z = torch.ones_like(grad_x)
        
        normal_vec = torch.cat([grad_x, grad_y, z], dim=1) # [B, 3, H, W]
        # Normalize
        normal_vec = torch.nn.functional.normalize(normal_vec, dim=1)
        
        # Map -1..1 to 0..1
        normal_map = (normal_vec + 1.0) * 0.5
        
        # Permute back to [B, H, W, C]
        normal_map = normal_map.permute(0, 2, 3, 1)
        
        # Expand roughness/metallic to 3 channels for preview consistency
        roughness = roughness.repeat(1, 1, 1, 3)
        metallic = metallic.repeat(1, 1, 1, 3)
        
        return (roughness, metallic, normal_map)
