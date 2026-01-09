# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

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

class TerrainPromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "forceInput": True}),
                "terrain_type": (["General", "Mountains", "Canyon", "Desert", "Islands"],),
                "input_view": (["Top Down / Map", "Side View / Photo"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "create_prompt"
    CATEGORY = "360_HDRI/Terrain"

    def create_prompt(self, description, terrain_type, input_view):
        # Stronger keywords for orthographic view to avoid isometric results
        base_prompt = "orthographic top-down depth map, height map, greyscale digital elevation model (DEM), satellite topography, nadir view, flat projection, vertical displacement data, white represents high elevation, black represents low elevation, no shadows, no lighting, 2d texture"
        
        if input_view == "Side View / Photo":
            base_prompt += ", convert to top-down map, flatten perspective, orthogonal projection"
        
        context_prompt = ""
        if terrain_type == "Mountains":
            context_prompt = "rugged mountain range, distinct peaks, ridge lines, glacial valleys, geological formations"
        elif terrain_type == "Canyon":
            context_prompt = "deep canyon network, steep cliffs, river meanders, eroded plateau"
        elif terrain_type == "Desert":
            context_prompt = "rolling sand dunes, aeolian landforms, ripples, arid landscape"
        elif terrain_type == "Islands":
            context_prompt = "volcanic island chain, atolls, coastal shelf, sea level gradient"
            
        # Combine
        positive = f"{base_prompt}, {context_prompt}, {description}"
        
        # Stronger negatives to kill perspective
        negative = "color, rgb, perspective, 3d render, isometric, tilted view, horizon, sky, clouds, cast shadows, sun lighting, glossy, water reflections, noise, text, watermark, labels, grid lines, roads, buildings, trees, vegetation, photo, photorealistic, side view"
        
        if input_view == "Side View / Photo":
            negative += ", ground level view, eye level, landscape photography, horizon line"
            
        return (positive, negative)

class SimpleHeightmapNormalizer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",)}}
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize"
    CATEGORY = "360_HDRI/Terrain"
    
    def normalize(self, images):
        # Normalize batch to 0.0 - 1.0 range to maximize displacement detail
        results = []
        for img in images:
            # Force Grayscale if RGB
            if img.shape[-1] == 3:
                # Luminance formula
                gray = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
                gray = gray.unsqueeze(-1) # [H, W, 1]
                # Repeat to 3 channels for compatibility with other nodes/previews
                img = gray.repeat(1, 1, 3)
                
            min_val = torch.min(img)
            max_val = torch.max(img)
            if max_val > min_val:
                norm = (img - min_val) / (max_val - min_val)
            else:
                norm = img
            results.append(norm)
        return (torch.stack(results),)

class TerrainTexturePromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "forceInput": True}),
                "terrain_type": (["General", "Mountains", "Canyon", "Desert", "Islands"],),
                "input_view": (["Top Down / Map", "Side View / Photo"],),
            },
            "optional": {
                "heightmap_description": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "create_prompt"
    CATEGORY = "360_HDRI/Terrain"

    def create_prompt(self, description, terrain_type, input_view, heightmap_description=""):
        # Keywords for color/albedo texture
        base_prompt = "orthographic nadir view, top-down satellite photography, albedo texture map, photorealistic terrain texture, 8k, highly detailed, seamless texture, flat lighting, no shadows"
        
        if input_view == "Side View / Photo":
            base_prompt += ", transform perspective to top down, reimagine as satellite map, bird's eye view"
            
        context_prompt = ""
        if terrain_type == "Mountains":
            context_prompt = "snowy peaks, rocky ridges, green valleys, alpine vegetation"
        elif terrain_type == "Canyon":
            context_prompt = "red rock, sedimentary layers, dry river bed, scrub brush"
        elif terrain_type == "Desert":
            context_prompt = "golden sand, ripples, dunes, dry landscape, sparse vegetation"
        elif terrain_type == "Islands":
            context_prompt = "tropical vegetation, sandy beaches, turquoise water, coral reefs"
            
        # Combine
        positive = f"{base_prompt}, {context_prompt}, {description}"
        
        if heightmap_description:
            positive += f", based on heightmap features: {heightmap_description}"
        
        # Negatives to avoid heightmap look or perspective
        negative = "heightmap, grayscale, depth map, isometric, perspective, 3d render, tilted, horizon, sky, clouds, strong shadows, text, labels, buildings, roads, blue tint"
        
        if input_view == "Side View / Photo":
            negative += ", ground level view, eye level, landscape photography, horizon line"
            
        return (positive, negative)

class TerrainHeightFieldPromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "forceInput": True}),
                "terrain_type": (["General", "Mountains", "Canyon", "Desert", "Islands"],),
                "detail_level": (["High Frequency (Detailed)", "Low Frequency (Smooth shapes)"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "create_prompt"
    CATEGORY = "360_HDRI/Terrain"

    def create_prompt(self, description, terrain_type, detail_level):
        # Extremely sterile, technical keywords for pure height data
        base_prompt = (
            "16-bit grayscale displacement map, linear height field data, pure elevation gradient, "
            "top-down orthographic projection, nadir view, "
            "black represents lowest elevation (0.0), white represents highest elevation (1.0), "
            "smooth continuous gradients, no lighting, no shadows, no ambient occlusion, no albedo"
        )
        
        context_prompt = ""
        if terrain_type == "Mountains":
            context_prompt = "eroded mountain peaks, ridge lines, glacial valleys, dendritic drainage patterns"
        elif terrain_type == "Canyon":
            context_prompt = "steep canyon walls, river beds, plateaus, terrace erosion"
        elif terrain_type == "Desert":
            context_prompt = "parabolic dunes, ripples, smooth hills, arid flats"
        elif terrain_type == "Islands":
            context_prompt = "volcanic cone, coastal shelf falloff, atoll ring, gradient to sea level"
            
        detail_prompt = ""
        if detail_level == "High Frequency (Detailed)":
            detail_prompt = "high frequency noise, jagged rocks, erosion details, sharp features"
        else:
            detail_prompt = "low frequency shapes, smooth rolling geometry, large scale forms, soft transitions"
            
        positive = f"{base_prompt}, {context_prompt}, {detail_prompt}, {description}"
        
        # Aggressive negation of anything that looks like a "picture"
        negative = (
            "color, rgb, texture, rock texture, grass, snow, water, trees, vegetation, photo, photorealistic, "
            "satellite image, shadows, lighting, sun, shading, ambient occlusion, noise, grain, dither, "
            "perspective, isometric, tilted, horizon, sky, clouds, buildings, roads, labels, text, watermark"
        )
            
        return (positive, negative)

class ColorToHeightmap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "invert": ("BOOLEAN", {"default": False}),
                "auto_levels": ("BOOLEAN", {"default": True}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("heightmap", )
    FUNCTION = "convert"
    CATEGORY = "360_HDRI/Terrain"

    def convert(self, images, invert, auto_levels, gamma):
        results = []
        for img in images:
            # 1. Grayscale Conversion (Luminance)
            # weights: 0.299 R + 0.587 G + 0.114 B
            if img.shape[-1] >= 3:
                gray = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
            else:
                gray = img[..., 0]
                
            # 2. Invert
            if invert:
                gray = 1.0 - gray
                
            # 3. Auto Levels (Normalize full range 0-1)
            if auto_levels:
                min_val = torch.min(gray)
                max_val = torch.max(gray)
                if max_val > min_val:
                    gray = (gray - min_val) / (max_val - min_val)
                    
            # 4. Gamma Correction
            if gamma != 1.0:
                # Add epsilon to avoid log(0) issues if any, though pow shouldn't mind 0
                gray = torch.pow(gray, 1.0 / gamma)
                
            # Expand to 3 channels (H, W) -> (H, W, 3)
            res = gray.unsqueeze(-1).repeat(1, 1, 3)
            results.append(res)
            
        return (torch.stack(results), )
