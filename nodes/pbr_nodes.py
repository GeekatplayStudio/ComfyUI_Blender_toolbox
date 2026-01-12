# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox

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
    CATEGORY = "Geekatplay Studio/360 HDRI/PBR"

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
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"

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
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"
    
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
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"

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
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"

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
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"

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

class TerrainErosionPromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "erosion_type": (["Hydraulic (Water flow)", "Thermal (Weathering)", "Glacial (Scouring)", "Aeolian (Wind/Sand)", "Coastal (Waves)", "Terrace (Agricultural)"],),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "description": ("STRING", {"multiline": True}),
                "input_positive": ("STRING", {"forceInput": True}),
                "input_negative": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "create_prompt"
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"

    def create_prompt(self, erosion_type, strength=1.0, description="", input_positive="", input_negative=""):
        base_prompt = "top-down orthographic heightmap, erosion simulation, weathering effects, geological detail"
        
        erosion_context = ""
        if erosion_type == "Hydraulic (Water flow)":
            erosion_context = "hydraulic erosion, dendritic drainage channels, river networks, branching water flow, gully formation, sharp ridges, sediment fan"
        elif erosion_type == "Thermal (Weathering)":
            erosion_context = "thermal weathering, rock crumbling, scree slopes, talus piles, smooth eroded peaks, diffused landscape features"
        elif erosion_type == "Glacial (Scouring)":
            erosion_context = "glacial erosion, U-shaped valleys, cirques, tarns, scoured bedrock, lateral moraines, deep scarring, smooth polished surfaces"
        elif erosion_type == "Aeolian (Wind/Sand)":
            erosion_context = "aeolian erosion, wind sculpted rocks, yardangs, ventifacts, sand ripples, dune fields, directional weathering"
        elif erosion_type == "Coastal (Waves)":
            erosion_context = "coastal erosion, sea cliffs, wave cut platforms, sea stacks, jagged coastline, tidal weathering"
        elif erosion_type == "Terrace (Agricultural)":
            erosion_context = "agricultural terraces, step farming, contour lines, man-made landscaping, leveled platforms, rice paddies"

        # Combine logic:
        # Start with input_positive if present
        parts = []
        if input_positive:
            parts.append(input_positive)
        
        if strength > 0.0:
            erosion_content = f"{base_prompt}, {erosion_context}"
            if strength != 1.0:
                parts.append(f"({erosion_content}:{strength})")
            else:
                parts.append(erosion_content)
        
        if description:
            parts.append(description)

        positive = ", ".join([p for p in parts if p])
        
        # Negative to keep it clean map data
        node_negative = "vegetation, trees, water, snow, ice, man-made structures (unless terrace), buildings, roads, text, labels, noise, artifacts, perspective, side view, 3d render, shadows"
        
        neg_parts = []
        if input_negative:
            neg_parts.append(input_negative)
            
        if strength > 0.0:
            if strength != 1.0:
                neg_parts.append(f"({node_negative}:{strength})")
            else:
                neg_parts.append(node_negative)

        negative = ", ".join([p for p in neg_parts if p])

        return (positive, negative)

class MaterialTexturePromptMaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "material_type": (["Rock", "Soil", "Sand", "Grass", "Forest Floor", "Water", "Swamp Water", "Snow", "Ice", "Brick", "Wood", "Metal", "Polished Metal", "Concrete", "Plaster", "Asphalt", "Ceramic", "Marble", "Fabric", "Leather", "Organic", "Sci-Fi"],),
                "description": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "create_prompt"
    CATEGORY = "Geekatplay Studio/360 HDRI/Texture"

    def create_prompt(self, material_type, description):
        base_prompt = "top down high resolution texture, seamless pattern, albedo material map, flat lighting, photorealistic, 4k, incredibly detailed"

        material_context = ""
        if material_type == "Rock":
            material_context = "stone surface, natural rock formation, rough texture, geological detail"
        elif material_type == "Soil":
            material_context = "rich soil, dirt, earthy texture, loam, clumps"
        elif material_type == "Sand":
            material_context = "sand dunes, ripples, beach sand, desert texture, granular"
        elif material_type == "Grass":
            material_context = "green grass, lawn, vegetation, meadow, turf"
        elif material_type == "Forest Floor":
            material_context = "fallen leaves, twigs, roots, pine needles, mossy ground"
        elif material_type == "Water":
            material_context = "water surface, waves, ripples, liquid, fluid simulation, caustic"
        elif material_type == "Swamp Water":
            material_context = "murky water, algae, duckweed, stagnant water, sludge, green tint"
        elif material_type == "Snow":
            material_context = "fresh snow, powder, cold, icy patches, white surface"
        elif material_type == "Ice":
            material_context = "frozen ice, cracks, bubbles, slippery surface, blue tint, glacial"
        elif material_type == "Brick":
            material_context = "brick wall, masonry, construction material, repetitive pattern, mortar joints"
        elif material_type == "Wood":
            material_context = "wood grain, timber, planks, natural bark, carpentry material, knots"
        elif material_type == "Metal":
            material_context = "metallic surface, rust, patina, brushed steel, industrial material, oxidation"
        elif material_type == "Polished Metal":
            material_context = "gold, silver, platinum, high polish, reflective, luxury material"
        elif material_type == "Concrete":
             material_context = "weathered concrete, cement surface, urban texture, crack details, aggregate"
        elif material_type == "Plaster":
             material_context = "stucco wall, plaster texture, painted surface, grunge details, rough cast"
        elif material_type == "Asphalt":
            material_context = "road surface, bitumen, tar, rough street, pebbles"
        elif material_type == "Ceramic":
            material_context = "bathroom tiles, kitchen tiles, glazed surface, grout lines, geometric pattern"
        elif material_type == "Marble":
            material_context = "polished marble, stone veins, luxury stone, smooth surface"
        elif material_type == "Fabric":
            material_context = "cloth weave, textile pattern, threads, soft material, high thread count"
        elif material_type == "Leather":
             material_context = "leather grain, tanned skin, animal hide, expensive material, pores"
        elif material_type == "Organic":
            material_context = "biological texture, skin, scales, cellular pattern, organic growth, microscopic detail"
        elif material_type == "Sci-Fi":
            material_context = "spaceship hull, greebles, tech panels, futuristic metal, vents"

        positive = f"{base_prompt}, {description}, {material_context}"
        
        negative = "perspective, 3d render, shadows, occlusion, frame, border, text, watermark, depth of field, blur, distorted, noisy, low resolution, spherical, sphere, ball"

        return (positive, negative)


