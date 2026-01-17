# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox
import folder_paths
import os
import torch
import numpy as np
from PIL import Image, ImageOps

class FluxTerrainPromptGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "erosion_type": (["Hydraulic (Water flow)", "Thermal (Weathering)", "Glacial (Scouring)", 
                                  "Aeolian (Wind/Sand)", "Coastal (Waves)", "Terrace (Agricultural)", "None"],),
                "erosion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "season": (["None", "Summer", "Winter", "Autumn", "Spring"], {"default": "None"}),
                "color_scheme": (["Grayscale (Heightmap)", "Rainbow (Red Low - Violet High)"], {"default": "Grayscale (Heightmap)"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "generate_flux_prompt"
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"

    def generate_flux_prompt(self, description, erosion_type, erosion_strength, season, color_scheme):
        # Base constraints for Flux Heightmap
        # Flux handles natural language well, but for technical maps we need specific technical tokens.

        # 1. Base Definition (The "What")
        # Aggressively remove "visual" terms and focus on "data" terms to prevent baked lighting.
        if color_scheme == "Rainbow (Red Low - Violet High)":
            base = (
                "False color topographic map, Rainbow elevation heatmap. "
                "Terrain elevation data visualization using a full spectrum rainbow gradient: "
                "pure red is lowest elevation, transitioning through orange, yellow, green, blue, to dark violet at highest elevation. "
                "Flat 2D top-down orthographic projection. "
                "Absolutely no lighting, no shadows, no shading, no ambient occlusion, no sun direction. "
                "Mathematical surface representation."
            )
        else:
            base = (
                "Raw grayscale 16-bit displacement map, height map texture. "
                "Pure elevation data visualization: white is high altitude, black is low altitude. "
                "Flat 2D top-down orthographic projection. "
                "Absolutely no lighting, no shadows, no shading, no ambient occlusion, no sun direction. "
                "Mathematical surface representation."
            )

        # 2. Erosion Logic
        erosion_text = ""
        if erosion_type != "None":
            strength_desc = ""
            if erosion_strength < 0.4: strength_desc = "subtle"
            elif erosion_strength < 0.8: strength_desc = "mild"
            elif erosion_strength < 1.4: strength_desc = "strong"
            else: strength_desc = "extreme"

            if erosion_type == "Hydraulic (Water flow)":
                erosion_text = f"Features {strength_desc} dendritic drainage patterns and river channels defined by elevation drop." 
            elif erosion_type == "Thermal (Weathering)":
                erosion_text = f"Features {strength_desc} smoothed gradients representing thermal weathering."
            elif erosion_type == "Glacial (Scouring)":
                erosion_text = f"Features {strength_desc} U-shaped valley gradients and cirque depressions."
            elif erosion_type == "Aeolian (Wind/Sand)":
                erosion_text = f"Features {strength_desc} directional gradient patterns from wind erosion."
            elif erosion_type == "Coastal (Waves)":
                erosion_text = f"Features {strength_desc} shelf drop-off gradients."
            elif erosion_type == "Terrace (Agricultural)":
                erosion_text = f"Features {strength_desc} stepped elevation plateaus."

        # 3. Seasonal Logic (Affects surface features in Flux)
        # Season often adds "snow" which is white (good for height) but also texture/shadows.
        # We rephrase to keep it geometric.
        season_text = ""
        if season != "None":
             if season == "Winter":
                 season_text = "with sharp, jagged alpine peaks"
             elif season == "Summer":
                 season_text = "with clear, distinct ridge definition"
             else:
                 season_text = "" # Ignore others to avoid 'leaves' or 'mud' textures

        # 4. Assembly
        positive = f"{base} {erosion_text} {description} {season_text}."

        # 5. Negative Prompt
        neg_parts = [
            "shadows, cast shadow, sun, lighting, self-shadowing, ambient occlusion, shading",
            "3d render, photorealistic, photography, satellite image",
            "perspective, isometric, side view, tilted, horizon",
            "clouds, trees, vegetation, grass, water reflections",
            "noise, dither, gradient banding, text, labels"
        ]
        
        # Only ban color if we are in Grayscale mode
        if color_scheme != "Rainbow (Red Low - Violet High)":
             neg_parts.append("color, rgb")

        negative = ", ".join(neg_parts)

        return (positive, negative)

class FluxOptionalImageRef:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "enabled": ("BOOLEAN", {"default": False}),
                "vae": ("VAE",),
                "image_path": ("STRING", {"default": "", "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_ref"
    CATEGORY = "Geekatplay Studio/360 HDRI/Terrain"

    def apply_ref(self, conditioning, enabled, vae, image_path):
        if not enabled:
            return (conditioning,)
        
        if not image_path or not os.path.exists(image_path):
            print(f"[FluxOptionalImageRef] Warning: Image path '{image_path}' not found or empty. Skipping reference injection.")
            return (conditioning,)

        try:
            # 1. Load Image
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            
            # 2. Convert to Tensor (Compatible with VAE Encode)
            # Ref: comfy.utils.image_to_tensor logic approx
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,] # [1, H, W, C]
            
            # 3. VAE Encode
            # We need to call the VAE encode method.
            # Usually: vae.encode(pixels) returns a latent dictionary.
            # Note: vae object structure depends on ComfyUI version, but .encode is standard.
            # However, VAEEncode node does: return ({"samples": vae.encode(pixels[:,:,:,:3])}, )
            
            latents = vae.encode(image[:,:,:,:3])
            
            # 4. Apply Reference Latent Logic
            # Standard ReferenceLatent logic (simplified):
            # c = copy.deepcopy(conditioning)
            # n = [c_concat_tensor, latent_samples]
            # but usually it's passed via 'c_concat' key in the conditioning dict.
            # In Flux/Std workflows, ReferenceLatent usually works by Concat.
            
            c_out = []
            for t in conditioning:
                d = t[1].copy()
                
                # Logic from 'ReferenceLatent' (ComfyUI Core) approx:
                # if 'c_concat' not in d: d['c_concat'] = []
                # else: d['c_concat'] = list(d['c_concat']) # copy list?
                # Actually, ReferenceLatent usually just replaces or appends?
                # Let's rely on standard practice: 'model_cond' vs 'ref'.
                # Wait, ReferenceInpainter/UniPC uses c_concat. ReferenceOnly uses a hack.
                # The user's workflow uses the node type "ReferenceLatent".
                # This node is likely from "comfy_extras" or a specific custom pack.
                # If it's the standard "ReferenceLatent" node, it concatenates.
                # Let's try attempting to locate the 'c_concat' key.
                
                # Note: 'conditioning' is a list of [tensor_text, dict]
                
                # Create a simple concatenated condition
                # The latent sample comes as a tensor [B, C, H, W]
                samples = latents
                
                # Retrieve existing concat or create new
                # In ComfyUI, 'c_concat' usually holds [latent_image] (as a tensor)
                # But sometimes it holds [mask]
                
                # To be safe and mimicking "ReferenceLatent" node behavior:
                # It typically adds the latent to the conditioning for UNet to consult.
                # Because we can't easily import the exact node class to delegate, 
                # we will assume the standard Reference Latent behavior which is:
                # Appending the latent to 'c_concat' 
                
                prev_concat = d.get('c_concat', None)
                if prev_concat is None:
                    d['c_concat'] = samples
                else:
                    # If it exists, we probably concatenate on batch dim or channel dim? 
                    # Usually reference is a separate signal.
                    # If we don't know the exact logic of the specific "ReferenceLatent" node the user has, 
                    # we might fail.
                    # However, if 'ReferenceLatent' is simply doing `concat`, it's replacing it or merging.
                    # Given this is a "Top Down" terrain, likely map injection.
                    
                    # Implementation of standard ControlNet/GLIGEN uses c_concat.
                    # Let's assume simplest safe: Just set it.
                    d['c_concat'] = samples

                n = [t[0], d]
                c_out.append(n)

            return (c_out,)

        except Exception as e:
            print(f"[FluxOptionalImageRef] Error loading/encoding image: {e}")
            return (conditioning,)

