import torch
import torch.nn.functional as F
import numpy as np

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
    RETURN_NAMES = ("albedo_out", "normal_map", "roughness_map", "height_map", "metallic_map")
    FUNCTION = "extract"
    CATEGORY = "360_HDRI/Gemini"

    def extract(self, albedo_image, fidelity):
        # albedo_image: [B, H, W, C]
        
        # 1. Albedo (Just pass through)
        albedo_out = albedo_image.clone()
        
        # 2. Height Map (Grayscale)
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
        
        if fidelity == "High":
            # "High" fidelity might imply we do some fake enhancements
            # Simply sharpening the height map?
            pass
            
        return (albedo_out, normal_map, roughness_map, height_map, metallic_map)
    
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
        # Assuming inputs are [B, H, W, C] (take C=0) or [B, H, W]
        
        def get_channel(img):
            if img.dim() == 4:
                return img[..., 0]
            return img
        
        r = get_channel(red_channel)
        g = get_channel(green_channel)
        b = get_channel(blue_channel)
        
        # Ensure sizes match (use R as reference)
        # For simplicity, assume they match.
        
        if alpha_channel is not None:
            a = get_channel(alpha_channel)
            packed = torch.stack([r, g, b, a], dim=-1)
        else:
            packed = torch.stack([r, g, b], dim=-1) # [B, H, W, 3]
            
        return (packed,)


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
