# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def seamless_padding(input, padding, axis):
    # input: [B, C, H, W]
    # padding: int or tuple. Conv2d padding is (padH, padW)
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding[0], padding[1]

    # F.pad tuple is (left, right, top, bottom)
    if axis == 'x':
        # Pad X circular
        x = F.pad(input, (pad_w, pad_w, 0, 0), mode='circular')
        # Pad Y with zeros (standard conv behavior)
        x = F.pad(x, (0, 0, pad_h, pad_h), mode='constant', value=0)
        return x
    elif axis == 'y':
        # Pad Y circular
        x = F.pad(input, (0, 0, pad_h, pad_h), mode='circular')
        # Pad X with zeros
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode='constant', value=0)
        return x
    else: # both
        return F.pad(input, (pad_w, pad_w, pad_h, pad_h), mode='circular')

class SeamlessConv2d(nn.Module):
    """
    A wrapper around torch.nn.Conv2d that applies circular padding before convolution.
    This enables seamless tiling logic while tricking the model inspector (via properties)
    to treat it as a standard Conv2d layer (fixing AttributeErrors in KSampler).
    """
    def __init__(self, original_conv, axis):
        super().__init__()
        self.original_conv = original_conv
        self.axis = axis
    
    @property
    def padding(self):
        return self.original_conv.padding

    @property
    def stride(self):
        return self.original_conv.stride
        
    @property
    def kernel_size(self):
        return self.original_conv.kernel_size
        
    @property
    def dilation(self):
        return self.original_conv.dilation
        
    def forward(self, input):
        # We need to apply circular padding manually before the convolution
        
        # 1. Get padding from original conv
        padding = self.original_conv.padding
        padding_mode = self.original_conv.padding_mode
        
        # Calculate expected output size to handle any potential mismatches
        # This ensures we don't break skip connections
        H, W = input.shape[2], input.shape[3]
        
        # Handle int/tuple parameters
        def get_param(p):
            return (p, p) if isinstance(p, int) else (p[0], p[1])
            
        pad_h, pad_w = get_param(padding)
        kernel_h, kernel_w = get_param(self.original_conv.kernel_size)
        stride_h, stride_w = get_param(self.original_conv.stride)
        dilation_h, dilation_w = get_param(self.original_conv.dilation)
        
        # Standard Conv2d output size formula
        out_h = int((H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)
        out_w = int((W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1)

        # 2. Apply circular padding
        padded_input = seamless_padding(input, padding, self.axis)
        
        # 3. Run the original convolution
        # We temporarily set padding to 0 because we padded manually
        try:
            self.original_conv.padding = (0, 0)
            self.original_conv.padding_mode = 'zeros' # Correct mode for manual padding
            
            result = self.original_conv(padded_input)
            
            # 4. Crop to expected size if necessary
            # Sometimes manual padding + stride can result in slightly different sizes 
            # or extra pixels if not perfectly divisible, or if we just need to be safe.
            curr_h, curr_w = result.shape[2], result.shape[3]
            
            if curr_h != out_h or curr_w != out_w:
                # print(f"SeamlessConv2d mismatch: Got {curr_h}x{curr_w}, expected {out_h}x{out_w}")
                
                # Center crop behavior usually makes sense contextually
                h_diff = curr_h - out_h
                w_diff = curr_w - out_w
                
                if h_diff >= 0 and w_diff >= 0:
                    h_start = h_diff // 2
                    w_start = w_diff // 2
                    result = result[:, :, h_start:h_start+out_h, w_start:w_start+out_w]
                else:
                    # If we are somehow smaller, we have a bigger problem (should not happen with circular pad logic)
                    pass

            return result

        finally:
            # CRITICAL: Restore original state
            self.original_conv.padding = padding
            self.original_conv.padding_mode = padding_mode

class SimpleSeamlessTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "axis": (["x", "y", "both"],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "360_HDRI"

    def patch(self, model, axis):
        # Clone the model wrapper (ModelPatcher)
        m = model.clone()
        
        # We need to patch the underlying diffusion model.
        # WARNING: This modifies the loaded model in memory, affecting other workflows if not careful.
        # A safer way is to use model.set_model_patch, but that's for weights/functions.
        # Replacing layers is structural.
        
        # To do this properly without permanently breaking the model for other nodes:
        # We should ideally restore it. But we can't easily.
        # We will proceed with in-place replacement on the diffusion_model.
        
        def replace_layers(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    # Skip 1x1 convolutions (kernel_size=1) as they don't need padding usually
                    if child.kernel_size == (1, 1) or child.kernel_size == 1:
                        continue
                        
                    # Check if already patched (avoid double patching)
                    if isinstance(child, SeamlessConv2d):
                        child.axis = axis # Update axis
                        continue

                    # Replace
                    new_layer = SeamlessConv2d(child, axis)
                    setattr(module, name, new_layer)
                else:
                    replace_layers(child)

        # Apply to the diffusion model
        replace_layers(m.model.diffusion_model)
        
        return (m, )

class SeamlessTileVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("VAE",),
                "axis": (["x", "y", "both"],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "360_HDRI"

    def patch(self, vae, axis):
        # Clone the VAE to avoid modifying the original
        v = copy.deepcopy(vae)
        
        def replace_layers(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    # Skip 1x1 convolutions
                    if child.kernel_size == (1, 1) or child.kernel_size == 1:
                        continue
                        
                    # Check if already patched
                    if isinstance(child, SeamlessConv2d):
                        child.axis = axis
                        continue

                    # Replace
                    new_layer = SeamlessConv2d(child, axis)
                    setattr(module, name, new_layer)
                else:
                    replace_layers(child)

        # Apply to the VAE model (first_stage_model)
        # Note: ComfyUI VAE object wraps the actual model in first_stage_model
        if hasattr(v, "first_stage_model"):
            replace_layers(v.first_stage_model)
        
        return (v, )

class Heal360Seam:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blend_amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "heal"
    CATEGORY = "360_HDRI"

    def heal(self, image, blend_amount):
        # image: [B, H, W, C]
        B, H, W, C = image.shape
        blend_pixels = int(W * blend_amount)
        
        if blend_pixels == 0:
            return (image,)

        # Extract strips
        left_strip = image[:, :, :blend_pixels, :]
        right_strip = image[:, :, -blend_pixels:, :]
        
        # Calculate target (average of both edges)
        target = (left_strip + right_strip) / 2.0
        
        # Create alpha masks
        # For Left: 1.0 at edge (index 0), 0.0 at interior (index N)
        alpha_left = torch.linspace(1, 0, blend_pixels, device=image.device).view(1, 1, blend_pixels, 1)
        
        # For Right: 0.0 at interior (index 0), 1.0 at edge (index N)
        alpha_right = torch.linspace(0, 1, blend_pixels, device=image.device).view(1, 1, blend_pixels, 1)
        
        # Apply blending
        image_copy = image.clone()
        
        # Blend Left strip towards Target
        image_copy[:, :, :blend_pixels, :] = left_strip * (1 - alpha_left) + target * alpha_left
        
        # Blend Right strip towards Target
        image_copy[:, :, -blend_pixels:, :] = right_strip * (1 - alpha_right) + target * alpha_right
        
        return (image_copy,)
