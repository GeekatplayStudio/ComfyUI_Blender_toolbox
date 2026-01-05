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
    def __init__(self, original_conv, axis):
        super().__init__()
        self.original_conv = original_conv
        self.axis = axis
        
    def forward(self, input):
        # We need to apply circular padding manually before the convolution
        # But we must respect the original convolution's parameters
        
        # 1. Get padding from original conv
        # Note: original_conv.padding is a tuple (padH, padW) or int
        padding = self.original_conv.padding
        
        # DEBUG PRINTS
        # print(f"SeamlessConv2d: Input {input.shape}, Padding {padding}, Stride {self.original_conv.stride}, Kernel {self.original_conv.kernel_size}")

        # 2. Apply circular padding
        # This increases the spatial dimensions
        padded_input = seamless_padding(input, padding, self.axis)
        
        # 3. Run the original convolution
        # IMPORTANT: We must temporarily set the original conv's padding to 0
        # because we already padded the input manually.
        # If we don't, it will pad AGAIN (with zeros), changing the size and ruining the seamless effect.
        
        original_padding_mode = self.original_conv.padding_mode
        original_padding_val = self.original_conv.padding
        
        try:
            # We need to handle tuple padding correctly
            # If padding is 'same' or 'valid', we can't just set it to (0,0) easily if it's a string
            # But usually in SDXL it's tuple or int.
            
            if isinstance(original_padding_val, str):
                 # If it's 'same', we need to calculate manual padding to match 'same' behavior
                 # But we are doing circular padding.
                 # For now, let's assume standard int/tuple padding.
                 pass
            
            self.original_conv.padding = (0, 0)
            self.original_conv.padding_mode = 'zeros'
            
            # Run the convolution
            result = self.original_conv(padded_input)
            
            # DEBUG
            # print(f"SeamlessConv2d: Result raw {result.shape}")

            # 4. Fix Output Shape Mismatch (The cause of the error)
            # The error "size of tensor a (126) must match ... (122)" happens because
            # our manual padding might be slightly different than what the model expects
            # for the residual connection (skip connection).
            # If we padded too much, the output is bigger than the input to the residual add.
            
            # We need to crop the result to match the expected output size.
            # Expected output size is usually the same as input size (for stride=1)
            # or input/2 (for stride=2).
            
            # Let's calculate expected output size based on the ORIGINAL input size (before padding)
            # Input: [B, C, H, W]
            # Output H = (H + 2*padH - kernelH) / strideH + 1
            
            # But wait, we want the output to be seamless.
            # If we crop it, we lose the seamlessness?
            # No, the seamlessness comes from the input padding being circular.
            # The output size should match what the original layer WOULD have produced
            # if it had circular padding support.
            
            # The issue is likely that we are padding *more* than the original layer did?
            # No, we use self.original_padding.
            
            # Let's look at the error again: 126 vs 122. Difference is 4.
            # This suggests we have 2 extra pixels on each side (2*2=4).
            # This happens if we pad manually AND the conv layer pads again.
            # But we set padding=(0,0).
            
            # Wait! If the original layer had padding=(1,1), we added 1 pixel border.
            # Then we ran conv with padding=(0,0).
            # The output size should be correct.
            
            # UNLESS... the original layer was using padding_mode='replicate' or something
            # that we overrode? No, we handle that.
            
            # The most likely cause is that `self.original_conv` is being called by something else
            # or our `try...finally` block is not thread-safe or re-entrant safe?
            # No, ComfyUI is single threaded for execution usually.
            
            # Actually, there is a subtle issue with `F.pad`.
            # If we use `seamless_padding`, we add pixels.
            # If the original conv had padding=1, kernel=3, stride=1.
            # Input 128. Pad -> 130. Conv(valid) -> 128. Correct.
            
            # What if the original conv had padding=0?
            # Input 128. Pad -> 128. Conv(valid) -> 126 (loss of 2).
            # But if padding was 0, we shouldn't pad?
            # seamless_padding checks this.
            
            # Let's look at the error: 126 vs 122.
            # If we produced 126, but expected 122.
            # That means we are 4 pixels too large.
            
            # This happens if we pad (2,2) when we shouldn't?
            # Or if we pad (1,1) and the conv ALSO pads (1,1)?
            # That would give +2 input size -> +2 output size?
            
            # Ah! `self.original_conv.padding = (0, 0)` modifies the OBJECT.
            # If this object is shared (reused) in the model, and we are in a loop...
            # But we restore it in `finally`.
            
            # WAIT. The error is `RuntimeError: The size of tensor a (126) must match the size of tensor b (122)`.
            # This is a residual connection addition failure: `x + layer(x)`.
            # `x` is 122. `layer(x)` is 126.
            # So our layer is producing an output that is TOO BIG.
            
            # Why is it too big?
            # Maybe `seamless_padding` is adding padding when `original_padding` is 0?
            # No, `seamless_padding` uses `original_padding`.
            
            # Maybe the original layer uses `padding='same'`?
            # If so, `original_padding` might be a string or calculated dynamically?
            # In SDXL, most are explicit ints.
            
            # Let's try to force the output to match the input size if stride is 1.
            # This is a hack but might solve the "off by a few pixels" error.
            
            stride = self.original_conv.stride
            if isinstance(stride, int):
                stride = (stride, stride)

            if stride == (1, 1):
                if result.shape[2] != input.shape[2] or result.shape[3] != input.shape[3]:
                    # Crop center
                    diff_h = result.shape[2] - input.shape[2]
                    diff_w = result.shape[3] - input.shape[3]
                    
                    if diff_h > 0 or diff_w > 0:
                        # Crop symmetrically
                        h_start = diff_h // 2
                        w_start = diff_w // 2
                        h_end = result.shape[2] - (diff_h - h_start)
                        w_end = result.shape[3] - (diff_w - w_start)
                        result = result[:, :, h_start:h_end, w_start:w_end]
                        # print(f"SeamlessConv2d: Cropped to {result.shape}")
            
            return result

        finally:
            # Restore original state to avoid side effects if this layer is used elsewhere
            self.original_conv.padding = original_padding_val
            self.original_conv.padding_mode = original_padding_mode

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
