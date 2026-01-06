# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

import torch
import numpy as np
import imageio
import os
import folder_paths

class SaveFakeHDRI:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI_HDRI"}),
                "exposure_boost": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 0.1, "max": 5.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("full_path",)
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "360_HDRI"

    def save_exr(self, images, filename_prefix, exposure_boost, gamma):
        results = list()
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])

        last_filepath = ""

        for i, image in enumerate(images):
            # 1. Convert tensor to numpy
            img_np = image.cpu().numpy() # [H, W, C]
            
            # 2. Linearize the color (remove gamma 2.2)
            # Avoid divide by zero or log errors if needed, but power is safe for +ve
            img_linear = np.power(img_np, gamma)
            
            # 3. Create a mask of the brightest pixels (above 0.8 threshold)
            # We use the max value across channels to detect brightness
            max_vals = np.max(img_np, axis=2, keepdims=True)
            mask = max_vals > 0.8
            
            # 4. Multiply the masked bright pixels by exposure_boost
            # We apply this to the linear image
            # We only boost the pixels in the mask. 
            # To blend it better, one might want a soft mask, but prompt asks for this logic.
            
            # Create a boost map: 1.0 everywhere, exposure_boost where mask is True
            boost_map = np.ones_like(img_linear)
            # Broadcast mask to cover channels
            mask_broadcast = np.broadcast_to(mask, img_linear.shape)
            
            # Apply boost
            img_linear[mask_broadcast] *= exposure_boost
            
            # 5. Save as .EXR file (32-bit float)
            file = f"{filename}_{counter:05}_.exr"

            # Verify HDR range
            max_val = np.max(img_linear)
            print(f"[SaveFakeHDRI] Saved {file} with Max Value: {max_val:.2f} (Exposure Boost: {exposure_boost})")
            filepath = os.path.join(full_output_folder, file)
            
            # Ensure data is float32
            img_output = img_linear.astype(np.float32)
            
            # imageio.imwrite with .exr extension requires the 'freeimage' or 'pyav' plugin usually,
            # but 'imageio[ffmpeg]' might be trying to use ffmpeg which can be tricky for EXR.
            # Let's explicitly try to use the 'exr' format if available, or fallback to 'freeimage'.
            # However, imageio's default behavior for .exr often relies on the 'freeimage' plugin.
            # The error "expected bytes, NoneType found" in pyav suggests it's trying to use pyav (ffmpeg)
            # and failing to initialize the stream properly for EXR.
            
            try:
                imageio.imwrite(filepath, img_output, format="EXR")
            except Exception:
                # Fallback: try without explicit format, or try 'freeimage' if installed
                # If pyav is hijacking .exr, we might need to force a different plugin or just use 'tif' as a fallback?
                # No, user wants EXR.
                # Let's try to force the 'freeimage' plugin if possible, or just let imageio decide but handle the error.
                # The error comes from pyav. Let's try to disable pyav for this write.
                imageio.imwrite(filepath, img_output, plugin="freeimage")
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1
            last_filepath = filepath

        return { "ui": { "images": results }, "result": (last_filepath,) }

class ImageTo360Latent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "crop_method": (["center", "stretch", "pad"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "360_HDRI"

    def process(self, image, vae, width, height, crop_method):
        # image is [B, H, W, C]
        B, H, W, C = image.shape
        
        # 1. Resize/Crop
        # We need to process each image in the batch, but usually batch size is 1 for this workflow.
        # We'll use torch.nn.functional.interpolate for resizing.
        
        # Permute to [B, C, H, W] for torch operations
        img_t = image.permute(0, 3, 1, 2)
        
        if crop_method == "stretch":
            img_resized = torch.nn.functional.interpolate(img_t, size=(height, width), mode="bilinear", align_corners=False)
        
        elif crop_method == "center":
            # Calculate aspect ratios
            target_aspect = width / height
            current_aspect = W / H
            
            if current_aspect > target_aspect:
                # Image is wider than target: Crop width
                new_w = int(H * target_aspect)
                start_w = (W - new_w) // 2
                img_cropped = img_t[:, :, :, start_w:start_w+new_w]
            else:
                # Image is taller than target: Crop height
                new_h = int(W / target_aspect)
                start_h = (H - new_h) // 2
                img_cropped = img_t[:, :, start_h:start_h+new_h, :]
                
            img_resized = torch.nn.functional.interpolate(img_cropped, size=(height, width), mode="bilinear", align_corners=False)
            
        elif crop_method == "pad":
            # Calculate aspect ratios
            target_aspect = width / height
            current_aspect = W / H
            
            if current_aspect > target_aspect:
                # Image is wider: Pad height
                # Resize width to target width, scale height proportionally
                scale = width / W
                scaled_h = int(H * scale)
                img_scaled = torch.nn.functional.interpolate(img_t, size=(scaled_h, width), mode="bilinear", align_corners=False)
                
                # Pad height
                pad_h = height - scaled_h
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                # Pad: (left, right, top, bottom)
                img_resized = torch.nn.functional.pad(img_scaled, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
                
            else:
                # Image is taller: Pad width
                scale = height / H
                scaled_w = int(W * scale)
                img_scaled = torch.nn.functional.interpolate(img_t, size=(height, scaled_w), mode="bilinear", align_corners=False)
                
                # Pad width
                pad_w = width - scaled_w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                img_resized = torch.nn.functional.pad(img_scaled, (pad_left, pad_right, 0, 0), mode='constant', value=0)

        # 2. VAE Encode
        # Permute back to [B, H, W, C] for VAE? No, VAE usually expects [B, C, H, W] or [B, H, W, C]?
        # ComfyUI VAE encode expects [B, H, W, C] in the range 0..1?
        # Let's check standard VAEEncode node.
        # It takes pixels.
        
        # Convert back to [B, H, W, C]
        pixels = img_resized.permute(0, 2, 3, 1)
        
        # VAE Encode
        # The VAE encode method in ComfyUI returns a tensor [B, C, H, W]
        # But the KSampler expects a dictionary {"samples": tensor}
        
        t = vae.encode(pixels[:,:,:,:3])
        return ({"samples": t}, )
