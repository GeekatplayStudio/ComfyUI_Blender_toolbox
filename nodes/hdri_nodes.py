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
