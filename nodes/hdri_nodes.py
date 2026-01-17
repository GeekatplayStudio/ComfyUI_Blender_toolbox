# (c) Geekatplay Studio
# ComfyUI-Blender-Toolbox

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
    CATEGORY = "Geekatplay Studio/360 HDRI"

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

            saved = False
            # Method 1: Try OpenCV (Robust for EXR)
            try:
                import cv2
                # OpenCV expects BGR
                img_bgr = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
                saved = cv2.imwrite(filepath, img_bgr)
            except Exception as e:
                print(f"[SaveFakeHDRI] OpenCV save failed: {e}")

            if not saved:
                # Method 2: Try ImageIO
                try:
                    # We try to use imageio. Explicitly providing format="EXR"
                    imageio.imwrite(filepath, img_output, format="EXR")
                except Exception as e:
                    print(f"[SaveFakeHDRI] imageio EXR save failed: {e}. Trying fallback...")
                    # Fallback: try without explicit format or try to catch the PyAV plugin issue
                    try:
                        # Try standard write, hoping it finds a suitable plugin (likely FreeImage if installed)
                        imageio.imwrite(filepath, img_output)
                    except Exception as e2:
                        print(f"[SaveFakeHDRI] All save methods failed. Last error: {e2}")
                        # If everything fails, re-raise the last exception so the user knows
                        raise e2

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
                "pole_stretch_power": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "process"
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def process(self, image, vae, width, height, crop_method, pole_stretch_power=1.0, seed=0):
        # image is [B, H, W, C]
        B, H, W, C = image.shape
        
        # 1. Resize/Crop logic
        img_t = image.permute(0, 3, 1, 2)
        
        if crop_method == "stretch":
            img_resized = torch.nn.functional.interpolate(img_t, size=(height, width), mode="bilinear", align_corners=False)
        elif crop_method == "center":
            target_aspect = width / height
            current_aspect = W / H
            if current_aspect > target_aspect:
                new_w = int(H * target_aspect)
                start_w = (W - new_w) // 2
                img_cropped = img_t[:, :, :, start_w:start_w+new_w]
            else:
                new_h = int(W / target_aspect)
                start_h = (H - new_h) // 2
                img_cropped = img_t[:, :, start_h:start_h+new_h, :]
            img_resized = torch.nn.functional.interpolate(img_cropped, size=(height, width), mode="bilinear", align_corners=False)
        elif crop_method == "pad":
            target_aspect = width / height
            current_aspect = W / H
            if current_aspect > target_aspect:
                scale = width / W
                scaled_h = int(H * scale)
                img_scaled = torch.nn.functional.interpolate(img_t, size=(scaled_h, width), mode="bilinear", align_corners=False)
                pad_h = height - scaled_h
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                img_resized = torch.nn.functional.pad(img_scaled, (0, 0, pad_top, pad_bottom), mode='constant', value=0)
            else:
                scale = height / H
                scaled_w = int(W * scale)
                img_scaled = torch.nn.functional.interpolate(img_t, size=(height, scaled_w), mode="bilinear", align_corners=False)
                pad_w = width - scaled_w
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                img_resized = torch.nn.functional.pad(img_scaled, (pad_left, pad_right, 0, 0), mode='constant', value=0)

        # 2. Apply Pole Stretcher (Non-linear vertical distortion)
        # Only if power is not 1.0
        if pole_stretch_power != 1.0:
            # We need to warp the image vertically.
            # Grid sample expects -1 to 1 range.
            device = image.device
            
            # Create a vertical grid
            # y: [-1, 1] for the target image
            y_rng = torch.linspace(-1, 1, height, device=device)
            x_rng = torch.linspace(-1, 1, width, device=device)
            
            grid_y, grid_x = torch.meshgrid(y_rng, x_rng, indexing='ij')
            
            # Map target y (-1..1) to source y (-1..1) using power law
            # v = (y + 1) / 2  (0 to 1)
            # v_src = 0.5 + sgn(v-0.5) * |2(v-0.5)|^power * 0.5
            # y_src = v_src * 2 - 1
            
            v = (grid_y + 1) / 2.0
            v_centered = v - 0.5 # -0.5 to 0.5
            
            # Function: v_src_centered = sgn(v_c) * |2 * v_c|^power * 0.5
            # If power < 1: compresses near center, expands edges (Stretches Poles)
            # If power > 1: expands near center, compresses edges (Squashes Poles)
            
            sign = torch.sign(v_centered)
            abs_v2 = torch.abs(v_centered * 2.0)
            
            # We use pole_stretch_power directly. 
            # Values < 1.0 (e.g. default 0.5) will stretch the poles (zoom in on zenith/nadir).
            # Values > 1.0 will compress the poles.
            eff_power = pole_stretch_power
            v_src_centered = sign * torch.pow(abs_v2, eff_power) * 0.5
            
            grid_y_src = v_src_centered * 2.0
            
            # Horizontal Stretcher (Equirectangular compensation)
            # latitude: -pi/2 to pi/2
            lat = grid_y * (np.pi / 2.0) 
            cos_lat = torch.cos(lat)
            
            # Use pole_stretch_power to modulate the horizontal squish at the poles.
            # If eff_power < 1 (stretching vertically): we want MORE horizontal squish (h_exp > 1) ??
            # Wait, let's stick to the fundamental math:
            # We want scale_x -> 0 at poles. 
            # cos(lat) -> 0 at poles.
            # scale_x = cos(lat) ^ h_exp.
            # If h_exp = 1, it's standard spherical.
            # If h_exp > 1, it gets closer to 0 faster (sharper squish, more smear).
            # If h_exp < 1, it gets closer to 0 slower (less squish).
            
            # The user wants "more stretched at top/bottom".
            # This usually means reducing the high-frequency noise.
            # So we probably want h_exp >= 1.
            
            # Previous logic: h_exp = max((1/p - 1), 1).
            # If p=0.5 -> 1/0.5 - 1 = 1 -> h_exp=1.
            # If p=1.0 -> 0 -> h_exp=1.
            # So currently it's ALWAYS 1.0 or higher.
            
            # If the user feels it's STILL pinched, we might need to increase h_exp artificially?
            # Or maybe the problem is the Vertical Stretch isn't strong enough?
            # With p=0.5, we stretch vertically.
            
            calc_exp = (1.0 / pole_stretch_power) - 1.0
            
            # Ensure we have at least standard spherical mapping (h_exp=1), 
            # but allow going higher if power is small (big stretch).
            h_exp = max(calc_exp, 1.0)
            
            scale_x = torch.pow(torch.clamp(cos_lat, min=1e-6), h_exp)
            grid_x_src = grid_x * scale_x
            
            # Stack grid
            grid = torch.stack((grid_x_src, grid_y_src), dim=-1).unsqueeze(0) # [1, H, W, 2]
            grid = grid.expand(B, -1, -1, -1)
            
            # Sample
            img_resized = torch.nn.functional.grid_sample(img_resized, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # 3. VAE Encode
        pixels = img_resized.permute(0, 2, 3, 1) # [B, H, W, C]
        t = vae.encode(pixels[:,:,:,:3])
        return ({"samples": t}, )

class Rotate360Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "pitch": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "roll": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate"
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def rotate(self, image, pitch, yaw, roll):
        # image: [B, H, W, C]
        device = image.device
        B, H, W, C = image.shape
        
        # Convert degrees to radians
        pitch_rad = torch.tensor(pitch * np.pi / 180.0, device=device)
        yaw_rad = torch.tensor(yaw * np.pi / 180.0, device=device)
        roll_rad = torch.tensor(roll * np.pi / 180.0, device=device)
        
        # Create output meshgrid
        # u: [-pi, pi], v: [-pi/2, pi/2]
        # x-axis (columns): u
        # y-axis (rows): v
        
        # We want the center of the pixel, so we use linspace carefully
        u_rng = torch.linspace(-np.pi, np.pi, W, device=device)
        v_rng = torch.linspace(-np.pi/2, np.pi/2, H, device=device)
        
        # Meshgrid returns [H, W]
        v_map_out, u_map_out = torch.meshgrid(v_rng, u_rng, indexing='ij')
        
        # Convert output spherical coords (u', v') to Cartesian (x', y', z')
        # Standard physics convention:
        # x = cos(v) * cos(u)
        # y = cos(v) * sin(u)
        # z = sin(v)
        
        x_out = torch.cos(v_map_out) * torch.cos(u_map_out)
        y_out = torch.cos(v_map_out) * torch.sin(u_map_out)
        z_out = torch.sin(v_map_out)
        
        # Stack into [H, W, 3] vectors
        xyz_out = torch.stack((x_out, y_out, z_out), dim=-1) # [H, W, 3]
        
        # Rotation Matrix
        # We need to rotate the "camera" or the "sphere".
        # If we want to "yaw" right, we rotate the sphere left.
        # Let's rotate the vectors.
        # To find where (u',v') comes from, we apply Inverse Rotation to (x',y',z').
        # R_total = R_roll * R_pitch * R_yaw
        # We need R_inv = (R_total)^T
        
        # Yaw (around Z)
        cy = torch.cos(yaw_rad)
        sy = torch.sin(yaw_rad)
        R_yaw = torch.tensor([
            [cy, -sy, 0],
            [sy, cy, 0],
            [0, 0, 1]
        ], device=device)
        
        # Pitch (around Y axis? Or around local X?) 
        # In equirectangular, pitch usually means looking up/down.
        # This corresponds to rotation around the Y axis if X is forward.
        # But our setup: X=cos(u)cos(v). u=0 => X=1 (Forward).
        # So Y-axis is Right (u=90). Z-axis is Up (v=90).
        # Pitching up means rotating around Y_axis?
        # No, if we look up, we rotate around the "Right" vector.
        # Let's assume standard Euler rotations.
        # Pitch connects X and Z. Rotation around Y.
        
        cp = torch.cos(pitch_rad)
        sp = torch.sin(pitch_rad)
        R_pitch = torch.tensor([
            [cp, 0, -sp],
            [0, 1, 0],
            [sp, 0, cp]
        ], device=device)
        
        # Roll (around X axis - Forward)
        cr = torch.cos(roll_rad)
        sr = torch.sin(roll_rad)
        R_roll = torch.tensor([
            [1, 0, 0],
            [0, cr, -sr],
            [0, sr, cr]
        ], device=device)
        
        # Combined Rotation
        # Order matters. Usually Yaw -> Pitch -> Roll.
        R = R_yaw @ R_pitch @ R_roll
        
        # We want Inverse for grid sampling (Target -> Source)
        R_inv = R.T
        
        # Apply rotation to all vectors
        # xyz_out: [H, W, 3] 
        # Flatten to [N, 3]
        xyz_flat = xyz_out.view(-1, 3).T # [3, N]
        
        # Rotate
        xyz_in = R_inv @ xyz_flat # [3, N]
        
        # Reshape back
        xyz_in = xyz_in.T.view(H, W, 3) # [H, W, 3]
        
        x_in = xyz_in[..., 0]
        y_in = xyz_in[..., 1]
        z_in = xyz_in[..., 2]
        
        # Convert back to Spherical (u, v)
        # v = asin(z)
        # u = atan2(y, x)
        
        # Clamp z to [-1, 1] to avoid NaNs
        z_in = torch.clamp(z_in, -1.0, 1.0)
        v_in = torch.asin(z_in)
        u_in = torch.atan2(y_in, x_in)
        
        # Map u, v to localized grid coords [-1, 1]
        # u: [-pi, pi] -> [-1, 1]
        # v: [-pi/2, pi/2] -> [-1, 1]
        
        grid_u = u_in / np.pi
        grid_v = v_in / (np.pi / 2.0)
        
        # Stack for grid_sample: [1, H, W, 2] (x=u, y=v)
        grid = torch.stack((grid_u, grid_v), dim=-1).unsqueeze(0)
        
        # Expand for batch size
        grid = grid.expand(B, -1, -1, -1)
        
        # Input image: [B, H, W, C] -> [B, C, H, W]
        img_nchw = image.permute(0, 3, 1, 2)
        
        # Sample
        # padding_mode='border' creates a streak if we go out of bounds vertically?
        # grid_sample treats -1 as left/top, 1 as right/bottom.
        # But our u wraps around.
        # grid_sample doesn't natively wrap 'circular' on just one axis easily?
        # Actually, standard grid_sample 'zeros', 'border', 'reflection'.
        # For U (Longitude), -pi should wrap to pi.
        # Because we used atan2, u_in is already in [-pi, pi].
        # So we should be sampling safely within the image horizontally.
        # What happens at the jump from pi to -pi?
        # grid_sample interpolates. If we have a pixel at -0.99 and 0.99, it might interpolate across the middle?
        # No, atan2 handles the jump. 
        # The only issue is if grid_sample tries to interpolate between pi and -pi, it sees a jump across the image.
        # But align_corners=True behavior might mitigate or we might see a seam.
        # For a perfect 360 wrap, we might need to pad the input image horizontally?
        
        # Let's pad width by 1 pixel (wrapping) to handle the seam?
        # Actually just using 'border' is usually fine if atan2 is correct.
        
        sampled = torch.nn.functional.grid_sample(img_nchw, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Back to [B, H, W, C]
        out_image = sampled.permute(0, 2, 3, 1)
        
        return (out_image,)

# New Node for Mask Generation
class GeneratePoleMask:
    """
    Generates a radial gradient mask centered at a specific point.
    Useful for masking the poles (zenith/nadir) of an image for targeted inpainting/healing.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "power": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                "invert": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "generate"
    CATEGORY = "Geekatplay Studio/360 HDRI"

    def generate(self, image, center_x, center_y, scale, power, invert):
        # image: [B, H, W, C]
        B, H, W, C = image.shape
        device = image.device
        
        cx = W * center_x
        cy = H * center_y
        
        x = torch.linspace(0, W - 1, W, device=device)
        y = torch.linspace(0, H - 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Euclidean distance from center
        dist = torch.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
        
        # Max distance reference (shortest side * 0.5 * scale)
        ref_size = min(H, W) * 0.5
        max_dist = ref_size * scale
        
        # Normalize: 1.0 at center, 0.0 at max_dist
        mask = 1.0 - (dist / max_dist)
        mask = torch.clamp(mask, 0.0, 1.0)
        
        # Apply falloff power (gamma) -> Higher power = sharper falloff (smaller white circle)
        mask = torch.pow(mask, power)
        
        if invert == "true":
            mask = 1.0 - mask
            
        return (mask,)
