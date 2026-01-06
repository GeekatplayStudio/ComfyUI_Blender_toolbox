# (c) Geekatplay Studio
# ComfyUI-360-HDRI-Suite

import torch
import numpy as np
import random

class TextureScrambler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "tiles_x": ("INT", {"default": 4, "min": 1, "max": 16}),
                "tiles_y": ("INT", {"default": 4, "min": 1, "max": 16}),
                "scramble": ("BOOLEAN", {"default": True}),
                "random_rotate": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scramble_image"
    CATEGORY = "360_HDRI/Texture"

    def scramble_image(self, images, tiles_x, tiles_y, scramble, random_rotate, seed):
        # images: [B, H, W, C]
        if not scramble and not random_rotate:
            return (images,)
            
        B, H, W, C = images.shape
        
        # Calculate tile sizes
        tile_h = H // tiles_y
        tile_w = W // tiles_x
        
        # Crop to exact multiple
        H_new = tile_h * tiles_y
        W_new = tile_w * tiles_x
        images = images[:, :H_new, :W_new, :]
        
        results = []
        
        random.seed(seed)
        
        for b in range(B):
            img = images[b] # [H, W, C]
            
            # Extract tiles
            tiles = []
            for y in range(tiles_y):
                for x in range(tiles_x):
                    y_start = y * tile_h
                    y_end = y_start + tile_h
                    x_start = x * tile_w
                    x_end = x_start + tile_w
                    
                    tile = img[y_start:y_end, x_start:x_end, :]
                    tiles.append(tile)
            
            # Scramble
            if scramble:
                random.shuffle(tiles)
                
            # Rotate
            if random_rotate:
                rotated_tiles = []
                for tile in tiles:
                    rot = random.choice([0, 1, 2, 3]) # 0, 90, 180, 270
                    if rot > 0:
                        # torch.rot90 requires [C, H, W] or [..., H, W]
                        # tile is [H, W, C] -> [C, H, W]
                        t = tile.permute(2, 0, 1)
                        t = torch.rot90(t, k=rot, dims=[1, 2])
                        # Back to [H, W, C]
                        t = t.permute(1, 2, 0)
                        
                        # Handle aspect ratio changes if not square?
                        # If we rotate 90, H and W swap. If tiles aren't square, this breaks grid.
                        # For simplicity, we only rotate 180 if non-square, or force resize.
                        # OR: We just skip rotation if non-square and warn?
                        # Let's check squareness.
                        if tile_h != tile_w and rot % 2 != 0:
                            # Can't rotate 90/270 in place without changing grid shape
                            # Fallback to 180 or 0
                            if rot == 1 or rot == 3:
                                t = torch.rot90(tile.permute(2,0,1), k=2, dims=[1,2]).permute(1,2,0)
                        
                        rotated_tiles.append(t)
                    else:
                        rotated_tiles.append(tile)
                tiles = rotated_tiles
            
            # Reconstruct
            # We need to stack rows, then stack columns
            # tiles is list of length tiles_x * tiles_y
            
            rows = []
            for y in range(tiles_y):
                row_tiles = tiles[y*tiles_x : (y+1)*tiles_x]
                row = torch.cat(row_tiles, dim=1) # Cat along Width
                rows.append(row)
                
            new_img = torch.cat(rows, dim=0) # Cat along Height
            results.append(new_img)
            
        return (torch.stack(results),)
