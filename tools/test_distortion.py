
import torch
import numpy as np
import math

def simulate_distortion(height, width, pole_stretch_power):
    # Simulate the grid sample logic
    
    # Grid: y increases downwards?
    # PyTorch grid_sample default: (-1, -1) is Top-Left. (1, 1) is Bottom-Right.
    
    y_rng = torch.linspace(-1, 1, height)
    x_rng = torch.linspace(-1, 1, width)
    
    grid_y, grid_x = torch.meshgrid(y_rng, x_rng, indexing='ij')
    
    # 1. Vertical Logic
    # v goes 0 (Top) to 1 (Bottom)
    v = (grid_y + 1) / 2.0
    v_centered = v - 0.5 
    
    sign = torch.sign(v_centered)
    
    # User uses 0.5. Goal: Stretch poles. (Push center content out).
    # Power = 0.5.
    # Inverse = 2.0.
    inv_power = 1.0 / pole_stretch_power
    
    abs_v2 = torch.abs(v_centered * 2.0)
    
    # formula
    v_src_centered = sign * torch.pow(abs_v2, inv_power) * 0.5
    grid_y_src = v_src_centered * 2.0
    
    # 2. Horizontal Logic (Proposed)
    # lat: -pi/2 (Top) to pi/2 (Bottom)?
    # grid_y: -1 (Top) -> lat = -pi/2. Correct.
    lat = grid_y * (math.pi / 2.0)
    cos_lat = torch.cos(lat)
    cos_lat = torch.clamp(cos_lat, min=1e-6)
    
    # h_exp
    h_exp = (1.0 / pole_stretch_power) - 1.0 # 2-1 = 1.0
    scale_x = torch.pow(cos_lat, h_exp)
    
    grid_x_src = grid_x * scale_x
    
    # Check Top Row (index 0)
    # grid_y[0] = -1.0. 
    # v[0] = 0.
    # v_centered[0] = -0.5.
    # abs_v2 = 1.0.
    # pow(1, 2) = 1.
    # v_src_c = -0.5.
    # grid_y_src = -1.0.
    # Output Top maps to Input Top. OK.
    
    # cos_lat at -pi/2 is 0.
    # scale_x = 0.
    # grid_x_src = 0.
    # For all x in top row, we sample input x=0.
    # x=0 is the CENTER column of input image.
    
    # Check a bit down. y = -0.9.
    # v = 0.05. v_c = -0.45.
    # abs*2 = 0.9.
    # 0.9^2 = 0.81.
    # v_src_c = -0.5 * 0.81 = -0.405.
    # grid_y_src = -0.81.
    # Output -0.9 maps to Input -0.81.
    # -0.81 is closer to equator than -0.9.
    # So we sample lower down. 
    # Center is being expanded. Correct.
    
    return grid_y_src, grid_x_src

print("Testing Distortion with Power=0.5")
h, w = 10, 10
gy, gx = simulate_distortion(h, w, 0.5)

print("\nTop Row Y Sample Coordinates:")
print(gy[0]) # Should be -1.0
print("\nTop Row X Sample Coordinates:")
print(gx[0]) # Should be all 0.0 (sampling center column)

print("\nMid Top Row (idx 2) Y:")
print(gy[2])
print("Mid Top Row X:")
print(gx[2])

