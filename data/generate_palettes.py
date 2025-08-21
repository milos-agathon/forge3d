#!/usr/bin/env python3
"""Generate built-in colormap palette data for Rust inclusion."""

import numpy as np
import forge3d as plt
from matplotlib import cm

def generate_palette(colormap_name: str, filename: str):
    """Generate 256 RGBA8 values for a matplotlib colormap."""
    cmap = cm.get_cmap(colormap_name)
    
    # Generate 256 evenly spaced values from 0 to 1
    values = np.linspace(0, 1, 256)
    
    # Sample colormap and convert to RGBA8
    colors = cmap(values)  # Returns RGBA float [0,1]
    rgba8 = (colors * 255).astype(np.uint8)
    
    # Write as binary file for Rust include_bytes!
    with open(filename, 'wb') as f:
        f.write(rgba8.tobytes())
    
    print(f"Generated {filename}: {len(rgba8)} RGBA8 values")

if __name__ == "__main__":
    generate_palette('viridis', 'viridis_256.rgba')
    generate_palette('magma', 'magma_256.rgba')
    
    # Custom terrain colormap: blue->green->brown->white
    terrain_colors = np.array([
        [0.0, 0.0, 0.5, 1.0],    # Deep blue (water)
        [0.0, 0.3, 0.8, 1.0],    # Light blue  
        [0.0, 0.5, 0.2, 1.0],    # Dark green
        [0.2, 0.7, 0.1, 1.0],    # Light green
        [0.6, 0.4, 0.2, 1.0],    # Brown
        [0.8, 0.8, 0.8, 1.0],    # Light gray (peaks)
        [1.0, 1.0, 1.0, 1.0],    # White (snow)
    ])
    
    # Interpolate to 256 values
    from scipy.interpolate import interp1d
    x_old = np.linspace(0, 1, len(terrain_colors))
    x_new = np.linspace(0, 1, 256)
    
    interp_func = interp1d(x_old, terrain_colors.T, kind='linear')
    terrain_256 = interp_func(x_new).T
    terrain_rgba8 = (terrain_256 * 255).astype(np.uint8)
    
    with open('terrain_256.rgba', 'wb') as f:
        f.write(terrain_rgba8.tobytes())
    
    print("Generated terrain_256.rgba: 256 RGBA8 values")
    print("Run this script and commit the .rgba files for Rust inclusion")