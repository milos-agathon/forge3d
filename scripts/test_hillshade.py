#!/usr/bin/env python3
"""Test hillshade renderer to match reference image."""

import numpy as np
from PIL import Image
import sys
sys.path.insert(0, 'python')

from forge3d.render import render_raster

# Load DEM
dem_path = "assets/Gore_Range_Albers_1m.tif"

# Hypsometric colormap matching reference (red-brown shadows to golden highlights)
colormap = ["#6a4030", "#7c4838", "#8c5545", "#9c6350", "#a87058", "#b48060", "#c09068", "#d0a878", "#d8b888", "#e0c898"]

# Render with hillshade pipeline
result = render_raster(
    dem_path,
    size=(1920, 1080),
    renderer="hillshade",
    palette=colormap,
    lighting_type="lambertian",
    lighting_azimuth=315.0,  # NW light direction
    lighting_elevation=37.0,
    shadow_enabled=True,
    shadow_intensity=0.7,
    contrast_pct=1.0,
    gamma=1.0,
    equalize=False,
    exaggeration=3.0,
    camera_phi=135.0,
    camera_theta=35.0,
    camera_distance=1400.0,
)

# Save result
output_path = "examples/output/terrain_hillshade.png"
Image.fromarray((np.clip(result, 0, 1) * 255).astype(np.uint8)).save(output_path)
print(f"Saved to {output_path}")
