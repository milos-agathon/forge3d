#!/usr/bin/env python3
"""
Direct test of VF_COLOR_DEBUG_MODE - sets env var in current process.
This eliminates subprocess env inheritance as a variable.
"""
import os
import sys
from pathlib import Path

# Set the env var BEFORE importing forge3d
os.environ["VF_COLOR_DEBUG_MODE"] = "110"  # Mode 110 = pure red
print(f"[TEST] Set VF_COLOR_DEBUG_MODE={os.environ['VF_COLOR_DEBUG_MODE']}")

# Now import forge3d
sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
import forge3d as f3d
from forge3d.terrain_params import make_terrain_params_config
import numpy as np

print(f"[TEST] Imported forge3d from: {f3d.__file__}")

# Create a small heightmap
heightmap = np.random.rand(128, 128).astype(np.float32) * 100

# Create session and renderer
sess = f3d.Session(window=False)
renderer = f3d.TerrainRenderer(sess)

# Create materials and IBL
materials = f3d.MaterialSet.terrain_default()
repo_root = Path(__file__).parents[1]
hdr_path = repo_root / "assets" / "snow_field_4k.hdr"
ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
ibl.set_base_resolution(64)  # Small for speed

# Create colormap
colormap = f3d.Colormap1D.from_stops(
    stops=[(0.0, "#2E8B57"), (100.0, "#FFFFFF")],
    domain=(0.0, 100.0),
)
overlays = [
    f3d.OverlayLayer.from_colormap1d(
        colormap,
        strength=1.0,
        offset=0.0,
        blend_mode="Alpha",
        domain=(0.0, 100.0),
    )
]

# Create params using the config pattern
config = make_terrain_params_config(
    size_px=(200, 150),
    render_scale=1.0,
    msaa_samples=1,
    z_scale=5.0,
    exposure=1.0,
    domain=(0.0, 100.0),
    albedo_mode="material",
    colormap_strength=0.5,
    ibl_enabled=True,
    cam_radius=100.0,
    cam_phi_deg=135.0,
    cam_theta_deg=30.0,
    overlays=overlays,
)
params = f3d.TerrainRenderParams(config)

print("[TEST] Rendering with VF_COLOR_DEBUG_MODE=110 (should be pure red)...")
frame = renderer.render_terrain_pbr_pom(
    material_set=materials,
    env_maps=ibl,
    params=params,
    heightmap=heightmap,
)

# Check if the output is red
rgba = frame.to_numpy()
avg_r = np.mean(rgba[:, :, 0])
avg_g = np.mean(rgba[:, :, 1])
avg_b = np.mean(rgba[:, :, 2])

print(f"[TEST] Average RGB: ({avg_r:.1f}, {avg_g:.1f}, {avg_b:.1f})")

if avg_r > 200 and avg_g < 50 and avg_b < 50:
    print("[PASS] Image is RED - debug mode 110 is working!")
else:
    print("[FAIL] Image is NOT red - debug mode is NOT reaching the shader!")
    print("       Check stderr above for [RENDER_INTERNAL] diagnostic.")

# Save the image for visual inspection
out_path = repo_root / "examples" / "out" / "test_mode_110.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
frame.save(str(out_path))
print(f"[TEST] Saved to: {out_path}")
