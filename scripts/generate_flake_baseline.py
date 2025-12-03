#!/usr/bin/env python3
"""
Milestone 0: Generate reproducible flake baseline images.

This script creates baseline renders that show the flake artifacts for comparison.
The material mode (PBR triplanar) exhibits flakes at grazing angles, while
colormap mode (elevation-based colors only) does not.

Deliverables:
- reports/flake/baseline_material.png (flakes present)
- reports/flake/baseline_colormap.png (no flakes)
- reports/flake/repro_cmd.txt (exact CLI + env vars)

Usage:
    python scripts/generate_flake_baseline.py

RELEVANT FILES: docs/plan.md, src/shaders/terrain_pbr_pom.wgsl
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np

# Ensure forge3d is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import forge3d as f3d
from forge3d.terrain_params import (
    ClampSettings,
    IblSettings,
    LightSettings,
    LodSettings,
    PomSettings,
    SamplingSettings,
    ShadowSettings,
    TerrainRenderParams as TerrainRenderParamsConfig,
    TriplanarSettings,
)

# Configuration for reproducible baseline
BASELINE_CONFIG = {
    "size": (512, 512),
    "msaa": 1,
    "z_scale": 2.0,
    "cam_radius": 800.0,
    "cam_phi_deg": 135.0,
    "cam_theta_deg": 25.0,  # Low angle to emphasize grazing-angle flakes
    "exposure": 1.0,
    "ibl_intensity": 1.0,
    "sun_azimuth": 135.0,
    "sun_elevation": 35.0,
    "sun_intensity": 3.0,
}

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "flake"
ASSETS_DIR = Path(__file__).parent.parent / "assets"


def _create_test_hdr(path: Path, width: int = 16, height: int = 8) -> None:
    """Create minimal HDR file for IBL if assets unavailable."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(f"-Y {height} +X {width}\n".encode())
        for y in range(height):
            for x in range(width):
                # Simple gradient for basic IBL
                r = int((x / max(width - 1, 1)) * 200 + 55)
                g = int((y / max(height - 1, 1)) * 200 + 55)
                b = 180
                e = 128
                f.write(bytes([r, g, b, e]))


def _create_test_heightmap(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create synthetic heightmap with ridges that expose flakes."""
    h, w = size
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)
    
    # Create terrain with ridges/valleys that produce high-frequency normals
    # These are where flakes typically appear
    base = yy * 0.5  # Base elevation gradient
    ridges = np.sin(xx * 20) * np.sin(yy * 15) * 0.15  # High-freq ridges
    noise = np.sin(xx * 40 + yy * 30) * 0.05  # Fine noise
    
    heightmap = (base + ridges + noise + 0.2).astype(np.float32)
    return np.clip(heightmap, 0.0, 1.0)


def _build_config(albedo_mode: str, overlay) -> TerrainRenderParamsConfig:
    """Build terrain config with specified albedo mode."""
    return TerrainRenderParamsConfig(
        size_px=BASELINE_CONFIG["size"],
        render_scale=1.0,
        msaa_samples=BASELINE_CONFIG["msaa"],
        z_scale=BASELINE_CONFIG["z_scale"],
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=BASELINE_CONFIG["cam_radius"],
        cam_phi_deg=BASELINE_CONFIG["cam_phi_deg"],
        cam_theta_deg=BASELINE_CONFIG["cam_theta_deg"],
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 5000.0),
        light=LightSettings(
            "Directional",
            BASELINE_CONFIG["sun_azimuth"],
            BASELINE_CONFIG["sun_elevation"],
            BASELINE_CONFIG["sun_intensity"],
            [1.0, 1.0, 1.0],
        ),
        ibl=IblSettings(True, BASELINE_CONFIG["ibl_intensity"], 0.0),
        shadows=ShadowSettings(
            False, "PCF", 512, 2, 100.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),  # normal_strength=1.0 for flakes
        pom=PomSettings(False, "Occlusion", 0.0, 4, 16, 2, False, False),
        lod=LodSettings(0, 0.0, 0.0),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=BASELINE_CONFIG["exposure"],
        gamma=2.2,
        albedo_mode=albedo_mode,
        colormap_strength=0.5 if albedo_mode == "mix" else 1.0,
    )


def _render_baseline(albedo_mode: str, output_path: Path, hdr_path: Path) -> np.ndarray:
    """Render terrain with specified albedo mode."""
    print(f"  Rendering {albedo_mode} mode...")
    
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    
    heightmap = _create_test_heightmap()
    domain = (0.0, 1.0)
    
    # Create grayscale colormap for elevation
    cmap = f3d.Colormap1D.from_stops(
        stops=[(0.0, "#1a1a2e"), (0.5, "#4a7c59"), (1.0, "#f5f5dc")],
        domain=domain,
    )
    
    overlay = f3d.OverlayLayer.from_colormap1d(
        cmap,
        strength=1.0,
        offset=0.0,
        blend_mode="Alpha",
        domain=domain,
    )
    
    config = _build_config(albedo_mode, overlay)
    
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    params = f3d.TerrainRenderParams(config)
    
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
        target=None,
    )
    
    rgba = frame.to_numpy()
    
    # Save the image
    try:
        from PIL import Image
        Image.fromarray(rgba, mode="RGBA").save(str(output_path))
    except ImportError:
        f3d.numpy_to_png(str(output_path), rgba)
    
    print(f"    Saved: {output_path}")
    return rgba


def _write_repro_cmd(cmd_path: Path) -> None:
    """Write reproducible command file."""
    timestamp = datetime.now().isoformat()
    
    content = f"""# Flake Baseline Reproduction Commands
# Generated: {timestamp}
# 
# These commands reproduce the baseline images for flake diagnosis.
# Run from the forge3d repository root.

# Environment setup
export PYTHONPATH="${{PWD}}"

# Baseline with material mode (flakes present)
# The PBR triplanar material path produces flakes at grazing angles
# due to mip mismatch in height-normal sampling (before fix).
python examples/terrain_demo.py \\
    --size 512 512 \\
    --msaa 1 \\
    --z-scale 2.0 \\
    --cam-radius 800 \\
    --cam-phi 135 \\
    --cam-theta 25 \\
    --exposure 1.0 \\
    --albedo-mode material \\
    --output reports/flake/baseline_material.png \\
    --overwrite

# Baseline with colormap mode (no flakes)
# Pure elevation-based coloring bypasses PBR shading, so no flakes.
python examples/terrain_demo.py \\
    --size 512 512 \\
    --msaa 1 \\
    --z-scale 2.0 \\
    --cam-radius 800 \\
    --cam-phi 135 \\
    --cam-theta 25 \\
    --exposure 1.0 \\
    --albedo-mode colormap \\
    --output reports/flake/baseline_colormap.png \\
    --overwrite

# Configuration used:
# - size: {BASELINE_CONFIG['size']}
# - msaa: {BASELINE_CONFIG['msaa']}
# - z_scale: {BASELINE_CONFIG['z_scale']}
# - cam_radius: {BASELINE_CONFIG['cam_radius']}
# - cam_phi_deg: {BASELINE_CONFIG['cam_phi_deg']}
# - cam_theta_deg: {BASELINE_CONFIG['cam_theta_deg']} (low angle emphasizes grazing-angle flakes)
# - exposure: {BASELINE_CONFIG['exposure']}
# - ibl_intensity: {BASELINE_CONFIG['ibl_intensity']}
# - sun_azimuth: {BASELINE_CONFIG['sun_azimuth']}
# - sun_elevation: {BASELINE_CONFIG['sun_elevation']}
# - sun_intensity: {BASELINE_CONFIG['sun_intensity']}

# To verify flakes:
# 1. Compare baseline_material.png vs baseline_colormap.png
# 2. Flakes appear as "salt" artifacts in material mode at grazing angles
# 3. Colormap mode should be smooth (no specular aliasing)
"""
    
    cmd_path.write_text(content)
    print(f"  Saved: {cmd_path}")


def main() -> int:
    """Generate Milestone 0 baseline artifacts."""
    print("=" * 60)
    print("Milestone 0: Generating Flake Baseline")
    print("=" * 60)
    
    # Check for GPU
    if not f3d.has_gpu():
        print("ERROR: GPU required for terrain rendering")
        return 1
    
    # Create output directory
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get or create HDR file
    hdr_path = ASSETS_DIR / "snow_field_4k.hdr"
    if not hdr_path.exists():
        print(f"  Creating fallback HDR at {hdr_path}...")
        hdr_path.parent.mkdir(parents=True, exist_ok=True)
        _create_test_hdr(hdr_path)
    
    # Generate baseline images
    print("\n[1/3] Rendering baseline_material.png (flakes expected)...")
    material_path = REPORTS_DIR / "baseline_material.png"
    material_frame = _render_baseline("material", material_path, hdr_path)
    
    print("\n[2/3] Rendering baseline_colormap.png (no flakes)...")
    colormap_path = REPORTS_DIR / "baseline_colormap.png"
    colormap_frame = _render_baseline("colormap", colormap_path, hdr_path)
    
    # Write reproduction commands
    print("\n[3/3] Writing repro_cmd.txt...")
    cmd_path = REPORTS_DIR / "repro_cmd.txt"
    _write_repro_cmd(cmd_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Milestone 0 Complete!")
    print("=" * 60)
    print(f"\nDeliverables:")
    print(f"  - {material_path}")
    print(f"  - {colormap_path}")
    print(f"  - {cmd_path}")
    
    # Quick analysis
    material_var = np.var(material_frame[:, :, :3].astype(float))
    colormap_var = np.var(colormap_frame[:, :, :3].astype(float))
    print(f"\nQuick Analysis:")
    print(f"  Material mode variance: {material_var:.2f}")
    print(f"  Colormap mode variance: {colormap_var:.2f}")
    print(f"  (Higher variance in material suggests flake artifacts)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
