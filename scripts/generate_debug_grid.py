#!/usr/bin/env python3
"""
Milestone 1: Generate debug mode grid image and log.

This script creates a 2x3 grid showing all flake diagnosis debug modes:
- Row 0: Mode 0 (baseline), Mode 23 (no specular), Mode 24 (no height normal)
- Row 1: Mode 25 (ddxddy normal), Mode 26 (height LOD), Mode 27 (normal blend)

Deliverables:
- reports/flake/debug_grid.png (2×3 grid of modes)
- reports/flake/debug_mode_log.txt (captured stdout with resolved mode values)

Usage:
    python scripts/generate_debug_grid.py

RELEVANT FILES: docs/plan.md, src/shaders/terrain_pbr_pom.wgsl
"""
from __future__ import annotations

import os
import sys
import io
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

import numpy as np

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

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "flake"
ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Debug modes for the grid
DEBUG_MODES = [
    (0, "Baseline"),
    (23, "No Specular"),
    (24, "No Height Normal"),
    (25, "ddxddy Normal"),
    (26, "Height LOD"),
    (27, "Normal Blend"),
]

# Per-cell resolution
CELL_SIZE = (256, 256)


def _create_test_heightmap(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create synthetic heightmap with ridges that expose flakes."""
    h, w = size
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)
    
    base = yy * 0.5
    ridges = np.sin(xx * 20) * np.sin(yy * 15) * 0.15
    noise = np.sin(xx * 40 + yy * 30) * 0.05
    
    heightmap = (base + ridges + noise + 0.2).astype(np.float32)
    return np.clip(heightmap, 0.0, 1.0)


def _build_config(overlay) -> TerrainRenderParamsConfig:
    """Build terrain config."""
    return TerrainRenderParamsConfig(
        size_px=CELL_SIZE,
        render_scale=1.0,
        msaa_samples=1,
        z_scale=2.0,
        cam_target=[0.0, 0.0, 0.0],
        cam_radius=800.0,
        cam_phi_deg=135.0,
        cam_theta_deg=25.0,
        cam_gamma_deg=0.0,
        fov_y_deg=55.0,
        clip=(0.1, 5000.0),
        light=LightSettings("Directional", 135.0, 35.0, 3.0, [1.0, 1.0, 1.0]),
        ibl=IblSettings(True, 1.0, 0.0),
        shadows=ShadowSettings(
            False, "PCF", 512, 2, 100.0, 1.0, 0.8, 0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9
        ),
        triplanar=TriplanarSettings(6.0, 4.0, 1.0),
        pom=PomSettings(False, "Occlusion", 0.0, 4, 16, 2, False, False),
        lod=LodSettings(0, 0.0, 0.0),
        sampling=SamplingSettings("Linear", "Linear", "Linear", 1, "Repeat", "Repeat", "Repeat"),
        clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        overlays=[overlay],
        exposure=1.0,
        gamma=2.2,
        albedo_mode="material",
        colormap_strength=0.5,
    )


def _render_debug_mode(debug_mode: int, hdr_path: Path, log_buffer: io.StringIO) -> np.ndarray:
    """Render terrain with specified debug mode."""
    # Set environment variable
    old_value = os.environ.get("VF_COLOR_DEBUG_MODE")
    os.environ["VF_COLOR_DEBUG_MODE"] = str(debug_mode)
    
    log_buffer.write(f"[DEBUG] Setting VF_COLOR_DEBUG_MODE={debug_mode}\n")
    
    try:
        session = f3d.Session(window=False)
        renderer = f3d.TerrainRenderer(session)
        material_set = f3d.MaterialSet.terrain_default()
        
        heightmap = _create_test_heightmap()
        domain = (0.0, 1.0)
        
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
        
        config = _build_config(overlay)
        
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
        
        log_buffer.write(f"[DEBUG] Mode {debug_mode}: rendered {rgba.shape}, mean={np.mean(rgba[:,:,:3]):.2f}\n")
        
        return rgba
        
    finally:
        # Restore environment variable
        if old_value is None:
            os.environ.pop("VF_COLOR_DEBUG_MODE", None)
        else:
            os.environ["VF_COLOR_DEBUG_MODE"] = old_value


def _add_label(img: np.ndarray, label: str, mode: int) -> np.ndarray:
    """Add text label to image (simple approach using PIL if available)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        pil_img = Image.fromarray(img, mode="RGBA")
        draw = ImageDraw.Draw(pil_img)
        
        # Use default font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
        
        # Draw label background
        text = f"Mode {mode}: {label}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = 4
        draw.rectangle(
            [(0, 0), (text_width + padding * 2, text_height + padding * 2)],
            fill=(0, 0, 0, 180)
        )
        draw.text((padding, padding), text, fill=(255, 255, 255, 255), font=font)
        
        return np.array(pil_img)
    except ImportError:
        # Fallback: just return original image
        return img


def _create_grid(images: list[np.ndarray], labels: list[tuple[int, str]], cols: int = 3) -> np.ndarray:
    """Create a grid of images with labels."""
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    
    grid = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
    
    for i, (img, (mode, label)) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        
        labeled_img = _add_label(img, label, mode)
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = labeled_img
    
    return grid


def main() -> int:
    """Generate Milestone 1 debug grid artifacts."""
    print("=" * 60)
    print("Milestone 1: Generating Debug Mode Grid")
    print("=" * 60)
    
    if not f3d.has_gpu():
        print("ERROR: GPU required for terrain rendering")
        return 1
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    hdr_path = ASSETS_DIR / "snow_field_4k.hdr"
    if not hdr_path.exists():
        print(f"ERROR: HDR file not found: {hdr_path}")
        return 1
    
    log_buffer = io.StringIO()
    log_buffer.write(f"# Debug Mode Log - {datetime.now().isoformat()}\n")
    log_buffer.write("# This file captures the resolved debug mode values\n\n")
    
    images = []
    for mode, name in DEBUG_MODES:
        print(f"  Rendering mode {mode} ({name})...")
        img = _render_debug_mode(mode, hdr_path, log_buffer)
        images.append(img)
    
    # Create grid
    print("\nCreating debug grid...")
    grid = _create_grid(images, DEBUG_MODES, cols=3)
    
    # Save grid
    grid_path = REPORTS_DIR / "debug_grid.png"
    try:
        from PIL import Image
        Image.fromarray(grid, mode="RGBA").save(str(grid_path))
    except ImportError:
        f3d.numpy_to_png(str(grid_path), grid)
    print(f"  Saved: {grid_path}")
    
    # Save log
    log_path = REPORTS_DIR / "debug_mode_log.txt"
    
    # Add analysis to log
    log_buffer.write("\n# Analysis:\n")
    for i, (mode, name) in enumerate(DEBUG_MODES):
        img = images[i]
        variance = np.var(img[:, :, :3].astype(float))
        mean_val = np.mean(img[:, :, :3])
        unique_colors = len(np.unique(img[:, :, :3].reshape(-1, 3), axis=0))
        log_buffer.write(f"Mode {mode} ({name}): variance={variance:.2f}, mean={mean_val:.2f}, unique_colors={unique_colors}\n")
        
        # Check for flatness (mode 26/27 should NOT be flat)
        if mode in [26, 27]:
            is_flat = variance < 100
            log_buffer.write(f"  -> {'WARNING: FLAT' if is_flat else 'OK: Non-uniform'}\n")
    
    log_path.write_text(log_buffer.getvalue())
    print(f"  Saved: {log_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Milestone 1 Debug Grid Complete!")
    print("=" * 60)
    print(f"\nDeliverables:")
    print(f"  - {grid_path}")
    print(f"  - {log_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
