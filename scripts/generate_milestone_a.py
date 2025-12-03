#!/usr/bin/env python3
"""
Milestone A: Generate sentinel images to verify debug mode branching.

This script renders modes 23-27 with sentinel colors to prove each branch
is actually being executed. If any two modes produce identical images,
the shader branching is broken.

Deliverables:
- reports/flake_diag/mode23_sentinel.png (should be PURE RED)
- reports/flake_diag/mode24_sentinel.png (should be PURE GREEN)
- reports/flake_diag/mode25_sentinel.png (should be PURE BLUE)
- reports/flake_diag/mode26_sentinel.png (grayscale LOD ramp)
- reports/flake_diag/mode27_sentinel.png (grayscale normal_blend ramp)
- reports/flake_diag/debug_grid_v2.png (6-up grid showing all modes)

RELEVANT FILES: docs/plan.md, src/shaders/terrain_pbr_pom.wgsl
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

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

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "flake_diag"
ASSETS_DIR = Path(__file__).parent.parent / "assets"

# Debug modes for Milestone A
DEBUG_MODES = [
    (0, "baseline", "Normal rendering (reference)"),
    (23, "no_specular", "SENTINEL: Pure RED"),
    (24, "no_height_normal", "SENTINEL: Pure GREEN"),
    (25, "ddxddy_normal", "SENTINEL: Pure BLUE"),
    (26, "height_lod", "Grayscale LOD ramp"),
    (27, "normal_blend", "Grayscale blend ramp"),
]

CELL_SIZE = (256, 256)


def _create_test_heightmap(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create synthetic heightmap."""
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


def _create_test_hdr(path: Path) -> None:
    """Create minimal HDR for IBL."""
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y 8 +X 16\n")
        for y in range(8):
            for x in range(16):
                r = int((x / 15) * 200 + 55)
                g = int((y / 7) * 200 + 55)
                b = 180
                e = 128
                f.write(bytes([r, g, b, e]))


def _render_mode(debug_mode: int, hdr_path: Path) -> np.ndarray:
    """Render terrain with specified debug mode."""
    old_value = os.environ.get("VF_COLOR_DEBUG_MODE")
    os.environ["VF_COLOR_DEBUG_MODE"] = str(debug_mode)
    
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
        
        return frame.to_numpy()
        
    finally:
        if old_value is None:
            os.environ.pop("VF_COLOR_DEBUG_MODE", None)
        else:
            os.environ["VF_COLOR_DEBUG_MODE"] = old_value


def _save_image(img: np.ndarray, path: Path) -> None:
    """Save image to file."""
    try:
        from PIL import Image
        Image.fromarray(img, mode="RGBA").save(str(path))
    except ImportError:
        # Fallback
        import struct
        import zlib
        
        def png_chunk(chunk_type, data):
            chunk_len = struct.pack(">I", len(data))
            chunk_crc = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xffffffff)
            return chunk_len + chunk_type + data + chunk_crc
        
        h, w = img.shape[:2]
        raw_data = b""
        for row in img:
            raw_data += b"\x00" + row.tobytes()
        compressed = zlib.compress(raw_data)
        
        with open(path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
            f.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)))
            f.write(png_chunk(b"IDAT", compressed))
            f.write(png_chunk(b"IEND", b""))


def _add_label(img: np.ndarray, label: str) -> np.ndarray:
    """Add text label to image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        pil_img = Image.fromarray(img, mode="RGBA")
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
        
        # Draw label with background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        draw.rectangle([(0, 0), (text_w + 8, text_h + 8)], fill=(0, 0, 0, 200))
        draw.text((4, 4), label, fill=(255, 255, 255, 255), font=font)
        
        return np.array(pil_img)
    except ImportError:
        return img


def _create_grid(images: list[np.ndarray], labels: list[str], cols: int = 3) -> np.ndarray:
    """Create a grid of images."""
    rows = (len(images) + cols - 1) // cols
    h, w = images[0].shape[:2]
    
    grid = np.zeros((rows * h, cols * w, 4), dtype=np.uint8)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        labeled_img = _add_label(img, label)
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = labeled_img
    
    return grid


def _check_pairwise_distinct(images: dict[int, np.ndarray]) -> dict:
    """Check that all mode images are pairwise distinct."""
    modes = sorted(images.keys())
    results = {"pairwise_comparisons": [], "all_distinct": True}
    
    for i, m1 in enumerate(modes):
        for m2 in modes[i+1:]:
            img1 = images[m1]
            img2 = images[m2]
            
            # Compute difference
            diff = np.abs(img1.astype(float) - img2.astype(float))
            max_diff = diff.max()
            mean_diff = diff.mean()
            identical = max_diff < 1.0  # Allow tiny floating point differences
            
            comparison = {
                "mode_a": m1,
                "mode_b": m2,
                "max_diff": float(max_diff),
                "mean_diff": float(mean_diff),
                "identical": identical,
            }
            results["pairwise_comparisons"].append(comparison)
            
            if identical:
                results["all_distinct"] = False
                print(f"  WARNING: Mode {m1} and {m2} are IDENTICAL!")
            else:
                print(f"  OK: Mode {m1} vs {m2}: max_diff={max_diff:.1f}, mean_diff={mean_diff:.2f}")
    
    return results


def main() -> int:
    """Generate Milestone A sentinel images."""
    print("=" * 60)
    print("Milestone A: Generating Sentinel Images")
    print("=" * 60)
    
    if not f3d.has_gpu():
        print("ERROR: GPU required for terrain rendering")
        return 1
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create or use HDR
    hdr_path = ASSETS_DIR / "snow_field_4k.hdr"
    if not hdr_path.exists():
        print(f"  Creating fallback HDR...")
        _create_test_hdr(hdr_path)
    
    # Render all modes
    images = {}
    labels = []
    
    for mode, name, desc in DEBUG_MODES:
        print(f"\n[{mode}] Rendering mode {mode} ({name})...")
        img = _render_mode(mode, hdr_path)
        images[mode] = img
        labels.append(f"Mode {mode}: {name}")
        
        # Save individual image for modes 23-27
        if mode >= 23:
            path = REPORTS_DIR / f"mode{mode}_sentinel.png"
            _save_image(img, path)
            print(f"  Saved: {path}")
            
            # Analyze color content
            r_mean = img[:, :, 0].mean()
            g_mean = img[:, :, 1].mean()
            b_mean = img[:, :, 2].mean()
            print(f"  Color analysis: R={r_mean:.1f}, G={g_mean:.1f}, B={b_mean:.1f}")
    
    # Create grid
    print("\n" + "-" * 60)
    print("Creating debug grid...")
    grid_images = [images[m] for m, _, _ in DEBUG_MODES]
    grid = _create_grid(grid_images, labels, cols=3)
    grid_path = REPORTS_DIR / "debug_grid_v2.png"
    _save_image(grid, grid_path)
    print(f"  Saved: {grid_path}")
    
    # Check pairwise distinctness
    print("\n" + "-" * 60)
    print("Checking pairwise distinctness for modes 23-27...")
    flake_images = {m: images[m] for m in [23, 24, 25, 26, 27]}
    results = _check_pairwise_distinct(flake_images)
    
    # Summary
    print("\n" + "=" * 60)
    print("Milestone A Results")
    print("=" * 60)
    
    if results["all_distinct"]:
        print("\n✅ SUCCESS: All modes 23-27 are pairwise distinct!")
        print("   Shader branching is working correctly.")
    else:
        print("\n❌ FAILURE: Some modes are identical!")
        print("   Shader branching is broken. Check debug_mode uniform.")
        return 1
    
    print(f"\nDeliverables:")
    for mode, name, _ in DEBUG_MODES:
        if mode >= 23:
            print(f"  - {REPORTS_DIR / f'mode{mode}_sentinel.png'}")
    print(f"  - {grid_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
