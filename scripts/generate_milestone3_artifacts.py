#!/usr/bin/env python3
"""
Milestone 3: Generate bandlimit/fade strategy artifacts.

This script creates artifacts showing the LOD-based normal blend fade:
- normal_blend_curve.png: Plot of blend vs LOD
- mode27_normal_blend.png: Visual proof the fade is applied
- p5_meta.json: Parameters used for the fade

The fade strategy is: normal_blend = base_blend * (1.0 - saturate((lod - lod_lo) / (lod_hi - lod_lo)))
- LOD 0-1: full height-normal contribution
- LOD 1-4: fade out height-normal
- LOD 4+: no height-normal (pure geometric normal)

RELEVANT FILES: docs/plan.md, src/shaders/terrain_pbr_pom.wgsl
"""
from __future__ import annotations

import os
import sys
import json
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

REPORTS_DIR = Path(__file__).parent.parent / "reports" / "flake"
ASSETS_DIR = Path(__file__).parent.parent / "assets"

# LOD fade parameters (from shader)
LOD_FADE_START = 1.0  # lod_lo: LOD below this gets full blend
LOD_FADE_END = 4.0    # lod_hi: LOD above this gets no blend
BLEND_MIN = 0.0       # Minimum blend at high LOD
BLEND_MAX = 1.0       # Maximum blend at low LOD (from normal_blend_base)


def _create_test_heightmap(size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create synthetic heightmap with features."""
    h, w = size
    y = np.linspace(0, 1, h)
    x = np.linspace(0, 1, w)
    xx, yy = np.meshgrid(x, y)
    
    base = yy * 0.5
    ridges = np.sin(xx * 20) * np.sin(yy * 15) * 0.15
    noise = np.sin(xx * 40 + yy * 30) * 0.05
    
    heightmap = (base + ridges + noise + 0.2).astype(np.float32)
    return np.clip(heightmap, 0.0, 1.0)


def _build_config(overlay, size: tuple[int, int] = (512, 512)) -> TerrainRenderParamsConfig:
    """Build terrain config."""
    return TerrainRenderParamsConfig(
        size_px=size,
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
        f3d.numpy_to_png(str(path), img)


def _compute_blend_curve() -> tuple[np.ndarray, np.ndarray]:
    """Compute the normal blend curve as function of LOD.
    
    Returns (lod_values, blend_values) arrays.
    """
    # Sample LOD from 0 to 6 (covers full range)
    lod = np.linspace(0, 6, 100)
    
    # Compute fade: 1.0 - saturate((lod - lod_lo) / (lod_hi - lod_lo))
    fade = 1.0 - np.clip((lod - LOD_FADE_START) / (LOD_FADE_END - LOD_FADE_START), 0, 1)
    
    # Normal blend = base_blend * fade (assuming base_blend = 1.0)
    blend = BLEND_MAX * fade
    
    return lod, blend


def _create_blend_curve_plot(lod: np.ndarray, blend: np.ndarray, output_path: Path) -> None:
    """Create a plot of the blend curve."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(lod, blend, 'b-', linewidth=2, label='Normal Blend')
        ax.axvline(x=LOD_FADE_START, color='g', linestyle='--', label=f'LOD Start ({LOD_FADE_START})')
        ax.axvline(x=LOD_FADE_END, color='r', linestyle='--', label=f'LOD End ({LOD_FADE_END})')
        
        ax.fill_between(lod, blend, alpha=0.3)
        
        ax.set_xlabel('Height LOD Level', fontsize=12)
        ax.set_ylabel('Normal Blend Factor', fontsize=12)
        ax.set_title('Milestone 3: LOD-Based Normal Blend Fade\n'
                     f'(lod_lo={LOD_FADE_START}, lod_hi={LOD_FADE_END})', fontsize=14)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        ax.annotate('Full height-normal\ncontribution', xy=(0.5, 0.9), fontsize=10,
                   ha='center', color='green')
        ax.annotate('Fade region', xy=(2.5, 0.5), fontsize=10,
                   ha='center', color='blue')
        ax.annotate('No height-normal\n(pure geometric)', xy=(5, 0.1), fontsize=10,
                   ha='center', color='red')
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150)
        plt.close()
        
        return True
        
    except ImportError:
        # Fallback: create a simple table image
        return False


def _create_blend_table_image(lod: np.ndarray, blend: np.ndarray, output_path: Path) -> None:
    """Create a simple table image of the blend curve (fallback if matplotlib unavailable)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create white image
        width, height = 600, 400
        img = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except:
            font = ImageFont.load_default()
            font_title = font
        
        # Title
        draw.text((width // 2 - 150, 20), "Normal Blend vs LOD Curve", fill=(0, 0, 0), font=font_title)
        
        # Parameters
        draw.text((50, 60), f"Parameters:", fill=(0, 0, 0), font=font)
        draw.text((50, 85), f"  lod_lo (fade start): {LOD_FADE_START}", fill=(0, 0, 0), font=font)
        draw.text((50, 110), f"  lod_hi (fade end):   {LOD_FADE_END}", fill=(0, 0, 0), font=font)
        draw.text((50, 135), f"  blend_min: {BLEND_MIN}", fill=(0, 0, 0), font=font)
        draw.text((50, 160), f"  blend_max: {BLEND_MAX}", fill=(0, 0, 0), font=font)
        
        # Table header
        draw.text((50, 200), "LOD Level    Normal Blend", fill=(0, 0, 0), font=font)
        draw.line([(50, 220), (300, 220)], fill=(0, 0, 0))
        
        # Sample points
        sample_lods = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        y_pos = 230
        for l in sample_lods:
            fade = 1.0 - max(0, min(1, (l - LOD_FADE_START) / (LOD_FADE_END - LOD_FADE_START)))
            b = BLEND_MAX * fade
            draw.text((50, y_pos), f"  {l:4.1f}         {b:.3f}", fill=(0, 0, 0), font=font)
            y_pos += 18
        
        img.save(str(output_path))
        
    except ImportError:
        # Last resort: create numpy array
        table = np.ones((400, 600, 4), dtype=np.uint8) * 255
        _save_image(table, output_path)


def main() -> int:
    """Generate Milestone 3 bandlimit/fade artifacts."""
    print("=" * 60)
    print("Milestone 3: Generating Bandlimit/Fade Strategy Artifacts")
    print("=" * 60)
    
    if not f3d.has_gpu():
        print("ERROR: GPU required for terrain rendering")
        return 1
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    hdr_path = ASSETS_DIR / "snow_field_4k.hdr"
    if not hdr_path.exists():
        print(f"ERROR: HDR file not found: {hdr_path}")
        return 1
    
    # 1. Create blend curve plot
    print("\n[1/3] Creating normal blend curve plot...")
    lod, blend = _compute_blend_curve()
    curve_path = REPORTS_DIR / "normal_blend_curve.png"
    
    if not _create_blend_curve_plot(lod, blend, curve_path):
        print("  matplotlib not available, creating table instead...")
        _create_blend_table_image(lod, blend, curve_path)
    print(f"  Saved: {curve_path}")
    
    # 2. Render mode 27 (normal blend visualization)
    print("[2/3] Rendering mode 27 (normal blend visualization)...")
    frame_blend = _render_mode(27, hdr_path)
    blend_path = REPORTS_DIR / "mode27_normal_blend.png"
    _save_image(frame_blend, blend_path)
    print(f"  Saved: {blend_path}")
    
    # Analyze the blend visualization
    blend_gray = frame_blend[:, :, 0].astype(float) / 255.0
    blend_min_val = blend_gray.min()
    blend_max_val = blend_gray.max()
    blend_mean = blend_gray.mean()
    print(f"  Normal blend range: [{blend_min_val:.3f}, {blend_max_val:.3f}], mean: {blend_mean:.3f}")
    
    # 3. Create/update p5_meta.json
    print("[3/3] Creating p5_meta.json...")
    meta = {
        "milestone": 3,
        "description": "Bandlimit/fade strategy for extreme minification",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "lod_lo": LOD_FADE_START,
            "lod_hi": LOD_FADE_END,
            "blend_min": BLEND_MIN,
            "blend_max": BLEND_MAX,
        },
        "formula": "normal_blend = base_blend * (1.0 - saturate((lod - lod_lo) / (lod_hi - lod_lo)))",
        "interpretation": {
            "lod_0_to_lod_lo": "Full height-normal contribution (base_blend)",
            "lod_lo_to_lod_hi": "Linear fade from base_blend to 0",
            "lod_above_lod_hi": "No height-normal contribution (pure geometric normal)",
        },
        "artifacts": [
            "normal_blend_curve.png",
            "mode27_normal_blend.png",
        ],
        "acceptance_criteria": {
            "no_popping": "Camera orbit should not cause temporal popping",
            "far_field_stable": "Far-field should stay stable",
            "near_field_detail": "Near-field should preserve detail",
        },
    }
    
    meta_path = REPORTS_DIR / "p5_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Milestone 3 Complete!")
    print("=" * 60)
    print(f"\nDeliverables:")
    print(f"  - {curve_path}")
    print(f"  - {blend_path}")
    print(f"  - {meta_path}")
    
    print(f"\nParameters:")
    print(f"  lod_lo (fade start): {LOD_FADE_START}")
    print(f"  lod_hi (fade end):   {LOD_FADE_END}")
    print(f"  blend_min:           {BLEND_MIN}")
    print(f"  blend_max:           {BLEND_MAX}")
    
    print("\nAcceptance Criteria:")
    print("  - No 'popping' when orbiting camera (temporal stability improves)")
    print("  - Far-field stays stable without washing out near-field detail")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
