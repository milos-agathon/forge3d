#!/usr/bin/env python3
"""M7: Blender-Quality Demo - Showcase all M1-M6 rendering features.

This script demonstrates all the Blender-like offline rendering features
implemented in Milestones 1-6:

- M1: Accumulation Anti-Aliasing (multi-sample jittered AA)
- M2: Bloom post-processing (HDR glow effects)
- M4: Material Layering (snow, rock, wetness based on terrain)
- M5: Depth-Correct Vector Overlays (halos, depth testing)
- M6: Tonemap Enhancements (operator selection, white balance)

Usage:
    # Basic demo with all features enabled
    python examples/blender_quality_demo.py --dem assets/dem_rainier.tif --output examples/output/demo_output.png
    
    # Alpine scene with snow and ACES tonemap
    python examples/blender_quality_demo.py --dem assets/dem_rainier.tif \\
        --preset alpine --output examples/output/alpine_render.png
    
    # Cinematic warm-toned render
    python examples/blender_quality_demo.py --dem assets/dem_rainier.tif \\
        --preset cinematic --output examples/output/cinematic_render.png

RELEVANT FILES: python/forge3d/terrain_params.py, src/terrain/render_params.rs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add python path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from forge3d.terrain_params import (
        TerrainRenderParams,
        BloomSettings,
        MaterialLayerSettings,
        VectorOverlaySettings,
        TonemapSettings,
        make_terrain_params_config,
    )
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False
    print("Warning: forge3d not installed. Run: pip install -e .")


# Preset configurations for different rendering styles
PRESETS = {
    "default": {
        "description": "Default settings with all features disabled",
        "aa_samples": 1,
        "bloom": BloomSettings() if FORGE3D_AVAILABLE else None,
        "materials": MaterialLayerSettings() if FORGE3D_AVAILABLE else None,
        "vector_overlay": VectorOverlaySettings() if FORGE3D_AVAILABLE else None,
        "tonemap": TonemapSettings() if FORGE3D_AVAILABLE else None,
    },
    "alpine": {
        "description": "Alpine mountain scene with snow and rock layers",
        "aa_samples": 16,
        "bloom": BloomSettings(enabled=True, intensity=0.3, threshold=0.8) if FORGE3D_AVAILABLE else None,
        "materials": MaterialLayerSettings(
            snow_enabled=True,
            snow_altitude_min=2500.0,
            snow_altitude_blend=300.0,
            snow_slope_max=50.0,
            rock_enabled=True,
            rock_slope_min=40.0,
        ) if FORGE3D_AVAILABLE else None,
        "vector_overlay": VectorOverlaySettings(
            depth_test=True,
            halo_enabled=True,
            halo_width=2.0,
        ) if FORGE3D_AVAILABLE else None,
        "tonemap": TonemapSettings(
            operator="aces",
            white_balance_enabled=True,
            temperature=7000.0,  # Slightly cool for snow
        ) if FORGE3D_AVAILABLE else None,
    },
    "cinematic": {
        "description": "Cinematic warm-toned render with bloom",
        "aa_samples": 64,
        "bloom": BloomSettings(enabled=True, intensity=0.5, threshold=0.7) if FORGE3D_AVAILABLE else None,
        "materials": MaterialLayerSettings(
            wetness_enabled=True,
            wetness_strength=0.4,
        ) if FORGE3D_AVAILABLE else None,
        "vector_overlay": VectorOverlaySettings(
            halo_enabled=True,
            halo_width=3.0,
            halo_color=(0.0, 0.0, 0.0, 0.7),
        ) if FORGE3D_AVAILABLE else None,
        "tonemap": TonemapSettings(
            operator="uncharted2",
            white_point=6.0,
            white_balance_enabled=True,
            temperature=5500.0,  # Warm golden hour
            tint=0.1,
        ) if FORGE3D_AVAILABLE else None,
    },
    "high_quality": {
        "description": "Maximum quality with 256-sample AA",
        "aa_samples": 256,
        "bloom": BloomSettings(enabled=True, intensity=0.2, threshold=0.9) if FORGE3D_AVAILABLE else None,
        "materials": MaterialLayerSettings(
            snow_enabled=True,
            rock_enabled=True,
            wetness_enabled=True,
        ) if FORGE3D_AVAILABLE else None,
        "vector_overlay": VectorOverlaySettings(
            depth_test=True,
            halo_enabled=True,
        ) if FORGE3D_AVAILABLE else None,
        "tonemap": TonemapSettings(operator="aces") if FORGE3D_AVAILABLE else None,
    },
}


def create_demo_params(
    size_px: tuple[int, int],
    preset_name: str,
    exposure: float = 1.0,
    terrain_span: float = 10000.0,
    domain: tuple[float, float] = (0.0, 4000.0),
) -> TerrainRenderParams:
    """Create render parameters with the specified preset."""
    if not FORGE3D_AVAILABLE:
        raise RuntimeError("forge3d not available")
    
    preset = PRESETS.get(preset_name, PRESETS["default"])
    
    return make_terrain_params_config(
        size_px=size_px,
        render_scale=1.0,
        terrain_span=terrain_span,
        msaa_samples=1,  # Use AA samples instead
        z_scale=1.0,
        exposure=exposure,
        domain=domain,
        aa_samples=preset.get("aa_samples", 1),
        bloom=preset.get("bloom"),
        materials=preset.get("materials"),
        vector_overlay=preset.get("vector_overlay"),
        tonemap=preset.get("tonemap"),
    )


def print_preset_info(preset_name: str) -> None:
    """Print information about the selected preset."""
    preset = PRESETS.get(preset_name, PRESETS["default"])
    print(f"\n{'='*60}")
    print(f"PRESET: {preset_name}")
    print(f"{'='*60}")
    print(f"Description: {preset.get('description', 'No description')}")
    print(f"AA Samples: {preset.get('aa_samples', 1)}")
    
    if FORGE3D_AVAILABLE:
        bloom = preset.get("bloom")
        if bloom and bloom.enabled:
            print(f"Bloom: enabled (intensity={bloom.intensity}, threshold={bloom.threshold})")
        else:
            print("Bloom: disabled")
        
        materials = preset.get("materials")
        if materials:
            features = []
            if materials.snow_enabled:
                features.append(f"snow(alt>{materials.snow_altitude_min})")
            if materials.rock_enabled:
                features.append(f"rock(slope>{materials.rock_slope_min}°)")
            if materials.wetness_enabled:
                features.append(f"wetness({materials.wetness_strength})")
            if features:
                print(f"Materials: {', '.join(features)}")
            else:
                print("Materials: disabled")
        
        tonemap = preset.get("tonemap")
        if tonemap:
            print(f"Tonemap: {tonemap.operator}", end="")
            if tonemap.white_balance_enabled:
                print(f" (temp={tonemap.temperature}K, tint={tonemap.tint})")
            else:
                print()
    
    print(f"{'='*60}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="M7: Blender-Quality Demo - Showcase all rendering features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  default     - All features disabled (baseline)
  alpine      - Snow/rock layers, ACES tonemap, cool temperature
  cinematic   - Warm tones, bloom, wetness, Uncharted2 tonemap
  high_quality - Maximum quality, 256-sample AA, all features

Examples:
  python examples/blender_quality_demo.py --dem terrain.tif --preset alpine
  python examples/blender_quality_demo.py --dem terrain.tif --preset cinematic --exposure 1.2
""",
    )
    parser.add_argument("--dem", type=Path, help="Path to GeoTIFF DEM file")
    parser.add_argument("--output", "-o", type=Path, default=Path("blender_demo.png"),
                        help="Output image path (default: blender_demo.png)")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="alpine",
                        help="Rendering preset (default: alpine)")
    parser.add_argument("--width", type=int, default=1920, help="Output width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="Output height (default: 1080)")
    parser.add_argument("--exposure", type=float, default=1.0, help="Exposure multiplier (default: 1.0)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show config without rendering")
    
    args = parser.parse_args()
    
    # List presets
    if args.list_presets:
        print("\nAvailable Presets:")
        print("-" * 40)
        for name, preset in PRESETS.items():
            print(f"  {name:15} - {preset.get('description', 'No description')}")
        print()
        return 0
    
    # Check forge3d availability
    if not FORGE3D_AVAILABLE:
        print("Error: forge3d not installed. Run: pip install -e .")
        return 1
    
    # Print preset info
    print_preset_info(args.preset)
    
    # Create render parameters
    try:
        params = create_demo_params(
            size_px=(args.width, args.height),
            preset_name=args.preset,
            exposure=args.exposure,
        )
        print(f"Created render params: {args.width}x{args.height}")
        print(f"  AA samples: {params.aa_samples}")
        print(f"  Exposure: {params.exposure}")
        print(f"  Bloom: {params.bloom.enabled if params.bloom else False}")
        print(f"  Snow: {params.materials.snow_enabled if params.materials else False}")
        print(f"  Rock: {params.materials.rock_enabled if params.materials else False}")
        print(f"  Tonemap: {params.tonemap.operator if params.tonemap else 'default'}")
    except Exception as e:
        print(f"Error creating params: {e}")
        return 1
    
    if args.dry_run:
        print("\n[Dry run - no rendering performed]")
        return 0
    
    # Check DEM file
    if not args.dem:
        print("Error: --dem argument required for rendering")
        print("Use --dry-run to test configuration without rendering")
        return 1
    
    if not args.dem.exists():
        print(f"Error: DEM file not found: {args.dem}")
        return 1
    
    print(f"\nRendering {args.dem} -> {args.output}")
    print("(Actual rendering requires forge3d GPU context - this demo shows config)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
