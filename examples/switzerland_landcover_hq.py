#!/usr/bin/env python3
"""
High-Quality Switzerland Terrain Rendering with Blinn-Phong Lighting

This example demonstrates advanced rendering features:
- White background for clean publication-quality output
- Blinn-Phong lighting model with specular highlights
- Configurable shininess and specular strength
- Multiple quality presets (standard, high, ultra)

Usage:
    python switzerland_landcover_hq.py --preset high --output swiss_hq.png
    python switzerland_landcover_hq.py --width 3840 --height 2160 --preset ultra
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image

try:
    import forge3d.terrain as terrain
except ImportError:
    print("ERROR: forge3d not installed. Run: pip install -e .")
    exit(1)


# ESA WorldCover 2021 color mapping (categorical)
ESA_COLORS = {
    10: "#006400",  # Tree cover
    20: "#ffbb22",  # Shrubland
    30: "#ffff4c",  # Grassland
    40: "#f096ff",  # Cropland
    50: "#fa0000",  # Built-up
    60: "#b4b4b4",  # Bare / sparse vegetation
    70: "#f0f0f0",  # Snow and ice
    80: "#0064c8",  # Permanent water bodies
    90: "#0096a0",  # Herbaceous wetland
    95: "#00cf75",  # Mangroves
    100: "#fae6a0", # Moss and lichen
    0: "#00000000", # NoData (transparent)
}


def hex_to_rgba(hex_color):
    """Convert hex color to RGBA tuple (0-255 range)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        # RGB, assume opaque
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)) + (255,)
    elif len(hex_color) == 8:
        # RGBA
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))
    else:
        return (0, 0, 0, 0)


def landcover_classes_to_rgba(landcover_classes):
    """
    Convert ESA WorldCover class codes to RGBA image.
    
    Parameters
    ----------
    landcover_classes : np.ndarray
        2D array of ESA WorldCover class codes (0-100)
    
    Returns
    -------
    np.ndarray
        3D RGBA array, shape (H, W, 4), dtype uint8
    """
    h, w = landcover_classes.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Create lookup table for all possible class values
    lut = np.zeros((256, 4), dtype=np.uint8)
    for class_code, hex_color in ESA_COLORS.items():
        lut[class_code] = hex_to_rgba(hex_color)
    
    # Apply lookup table
    rgba = lut[landcover_classes]
    
    return rgba


def fetch_switzerland_data(country_code="CH", cache_dir="./cache"):
    """
    Fetch DEM and land cover data for Switzerland.
    Uses cached data if available.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    dem_file = cache_path / f"dem_{country_code}_1200.npz"
    lc_file = cache_path / f"landcover_{country_code}_1200.npz"
    
    if dem_file.exists() and lc_file.exists():
        print(f"Loading cached data from {cache_dir}...")
        dem_data = np.load(dem_file)
        lc_data = np.load(lc_file)
        
        dem = dem_data['dem']
        landcover_classes = lc_data['landcover']
        
        # Convert class codes to RGBA if needed
        if landcover_classes.ndim == 2:
            print(f"Converting landcover classes to RGBA...")
            landcover_rgba = landcover_classes_to_rgba(landcover_classes)
        else:
            landcover_rgba = landcover_classes
        
        return dem, landcover_rgba
    
    print(f"Cached data not found. Please run the base example first:")
    print(f"  python examples/switzerland_landcover_drape.py --country {country_code}")
    exit(1)


def get_preset(preset_name):
    """
    Get rendering preset configuration.
    
    Presets:
    - standard: Fast, good quality (default)
    - high: High quality with stronger specular
    - ultra: Maximum quality settings
    - rayshader: Mimics rayshader::render_highquality appearance
    """
    presets = {
        "standard": {
            "lighting_model": "blinn_phong",
            "shininess": 32.0,
            "specular_strength": 0.3,
            "light_elevation": 45.0,
            "light_azimuth": 315.0,
            "light_intensity": 1.0,
            "ambient": 0.25,
            "camera_phi": 25.0,
            "camera_theta": 45.0,
        },
        "high": {
            "lighting_model": "blinn_phong",
            "shininess": 48.0,
            "specular_strength": 0.4,
            "light_elevation": 50.0,
            "light_azimuth": 320.0,
            "light_intensity": 1.1,
            "ambient": 0.2,
            "camera_phi": 30.0,
            "camera_theta": 45.0,
        },
        "ultra": {
            "lighting_model": "blinn_phong",
            "shininess": 64.0,
            "specular_strength": 0.5,
            "light_elevation": 55.0,
            "light_azimuth": 325.0,
            "light_intensity": 1.2,
            "ambient": 0.18,
            "camera_phi": 32.0,
            "camera_theta": 48.0,
        },
        "rayshader": {
            "lighting_model": "blinn_phong",
            "shininess": 40.0,
            "specular_strength": 0.35,
            "light_elevation": 45.0,
            "light_azimuth": 300.0,
            "light_intensity": 1.15,
            "ambient": 0.22,
            "camera_phi": 28.0,
            "camera_theta": 50.0,
        },
    }
    
    if preset_name not in presets:
        print(f"Unknown preset: {preset_name}")
        print(f"Available presets: {', '.join(presets.keys())}")
        exit(1)
    
    return presets[preset_name]


def main():
    parser = argparse.ArgumentParser(
        description="High-Quality Switzerland 3D Land Cover Rendering"
    )
    
    # Data options
    parser.add_argument("--country", default="CH", help="Country code (default: CH)")
    parser.add_argument("--cache-dir", default="./cache", help="Cache directory")
    
    # Output options
    parser.add_argument("--output", default="swiss_hq.png", help="Output image path")
    parser.add_argument("--width", type=int, default=2400, help="Output width")
    parser.add_argument("--height", type=int, default=1600, help="Output height")
    
    # Quality preset
    parser.add_argument(
        "--preset",
        default="high",
        choices=["standard", "high", "ultra", "rayshader"],
        help="Quality preset (default: high)",
    )
    
    # Advanced overrides (optional)
    parser.add_argument("--zscale", type=float, help="Vertical exaggeration (auto if not set)")
    parser.add_argument("--lighting-model", choices=["lambert", "phong", "blinn_phong"],
                       help="Override lighting model")
    parser.add_argument("--fov", type=float, default=0.0, help="Field of view (degrees)")
    parser.add_argument("--shininess", type=float, help="Override shininess (1-256)")
    parser.add_argument("--specular", type=float, help="Override specular strength (0-1)")
    parser.add_argument("--background", default="#FFFFFF", help="Background color (hex)")
    parser.add_argument("--camera-theta", type=float, help="Camera azimuth override (0=North, 90=East, 180=South, 270=West)")
    parser.add_argument("--camera-phi", type=float, help="Camera elevation override (0=horizon, 90=top-down)")
    parser.add_argument("--camera-gamma", type=float, default=0.0, help="Camera roll/rotation (degrees)")
    
    # Shadow mapping parameters
    parser.add_argument("--shadow-intensity", type=float, default=0.6, help="Shadow darkness (0=no shadows, 1=full shadows)")
    parser.add_argument("--shadow-softness", type=float, default=2.0, help="Shadow softness/PCF kernel radius (1.0-5.0)")
    parser.add_argument("--shadow-map-res", type=int, default=2048, help="Shadow map resolution (512-4096)")
    parser.add_argument("--shadow-bias", type=float, default=0.0015, help="Depth bias for shadow acne prevention")
    parser.add_argument("--no-shadows", action="store_true", help="Disable shadow rendering")
    
    # HDRI environment lighting
    parser.add_argument("--hdri", type=str, default=None, help="Path to HDRI environment map (.hdr, .exr)")
    parser.add_argument("--hdri-intensity", type=float, default=1.0, help="HDRI intensity multiplier (0-2+)")
    parser.add_argument("--hdri-rotation", type=float, default=0.0, help="HDRI rotation in degrees (0-360)")
    
    # Denoising
    parser.add_argument("--denoiser", default="oidn", choices=["none", "oidn", "bilateral"], help="Denoising method (default: oidn with fallback)")
    parser.add_argument("--denoise-strength", type=float, default=0.8, help="Denoising intensity 0.0-1.0 (default: 0.8)")
    
    args = parser.parse_args()
    
    # Fetch data
    print(f"Fetching data for {args.country}...")
    dem, landcover = fetch_switzerland_data(args.country, args.cache_dir)
    
    print(f"DEM shape: {dem.shape}, range: [{dem.min():.1f}, {dem.max():.1f}] m")
    print(f"Land cover shape: {landcover.shape}")
    
    # Convert to relative heights (important for proper scale)
    dem_relative = dem - dem.min()
    print(f"Using relative heights: [0, {dem_relative.max():.1f}] m")
    
    # Get preset configuration
    config = get_preset(args.preset)
    print(f"\nUsing preset: {args.preset}")
    
    # Apply overrides
    if args.zscale is not None:
        zscale = args.zscale
    else:
        # Auto-calculate zscale based on terrain extent
        h_range = dem_relative.max()
        terrain_extent = max(dem.shape) * 30  # Approximate meters (30m pixel)
        zscale = terrain_extent / h_range * 0.0008  # Scale for good visual balance
        zscale = max(0.5, min(2.0, zscale))  # Clamp to reasonable range
        print(f"Auto zscale: {zscale:.3f}")
    
    if args.lighting_model:
        config["lighting_model"] = args.lighting_model
    if args.shininess:
        config["shininess"] = args.shininess
    if args.specular:
        config["specular_strength"] = args.specular
    if args.camera_theta:
        config["camera_theta"] = args.camera_theta
    if args.camera_phi:
        config["camera_phi"] = args.camera_phi
    
    # Render
    print(f"\nRendering {args.width}×{args.height} with {config['lighting_model']} lighting...")
    print(f"  Shininess: {config['shininess']}")
    print(f"  Specular strength: {config['specular_strength']}")
    print(f"  Light elevation: {config['light_elevation']}°")
    print(f"  Light azimuth: {config['light_azimuth']}°")
    print(f"  Background: {args.background}")
    print(f"  Shadows: {'disabled' if args.no_shadows else f'enabled (intensity={args.shadow_intensity}, softness={args.shadow_softness}, res={args.shadow_map_res})'}")
    print(f"  HDRI: {args.hdri if args.hdri else 'none (using constant ambient)'}")
    if args.hdri:
        print(f"  HDRI intensity: {args.hdri_intensity}, rotation: {args.hdri_rotation}°")
    print(f"  Denoiser: {args.denoiser} (strength={args.denoise_strength})")
    
    img = terrain.drape_landcover(
        dem_relative,
        landcover,
        width=args.width,
        height=args.height,
        zscale=zscale,
        max_dim=3000,
        # Camera
        camera_theta=config["camera_theta"],
        camera_phi=config["camera_phi"],
        camera_fov=args.fov if args.fov > 0 else 35.0,
        camera_gamma=args.camera_gamma,
        # Lighting
        light_type="directional",
        light_elevation=config["light_elevation"],
        light_azimuth=config["light_azimuth"],
        light_intensity=config["light_intensity"],
        ambient=config["ambient"],
        # HQ features
        lighting_model=config["lighting_model"],
        shininess=config["shininess"],
        specular_strength=config["specular_strength"],
        # Shadow mapping
        shadow_intensity=args.shadow_intensity,
        shadow_softness=args.shadow_softness,
        shadow_map_res=args.shadow_map_res,
        shadow_bias=args.shadow_bias,
        enable_shadows=not args.no_shadows,
        # HDRI environment
        hdri=args.hdri,
        hdri_intensity=args.hdri_intensity,
        hdri_rotation_deg=args.hdri_rotation,
        # Denoising
        denoiser=args.denoiser,
        denoise_strength=args.denoise_strength,
        background=args.background,
    )
    
    # Save output
    output_path = Path(args.output)
    Image.fromarray(img).save(output_path)
    
    print(f"\n✓ Saved to {output_path}")
    print(f"  Resolution: {args.width}×{args.height}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print quality notes
    print("\n" + "="*60)
    print("QUALITY NOTES")
    print("="*60)
    print(f"Preset: {args.preset}")
    print(f"  - Lambert: Fast, no specular (simple diffuse)")
    print(f"  - Phong: View-dependent specular (reflected ray)")
    print(f"  - Blinn-Phong: Wider, smoother highlights (half-vector)")
    print(f"\nCurrent: {config['lighting_model']} with shininess={config['shininess']}")
    print(f"Background: {args.background} ({'white' if args.background == '#FFFFFF' else 'custom'})")
    
    print("\nTry different presets:")
    print("  --preset standard    # Fast, good quality")
    print("  --preset high        # Better lighting and specular")
    print("  --preset ultra       # Maximum quality")
    print("  --preset rayshader   # Mimics rayshader appearance")


if __name__ == "__main__":
    main()
