#!/usr/bin/env python3
"""
Water Surface Color Toggle demonstration for forge3d.

Showcases B11 Water Surface Color Toggle functionality including:
- Water surface color/albedo control
- Hue shifting capabilities
- Water transparency and alpha blending
- Different water presets (ocean, lake, river)
- Wave animation and flow direction
- Water surface mode toggling

Creates multiple water surface renders with different settings.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
except Exception as e:
    print(f"Failed to import forge3d: {e}")
    print("Make sure the package is installed or run 'maturin develop' first")
    sys.exit(0)


def create_simple_terrain(size: int) -> np.ndarray:
    """Create a simple terrain for water surface demonstration."""
    x = np.linspace(-3, 3, size, dtype=np.float32)
    y = np.linspace(-3, 3, size, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # Create gentle rolling hills around water level
    heights = (
        0.3 * np.sin(X * 2) * np.cos(Y * 2) +
        0.2 * np.sin(X * 4) * np.sin(Y * 3) +
        0.1 * np.cos(X * 6) * np.cos(Y * 5)
    )

    # Add some areas above and below water level
    heights += 0.2

    return heights


def demonstrate_water_modes(scene: f3d.Scene, output_dir: Path, terrain_size: int = 256):
    """Demonstrate different water surface modes and settings."""

    print("🌊 Creating water surface demonstrations...")

    # Create and set terrain
    heightmap = create_simple_terrain(terrain_size)
    scene.set_terrain_dims(terrain_size, terrain_size, 100.0)
    scene.set_terrain_heights(heightmap)

    # Set camera position to view water surface
    eye = (150.0, 80.0, -120.0)
    target = (0.0, 0.0, 0.0)
    up = (0.0, 1.0, 0.0)
    scene.set_camera_look_at(eye, target, up, 45.0, 1.0, 500.0)

    demonstrations = [
        {
            "name": "basic_water",
            "description": "Basic transparent water surface",
            "setup": lambda s: basic_water_setup(s),
        },
        {
            "name": "ocean_preset",
            "description": "Ocean water with deep blue color and waves",
            "setup": lambda s: ocean_preset_setup(s),
        },
        {
            "name": "lake_preset",
            "description": "Lake water with reflective surface",
            "setup": lambda s: lake_preset_setup(s),
        },
        {
            "name": "river_preset",
            "description": "River water with flow animation",
            "setup": lambda s: river_preset_setup(s),
        },
        {
            "name": "hue_shift_demo",
            "description": "Water with hue shift color transformation",
            "setup": lambda s: hue_shift_demo_setup(s),
        },
        {
            "name": "tinted_water",
            "description": "Water with strong green tint overlay",
            "setup": lambda s: tinted_water_setup(s),
        },
        {
            "name": "animated_waves",
            "description": "Water with large animated waves",
            "setup": lambda s: animated_waves_setup(s),
        },
    ]

    for demo in demonstrations:
        print(f"  📸 Rendering {demo['name']}: {demo['description']}")

        # Reset water surface
        scene.disable_water_surface()

        # Setup this demonstration
        demo["setup"](scene)

        # Render
        output_path = output_dir / f"water_{demo['name']}.png"
        start_time = time.time()
        scene.render_png(str(output_path))
        render_time = time.time() - start_time

        print(f"    ✅ Saved {output_path} (render time: {render_time:.2f}s)")


def basic_water_setup(scene: f3d.Scene):
    """Setup basic transparent water surface."""
    scene.enable_water_surface()
    scene.set_water_surface_mode("transparent")
    scene.set_water_surface_height(0.0)  # At water level
    scene.set_water_base_color(0.1, 0.3, 0.6)  # Blue water
    scene.set_water_alpha(0.7)  # Semi-transparent


def ocean_preset_setup(scene: f3d.Scene):
    """Setup ocean water preset with waves."""
    scene.enable_water_surface()
    scene.set_water_preset("ocean")
    scene.set_water_surface_height(-0.1)  # Slightly below terrain

    # Add wave animation
    scene.set_water_wave_params(0.3, 1.5, 1.2)  # amplitude, frequency, speed
    scene.update_water_animation(2.0)  # Advance animation


def lake_preset_setup(scene: f3d.Scene):
    """Setup lake water preset with reflections."""
    scene.enable_water_surface()
    scene.set_water_preset("lake")
    scene.set_water_surface_height(0.05)

    # Enhance reflectivity
    scene.set_water_lighting_params(1.0, 0.3, 5.0, 0.05)  # reflection, refraction, fresnel, roughness


def river_preset_setup(scene: f3d.Scene):
    """Setup river water with flow direction."""
    scene.enable_water_surface()
    scene.set_water_preset("river")
    scene.set_water_surface_height(0.0)

    # Set flow direction
    scene.set_water_flow_direction(1.0, 0.2)  # Flowing northeast
    scene.update_water_animation(3.0)  # Show flow animation


def hue_shift_demo_setup(scene: f3d.Scene):
    """Demonstrate hue shifting capabilities."""
    scene.enable_water_surface()
    scene.set_water_surface_mode("transparent")
    scene.set_water_base_color(0.2, 0.4, 0.8)  # Blue base
    scene.set_water_hue_shift(1.57)  # 90 degrees (π/2 radians) - shift to cyan/green
    scene.set_water_alpha(0.8)


def tinted_water_setup(scene: f3d.Scene):
    """Setup water with strong color tinting."""
    scene.enable_water_surface()
    scene.set_water_surface_mode("transparent")
    scene.set_water_base_color(0.1, 0.2, 0.4)  # Dark blue base
    scene.set_water_tint(0.2, 0.8, 0.3, 0.6)  # Strong green tint
    scene.set_water_alpha(0.75)


def animated_waves_setup(scene: f3d.Scene):
    """Setup water with large animated waves."""
    scene.enable_water_surface()
    scene.set_water_surface_mode("animated")
    scene.set_water_base_color(0.0, 0.4, 0.7)  # Bright blue

    # Large waves
    scene.set_water_wave_params(0.5, 1.0, 2.0)  # High amplitude, low frequency, fast speed
    scene.update_water_animation(1.5)  # Wave animation time
    scene.set_water_alpha(0.6)


def create_parameter_comparison(scene: f3d.Scene, output_dir: Path):
    """Create a comparison showing different parameter effects."""
    print("🎛️  Creating parameter comparison renders...")

    # Base water setup
    scene.enable_water_surface()
    scene.set_water_surface_mode("transparent")
    scene.set_water_surface_height(0.0)

    # Test different alphas
    alphas = [0.3, 0.5, 0.7, 0.9]
    for i, alpha in enumerate(alphas):
        scene.set_water_alpha(alpha)
        scene.set_water_base_color(0.1, 0.3, 0.6)

        output_path = output_dir / f"water_alpha_{alpha:.1f}.png"
        scene.render_png(str(output_path))
        print(f"  📸 Alpha {alpha:.1f} -> {output_path}")

    # Test different hue shifts
    hue_shifts = [0.0, 0.79, 1.57, 3.14]  # 0°, 45°, 90°, 180°
    for i, hue_shift in enumerate(hue_shifts):
        scene.set_water_alpha(0.7)
        scene.set_water_hue_shift(hue_shift)
        scene.set_water_base_color(0.2, 0.4, 0.8)

        degrees = int(hue_shift * 180 / 3.14159)
        output_path = output_dir / f"water_hue_{degrees:03d}deg.png"
        scene.render_png(str(output_path))
        print(f"  🌈 Hue shift {degrees}° -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Water Surface Color Toggle demonstration")
    parser.add_argument("--output-dir", type=str, default="water_surface_outputs",
                       help="Output directory for rendered images")
    parser.add_argument("--size", type=int, default=256,
                       help="Terrain size (default: 256)")
    parser.add_argument("--width", type=int, default=800,
                       help="Render width (default: 800)")
    parser.add_argument("--height", type=int, default=600,
                       help="Render height (default: 600)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"🌊 B11 Water Surface Color Toggle Demo")
    print(f"   Output directory: {output_dir}")
    print(f"   Terrain size: {args.size}x{args.size}")
    print(f"   Render size: {args.width}x{args.height}")
    print()

    try:
        # Create scene
        scene = f3d.Scene(args.width, args.height, args.size)

        # Enable some basic terrain features for context
        scene.set_lighting(sun_dir=(0.3, 0.7, 0.2))
        scene.set_colormap_name("viridis")

        # Run demonstrations
        start_time = time.time()

        demonstrate_water_modes(scene, output_dir, args.size)
        create_parameter_comparison(scene, output_dir)

        total_time = time.time() - start_time

        print()
        print(f"✅ Water surface demonstration complete!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Output files in: {output_dir}")
        print()
        print("🎯 B11 Water Surface Features Demonstrated:")
        print("   ✓ Pipeline uniform controlling water albedo/hue")
        print("   ✓ Python setter methods for water color control")
        print("   ✓ Water tint toggling and transparency")
        print("   ✓ Water surface mode switching")
        print("   ✓ Wave animation and flow direction")
        print("   ✓ Water presets (ocean, lake, river)")
        print("   ✓ Hue shift color transformation")
        print("   ✓ Alpha blending and transparency control")

    except Exception as e:
        print(f"❌ Error during water surface demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())