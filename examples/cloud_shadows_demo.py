# examples/cloud_shadows_demo.py
# Cloud Shadow Overlay demo for Workstream B7.
# Showcases 2D shadow texture modulation over terrain with parametric density/speed.
# RELEVANT FILES:shaders/cloud_shadows.wgsl,src/core/cloud_shadows.rs,src/scene/mod.rs,tests/test_b7_cloud_shadows.py

"""Generate cloud shadow demonstration renders showing moving shadows over terrain."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import time

from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
except Exception:
    print("forge3d extension not available; skipping cloud shadow demo.")
    import sys; sys.exit(0)


def _build_varied_terrain(size: int) -> np.ndarray:
    """Create a varied terrain heightmap to showcase cloud shadow effects."""
    coords = np.linspace(-3.0, 3.0, size, dtype=np.float32)
    ys, xs = np.meshgrid(coords, coords, indexing="ij")

    # Create rolling hills with valleys
    height = np.zeros_like(xs)

    # Large rolling hills
    height += 0.3 * np.sin(xs * 0.8) * np.cos(ys * 0.6)

    # Medium frequency detail
    height += 0.15 * np.sin(xs * 2.0) * np.cos(ys * 1.8)

    # Fine detail
    height += 0.05 * np.sin(xs * 6.0) * np.cos(ys * 5.5)

    # Add some valleys
    valleys = np.exp(-((xs - 1.0)**2 + (ys - 0.5)**2) / 0.5)
    height -= 0.2 * valleys

    valleys2 = np.exp(-((xs + 0.8)**2 + (ys + 1.2)**2) / 0.7)
    height -= 0.15 * valleys2

    # Ensure terrain has good elevation variation
    height = height - height.min() + 0.1

    return height.astype(np.float32)


def main() -> None:
    output_dir = Path("cloud_shadows_demo_out")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scene with terrain
    scene = f3d.Scene(512, 512, grid=256, colormap="terrain")

    # Set up varied terrain
    heightmap = _build_varied_terrain(256)
    scene.set_height_from_r32f(heightmap)

    # Position camera for good overview
    scene.set_camera_look_at(
        eye=(2.0, 2.5, 2.0),
        target=(0.0, 0.3, 0.0),
        up=(0.0, 1.0, 0.0),
        fovy_deg=50.0,
        znear=0.1,
        zfar=10.0
    )

    print("Rendering baseline (no cloud shadows)...")
    baseline_path = output_dir / "cloud_shadows_baseline.png"
    scene.render_png(baseline_path)

    # Enable cloud shadows with medium quality
    print("Enabling cloud shadows...")
    scene.enable_cloud_shadows(quality="medium")

    # Test 1: Basic cloud shadows with default settings
    print("Rendering basic cloud shadows...")
    basic_path = output_dir / "cloud_shadows_basic.png"
    scene.render_png(basic_path)

    # Test 2: Dense clouds with strong shadows
    print("Rendering dense clouds with strong shadows...")
    scene.set_cloud_density(0.8)
    scene.set_cloud_coverage(0.6)
    scene.set_cloud_shadow_intensity(0.9)
    dense_path = output_dir / "cloud_shadows_dense.png"
    scene.render_png(dense_path)

    # Test 3: Sparse, soft clouds
    print("Rendering sparse, soft clouds...")
    scene.set_cloud_density(0.4)
    scene.set_cloud_coverage(0.3)
    scene.set_cloud_shadow_intensity(0.5)
    scene.set_cloud_shadow_softness(0.8)
    sparse_path = output_dir / "cloud_shadows_sparse.png"
    scene.render_png(sparse_path)

    # Test 4: Fast moving clouds
    print("Rendering fast moving clouds...")
    scene.set_cloud_density(0.6)
    scene.set_cloud_coverage(0.5)
    scene.set_cloud_shadow_intensity(0.7)
    scene.set_cloud_shadow_softness(0.4)
    scene.set_cloud_speed(0.1, 0.05)  # Fast movement
    fast_path = output_dir / "cloud_shadows_fast.png"
    scene.render_png(fast_path)

    # Test 5: Large scale clouds
    print("Rendering large scale clouds...")
    scene.set_cloud_speed(0.02, 0.01)  # Reset to normal speed
    scene.set_cloud_scale(1.0)  # Large scale
    large_scale_path = output_dir / "cloud_shadows_large_scale.png"
    scene.render_png(large_scale_path)

    # Test 6: Small scale, detailed clouds
    print("Rendering small scale, detailed clouds...")
    scene.set_cloud_scale(4.0)  # Small scale, more detail
    small_scale_path = output_dir / "cloud_shadows_small_scale.png"
    scene.render_png(small_scale_path)

    # Test 7: Wind effects
    print("Rendering with wind effects...")
    scene.set_cloud_scale(2.0)  # Reset scale
    scene.set_cloud_wind(45.0, 1.5)  # 45 degree wind direction, moderate strength
    wind_path = output_dir / "cloud_shadows_wind.png"
    scene.render_png(wind_path)

    # Test 8: Animation presets - Calm weather
    print("Rendering calm weather preset...")
    scene.set_cloud_animation_preset("calm")
    scene.update_cloud_animation(0.0)  # Reset time
    calm_path = output_dir / "cloud_shadows_calm.png"
    scene.render_png(calm_path)

    # Test 9: Animation presets - Windy weather
    print("Rendering windy weather preset...")
    scene.set_cloud_animation_preset("windy")
    windy_path = output_dir / "cloud_shadows_windy.png"
    scene.render_png(windy_path)

    # Test 10: Animation presets - Stormy weather
    print("Rendering stormy weather preset...")
    scene.set_cloud_animation_preset("stormy")
    stormy_path = output_dir / "cloud_shadows_stormy.png"
    scene.render_png(stormy_path)

    # Test 11: High quality clouds
    print("Rendering high quality clouds...")
    scene.disable_cloud_shadows()
    scene.enable_cloud_shadows(quality="high")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.5)
    scene.set_cloud_shadow_intensity(0.8)
    scene.set_cloud_animation_preset("windy")
    high_quality_path = output_dir / "cloud_shadows_high_quality.png"
    scene.render_png(high_quality_path)

    # Test 12: Cloud pattern visualization (debug mode)
    print("Rendering cloud pattern visualization...")
    scene.set_cloud_show_clouds_only(True)
    clouds_only_path = output_dir / "cloud_shadows_clouds_only.png"
    scene.render_png(clouds_only_path)
    scene.set_cloud_show_clouds_only(False)

    # Test 13: Animated sequence (multiple frames with time progression)
    print("Rendering animated sequence...")
    scene.set_cloud_animation_preset("windy")
    scene.set_cloud_density(0.6)
    scene.set_cloud_coverage(0.4)
    scene.set_cloud_shadow_intensity(0.7)

    for i, time_step in enumerate([0.0, 2.0, 4.0, 6.0, 8.0]):
        scene.update_cloud_animation(time_step)
        frame_path = output_dir / f"cloud_shadows_anim_frame_{i:02d}.png"
        scene.render_png(frame_path)
        print(f"  Rendered animation frame {i+1}/5")

    # Get current cloud parameters for verification
    density, coverage, intensity, softness = scene.get_cloud_params()

    print(f"\nCloud Shadow Parameters:")
    print(f"  Density: {density:.2f}")
    print(f"  Coverage: {coverage:.2f}")
    print(f"  Shadow Intensity: {intensity:.2f}")
    print(f"  Shadow Softness: {softness:.2f}")

    print(f"\nRendered images:")
    print(f"  Baseline (no clouds): {baseline_path}")
    print(f"  Basic clouds: {basic_path}")
    print(f"  Dense clouds: {dense_path}")
    print(f"  Sparse clouds: {sparse_path}")
    print(f"  Fast moving: {fast_path}")
    print(f"  Large scale: {large_scale_path}")
    print(f"  Small scale: {small_scale_path}")
    print(f"  With wind: {wind_path}")
    print(f"  Calm preset: {calm_path}")
    print(f"  Windy preset: {windy_path}")
    print(f"  Stormy preset: {stormy_path}")
    print(f"  High quality: {high_quality_path}")
    print(f"  Clouds only: {clouds_only_path}")
    print(f"  Animation frames: cloud_shadows_anim_frame_*.png")

    print(f"\nCloud shadow demo completed! Check the {output_dir} directory for comparison images.")
    print(f"The cloud shadows demonstrate 2D shadow texture modulation over terrain")
    print(f"with parametric density and speed control as specified in B7.")


if __name__ == "__main__":
    main()