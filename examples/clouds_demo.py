# examples/clouds_demo.py
# B8: Realtime Clouds demonstration
# Showcases billboard/volumetric cloud rendering with IBL-aware scattering
# RELEVANT FILES: shaders/clouds.wgsl, src/core/clouds.rs, src/scene/mod.rs, tests/test_b8_clouds.py

"""Generate realtime cloud rendering demonstrations with various quality and animation settings."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import time

from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
except Exception:
    print("forge3d extension not available; skipping cloud rendering demo.")
    import sys; sys.exit(0)


def _build_scenic_terrain(size: int) -> np.ndarray:
    """Create a scenic terrain heightmap to showcase cloud rendering effects."""
    coords = np.linspace(-4.0, 4.0, size, dtype=np.float32)
    ys, xs = np.meshgrid(coords, coords, indexing="ij")

    # Create mountainous terrain with peaks and valleys
    height = np.zeros_like(xs)

    # Mountain ranges
    height += 0.8 * np.exp(-((xs - 1.0)**2 + (ys - 0.5)**2) / 1.5)  # Main peak
    height += 0.6 * np.exp(-((xs + 1.5)**2 + (ys + 1.0)**2) / 1.2)  # Secondary peak
    height += 0.4 * np.exp(-((xs - 0.5)**2 + (ys - 1.8)**2) / 0.8)  # Small hill

    # Rolling hills base
    height += 0.3 * np.sin(xs * 0.6) * np.cos(ys * 0.4)

    # Medium frequency detail
    height += 0.15 * np.sin(xs * 1.8) * np.cos(ys * 1.5)

    # Fine detail noise
    height += 0.08 * np.sin(xs * 4.0) * np.cos(ys * 3.5)

    # Create valleys between mountains
    valleys = np.exp(-((xs - 0.2)**2 + (ys + 0.8)**2) / 0.6)
    height -= 0.25 * valleys

    # Ensure positive height and good variation
    height = height - height.min() + 0.1

    return height.astype(np.float32)


def main() -> None:
    output_dir = Path("clouds_demo_out")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scene with scenic terrain
    scene = f3d.Scene(1024, 768, grid=256, colormap="terrain")  # 1024x768 for performance testing

    # Set up scenic terrain
    heightmap = _build_scenic_terrain(256)
    scene.set_height_from_r32f(heightmap)

    # Position camera for good cloud and terrain view
    scene.set_camera_look_at(
        eye=(3.0, 4.0, 3.0),
        target=(0.0, 1.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fovy_deg=60.0,
        znear=0.1,
        zfar=50.0
    )

    print("Rendering baseline (no clouds)...")
    baseline_path = output_dir / "clouds_baseline.png"
    scene.render_png(baseline_path)

    # Test 1: Low quality clouds for performance
    print("Testing low quality clouds (performance optimized)...")
    scene.enable_clouds(quality="low")
    scene.set_cloud_density(0.6)
    scene.set_cloud_coverage(0.5)

    # Measure performance
    start_time = time.time()
    for _ in range(5):  # Render 5 frames to measure average performance
        scene.render_rgba()
    avg_frame_time = (time.time() - start_time) / 5
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

    low_quality_path = output_dir / "clouds_low_quality.png"
    scene.render_png(low_quality_path)
    print(f"  Low quality performance: {fps:.1f} FPS (target: 60+ FPS)")

    # Test 2: Medium quality clouds (balanced)
    print("Testing medium quality clouds (balanced)...")
    scene.enable_clouds(quality="medium")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.6)

    medium_quality_path = output_dir / "clouds_medium_quality.png"
    scene.render_png(medium_quality_path)

    # Test 3: High quality clouds (best visual quality)
    print("Testing high quality clouds (high detail)...")
    scene.enable_clouds(quality="high")
    scene.set_cloud_density(0.8)
    scene.set_cloud_coverage(0.65)

    high_quality_path = output_dir / "clouds_high_quality.png"
    scene.render_png(high_quality_path)

    # Test 4: Ultra quality clouds (maximum quality)
    print("Testing ultra quality clouds (maximum detail)...")
    scene.enable_clouds(quality="ultra")
    scene.set_cloud_density(0.75)
    scene.set_cloud_coverage(0.7)

    ultra_quality_path = output_dir / "clouds_ultra_quality.png"
    scene.render_png(ultra_quality_path)

    # Test 5: Billboard render mode (fastest)
    print("Testing billboard render mode (fastest)...")
    scene.enable_clouds(quality="medium")
    scene.set_cloud_render_mode("billboard")
    scene.set_cloud_density(0.6)
    scene.set_cloud_coverage(0.5)

    billboard_path = output_dir / "clouds_billboard_mode.png"
    scene.render_png(billboard_path)

    # Test 6: Volumetric render mode (highest quality)
    print("Testing volumetric render mode (highest quality)...")
    scene.set_cloud_render_mode("volumetric")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.6)

    volumetric_path = output_dir / "clouds_volumetric_mode.png"
    scene.render_png(volumetric_path)

    # Test 7: Hybrid render mode (distance-based LOD)
    print("Testing hybrid render mode (distance-based LOD)...")
    scene.set_cloud_render_mode("hybrid")
    scene.set_cloud_density(0.65)
    scene.set_cloud_coverage(0.55)

    hybrid_path = output_dir / "clouds_hybrid_mode.png"
    scene.render_png(hybrid_path)

    # Test 8: Static clouds (no animation)
    print("Testing static clouds (no animation)...")
    scene.set_cloud_animation_preset("static")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.6)

    static_path = output_dir / "clouds_static.png"
    scene.render_png(static_path)

    # Test 9: Gentle animation preset
    print("Testing gentle animation preset...")
    scene.set_cloud_animation_preset("gentle")
    scene.update_cloud_animation(0.0)  # Reset animation time

    gentle_path = output_dir / "clouds_gentle.png"
    scene.render_png(gentle_path)

    # Test 10: Moderate animation preset
    print("Testing moderate animation preset...")
    scene.set_cloud_animation_preset("moderate")

    moderate_path = output_dir / "clouds_moderate.png"
    scene.render_png(moderate_path)

    # Test 11: Stormy animation preset
    print("Testing stormy animation preset...")
    scene.set_cloud_animation_preset("stormy")

    stormy_path = output_dir / "clouds_stormy.png"
    scene.render_png(stormy_path)

    # Test 12: Custom wind effects
    print("Testing custom wind effects...")
    scene.set_cloud_animation_preset("moderate")
    scene.set_cloud_wind_vector(45.0, 30.0, 0.8)  # Northeast wind, strong

    custom_wind_path = output_dir / "clouds_custom_wind.png"
    scene.render_png(custom_wind_path)

    # Test 13: Large scale clouds
    print("Testing large scale clouds...")
    scene.set_cloud_scale(400.0)  # Large, distant clouds
    scene.set_cloud_density(0.5)
    scene.set_cloud_coverage(0.7)

    large_scale_path = output_dir / "clouds_large_scale.png"
    scene.render_png(large_scale_path)

    # Test 14: Small scale clouds (detailed)
    print("Testing small scale clouds...")
    scene.set_cloud_scale(100.0)  # Small, detailed clouds
    scene.set_cloud_density(0.8)
    scene.set_cloud_coverage(0.4)

    small_scale_path = output_dir / "clouds_small_scale.png"
    scene.render_png(small_scale_path)

    # Test 15: Dense cloud layer
    print("Testing dense cloud layer...")
    scene.set_cloud_scale(200.0)
    scene.set_cloud_density(1.0)  # Maximum density
    scene.set_cloud_coverage(0.9)  # High coverage

    dense_path = output_dir / "clouds_dense_layer.png"
    scene.render_png(dense_path)

    # Test 16: Sparse wispy clouds
    print("Testing sparse wispy clouds...")
    scene.set_cloud_density(0.3)  # Low density
    scene.set_cloud_coverage(0.2)  # Low coverage

    sparse_path = output_dir / "clouds_sparse_wispy.png"
    scene.render_png(sparse_path)

    # Test 17: Animation sequence demonstrating cloud movement
    print("Rendering animation sequence...")
    scene.enable_clouds(quality="medium")  # Use medium quality for smooth animation
    scene.set_cloud_render_mode("hybrid")
    scene.set_cloud_animation_preset("moderate")
    scene.set_cloud_density(0.7)
    scene.set_cloud_coverage(0.6)
    scene.set_cloud_scale(200.0)

    animation_frames = []
    for i, time_step in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]):
        scene.update_cloud_animation(time_step)
        frame_path = output_dir / f"clouds_animation_frame_{i:02d}.png"
        scene.render_png(frame_path)
        animation_frames.append(frame_path)
        print(f"  Rendered animation frame {i+1}/8")

    # Test 18: Performance benchmark at 1080p (target: 60 FPS)
    print("Running 1080p performance benchmark...")

    # Create a 1080p scene for performance testing
    perf_scene = f3d.Scene(1920, 1080, grid=256, colormap="terrain")
    perf_scene.set_height_from_r32f(heightmap)
    perf_scene.set_camera_look_at(
        eye=(3.0, 4.0, 3.0),
        target=(0.0, 1.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fovy_deg=60.0,
        znear=0.1,
        zfar=50.0
    )

    # Test different quality levels at 1080p
    quality_performance = {}

    for quality in ["low", "medium", "high"]:
        perf_scene.enable_clouds(quality=quality)
        perf_scene.set_cloud_render_mode("hybrid")  # Balanced mode
        perf_scene.set_cloud_density(0.7)
        perf_scene.set_cloud_coverage(0.6)

        # Warm up
        perf_scene.render_rgba()

        # Measure performance
        start_time = time.time()
        frame_count = 10
        for _ in range(frame_count):
            perf_scene.render_rgba()
        total_time = time.time() - start_time
        avg_frame_time = total_time / frame_count
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        quality_performance[quality] = fps
        print(f"  {quality.capitalize()} quality at 1080p: {fps:.1f} FPS")

        # Save one frame from 1080p test
        if quality == "medium":
            perf_path = output_dir / "clouds_1080p_performance.png"
            perf_scene.render_png(perf_path)

    # Get current cloud parameters for verification
    try:
        density, coverage, scale, wind_strength = scene.get_clouds_params()
        print(f"\nFinal Cloud Parameters:")
        print(f"  Density: {density:.2f}")
        print(f"  Coverage: {coverage:.2f}")
        print(f"  Scale: {scale:.1f}")
        print(f"  Wind Strength: {wind_strength:.2f}")
    except Exception as e:
        print(f"Could not retrieve cloud parameters: {e}")

    print(f"\nPerformance Results:")
    print(f"  720p Low Quality: {fps:.1f} FPS")
    for quality, fps in quality_performance.items():
        meets_target = "✓" if fps >= 60.0 else "✗"
        print(f"  1080p {quality.capitalize()} Quality: {fps:.1f} FPS {meets_target}")

    print(f"\nRendered images:")
    print(f"  Baseline (no clouds): {baseline_path}")
    print(f"  Low quality: {low_quality_path}")
    print(f"  Medium quality: {medium_quality_path}")
    print(f"  High quality: {high_quality_path}")
    print(f"  Ultra quality: {ultra_quality_path}")
    print(f"  Billboard mode: {billboard_path}")
    print(f"  Volumetric mode: {volumetric_path}")
    print(f"  Hybrid mode: {hybrid_path}")
    print(f"  Static animation: {static_path}")
    print(f"  Gentle animation: {gentle_path}")
    print(f"  Moderate animation: {moderate_path}")
    print(f"  Stormy animation: {stormy_path}")
    print(f"  Custom wind: {custom_wind_path}")
    print(f"  Large scale: {large_scale_path}")
    print(f"  Small scale: {small_scale_path}")
    print(f"  Dense layer: {dense_path}")
    print(f"  Sparse wispy: {sparse_path}")
    print(f"  1080p performance: {perf_path}")
    print(f"  Animation frames: clouds_animation_frame_*.png")

    print(f"\nRealtime cloud rendering demo completed! Check the {output_dir} directory for all variations.")
    print(f"The clouds demonstrate billboard/volumetric rendering with IBL-aware scattering")
    print(f"and performance optimization for 60 FPS at 1080p as specified in B8.")


if __name__ == "__main__":
    main()