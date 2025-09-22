# examples/dof_demo.py
# Depth of Field (DOF) demo for Workstream B6.
# Showcases DOF configuration with circle-of-confusion calculations and blur effects.
# RELEVANT FILES:shaders/dof.wgsl,src/core/dof.rs,src/camera.rs,tests/test_b6_dof.py

"""Generate DOF comparison renders demonstrating depth-based blur effects."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from _import_shim import ensure_repo_import
ensure_repo_import()

try:
    import forge3d as f3d
except Exception:
    print("forge3d extension not available; skipping DOF demo.")
    import sys; sys.exit(0)


def _build_depth_test_heightmap(size: int) -> np.ndarray:
    """Create a heightmap with distinct depth layers to showcase DOF."""
    coords = np.linspace(-2.0, 2.0, size, dtype=np.float32)
    ys, xs = np.meshgrid(coords, coords, indexing="ij")

    # Create stepped terrain with different elevation levels
    height = np.zeros_like(xs)

    # Background mountains (far)
    far_mask = (xs**2 + ys**2) > 1.5
    height[far_mask] = 0.8 + 0.2 * np.sin(xs[far_mask] * 3) * np.cos(ys[far_mask] * 3)

    # Middle ground hills (focus plane)
    mid_mask = ((xs**2 + ys**2) <= 1.5) & ((xs**2 + ys**2) > 0.8)
    height[mid_mask] = 0.4 + 0.1 * np.sin(xs[mid_mask] * 5) * np.cos(ys[mid_mask] * 5)

    # Foreground features (near, should be blurred)
    near_mask = (xs**2 + ys**2) <= 0.8
    height[near_mask] = 0.1 + 0.05 * np.sin(xs[near_mask] * 8) * np.cos(ys[near_mask] * 8)

    return height.astype(np.float32)


def main() -> None:
    output_dir = Path("dof_demo_out")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create scene with MSAA enabled (required for depth buffer)
    scene = f3d.Scene(512, 512, grid=256, colormap="terrain")
    scene.set_msaa_samples(4)  # Enable depth buffer for DOF

    # Set up terrain with distinct depth layers
    heightmap = _build_depth_test_heightmap(256)
    scene.set_height_from_r32f(heightmap)

    # Position camera to see depth layers clearly
    # Eye at (0, 1.5, 2), looking at origin, focus on middle ground
    scene.set_camera_look_at(
        eye=(0.0, 1.5, 2.0),
        target=(0.0, 0.3, 0.0),  # Looking at middle elevation
        up=(0.0, 1.0, 0.0),
        fovy_deg=60.0,
        znear=0.1,
        zfar=10.0
    )

    # Render baseline without DOF
    print("Rendering baseline (no DOF)...")
    baseline_path = output_dir / "dof_demo_baseline.png"
    scene.render_png(baseline_path)

    # Enable DOF with medium quality
    print("Enabling DOF...")
    scene.enable_dof(quality="medium")

    # Test 1: Focus on middle ground with shallow depth of field (f/2.8)
    print("Rendering shallow DOF (f/2.8, focus=2.5)...")
    scene.set_dof_f_stop(2.8)
    scene.set_dof_focus_distance(2.5)  # Focus on middle ground
    scene.set_dof_focal_length(50.0)   # 50mm lens
    shallow_path = output_dir / "dof_demo_shallow.png"
    scene.render_png(shallow_path)

    # Test 2: Deep depth of field (f/11)
    print("Rendering deep DOF (f/11, focus=2.5)...")
    scene.set_dof_f_stop(11.0)
    deep_path = output_dir / "dof_demo_deep.png"
    scene.render_png(deep_path)

    # Test 3: Focus on foreground (near focus)
    print("Rendering near focus (f/2.8, focus=1.2)...")
    scene.set_dof_f_stop(2.8)
    scene.set_dof_focus_distance(1.2)  # Focus on foreground
    near_focus_path = output_dir / "dof_demo_near_focus.png"
    scene.render_png(near_focus_path)

    # Test 4: Focus on background (far focus)
    print("Rendering far focus (f/2.8, focus=4.0)...")
    scene.set_dof_focus_distance(4.0)  # Focus on background
    far_focus_path = output_dir / "dof_demo_far_focus.png"
    scene.render_png(far_focus_path)

    # Test 5: High quality DOF with gather method
    print("Rendering high quality DOF (ultra quality, gather method)...")
    scene.disable_dof()
    scene.enable_dof(quality="ultra")
    scene.set_dof_f_stop(1.8)  # Very shallow depth of field
    scene.set_dof_focus_distance(2.5)
    scene.set_dof_method("gather")  # High quality method
    ultra_path = output_dir / "dof_demo_ultra_quality.png"
    scene.render_png(ultra_path)

    # Test 6: Circle of confusion visualization
    print("Rendering circle of confusion visualization...")
    scene.set_dof_show_coc(True)
    coc_path = output_dir / "dof_demo_coc_visualization.png"
    scene.render_png(coc_path)
    scene.set_dof_show_coc(False)

    # Test 7: Bokeh rotation effect
    print("Rendering with bokeh rotation...")
    scene.set_dof_bokeh_rotation(45.0)  # 45 degree rotation
    bokeh_path = output_dir / "dof_demo_bokeh_rotation.png"
    scene.render_png(bokeh_path)

    # Test 8: Performance comparison - separable blur
    print("Rendering with separable blur method...")
    scene.set_dof_method("separable")  # Faster method
    scene.set_dof_bokeh_rotation(0.0)  # Reset rotation
    separable_path = output_dir / "dof_demo_separable.png"
    scene.render_png(separable_path)

    # Print DOF parameters for verification
    aperture, focus_distance, focal_length = scene.get_dof_params()
    f_stop = f3d.camera_aperture_to_f_stop(aperture)

    print(f"\nDOF Parameters:")
    print(f"  F-stop: f/{f_stop:.1f}")
    print(f"  Focus distance: {focus_distance:.1f}")
    print(f"  Focal length: {focal_length:.1f}mm")

    # Calculate depth of field range
    near, far = f3d.camera_depth_of_field_range(
        focal_length=focal_length,
        f_stop=f_stop,
        focus_distance=focus_distance
    )
    print(f"  Depth of field: {near:.2f} - {far:.2f}")

    # Calculate hyperfocal distance
    hyperfocal = f3d.camera_hyperfocal_distance(
        focal_length=focal_length,
        f_stop=f_stop
    )
    print(f"  Hyperfocal distance: {hyperfocal:.2f}")

    print(f"\nRendered images:")
    print(f"  Baseline (no DOF): {baseline_path}")
    print(f"  Shallow DOF (f/2.8): {shallow_path}")
    print(f"  Deep DOF (f/11): {deep_path}")
    print(f"  Near focus: {near_focus_path}")
    print(f"  Far focus: {far_focus_path}")
    print(f"  Ultra quality: {ultra_path}")
    print(f"  CoC visualization: {coc_path}")
    print(f"  Bokeh rotation: {bokeh_path}")
    print(f"  Separable blur: {separable_path}")

    print(f"\nDOF demo completed! Check the {output_dir} directory for comparison images.")


if __name__ == "__main__":
    main()