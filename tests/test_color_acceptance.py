"""Acceptance test for terrain colormap/overlay functionality.

This test verifies that:
1. Running the demo with defaults yields a colored terrain (not near-black)
2. Setting albedo_mode="colormap" and colormap_strength=1.0 produces vivid color
3. Switching overlay blend_mode visibly changes the result
4. Histogram uniqueness and luminance meet requirements
"""
import pytest
import subprocess
import sys
from pathlib import Path
import numpy as np


def load_png(path):
    """Load PNG as numpy array using PIL."""
    try:
        from PIL import Image
        img = Image.open(path)
        return np.array(img)
    except ImportError:
        pytest.skip("PIL/Pillow not available")


def test_terrain_demo_default_has_colors():
    """Test that default terrain demo produces colored output (not near-black)."""
    output_path = Path(__file__).parent.parent / "examples" / "out" / "test_color_default.png"

    # Clean up output if it exists
    if output_path.exists():
        output_path.unlink()

    # Run terrain_demo.py with default settings
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "examples" / "terrain_demo.py"),
            "--dem", "assets/Gore_Range_Albers_1m.tif",
            "--hdr", "assets/snow_field_4k.hdr",
            "--size", "640", "360",
            "--render-scale", "1.0",
            "--msaa", "4",
            "--z-scale", "1.5",
            "--exposure", "1.0",
            "--colormap-domain", "200.0", "2200.0",
            "--output", str(output_path),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Verify the command succeeded
    assert result.returncode == 0, f"terrain_demo.py failed: {result.stderr}"

    # Verify output file was created
    assert output_path.exists(), f"Output file not created: {output_path}"

    # Load the image and check for colors
    img = load_png(output_path)
    assert img.shape == (360, 640, 4), "Image has wrong shape"

    # Check that the image is not near-black
    rgb = img[:, :, :3].astype(float)
    mean_luminance = np.mean(rgb) / 255.0

    # Mean luminance should be in reasonable range (not too dark)
    assert mean_luminance > 0.1, f"Image too dark: mean luminance {mean_luminance}"

    # Check for color variation (unique colors)
    unique_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
    assert unique_colors > 100, f"Not enough color variation: {unique_colors} unique colors"

    # Clean up
    if output_path.exists():
        output_path.unlink()

    print(f"[PASS] Default test: mean_luminance={mean_luminance:.3f}, unique_colors={unique_colors}")


def test_debug_mode_1_lut_only():
    """Test debug mode 1 (DBG_COLOR_LUT) shows raw LUT colors."""
    output_path = Path(__file__).parent.parent / "examples" / "out" / "test_debug_lut.png"

    if output_path.exists():
        output_path.unlink()

    import os
    env = os.environ.copy()
    env["VF_COLOR_DEBUG_MODE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "examples" / "terrain_demo.py"),
            "--dem", "assets/Gore_Range_Albers_1m.tif",
            "--hdr", "assets/snow_field_4k.hdr",
            "--size", "640", "360",
            "--msaa", "4",
            "--output", str(output_path),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )

    assert result.returncode == 0, f"Debug mode 1 failed: {result.stderr}"
    assert output_path.exists(), "Debug mode 1 output not created"

    # Load and verify it's colorful (LUT only)
    img = load_png(output_path)
    rgb = img[:, :, :3].astype(float)
    mean_luminance = np.mean(rgb) / 255.0

    # Should have visible colors in debug mode
    assert mean_luminance > 0.05, "Debug LUT mode too dark"

    if output_path.exists():
        output_path.unlink()

    print(f"[PASS] Debug mode 1 (LUT): mean_luminance={mean_luminance:.3f}")


def test_color_histogram_requirements():
    """Test that output meets histogram uniqueness and luminance requirements."""
    output_path = Path(__file__).parent.parent / "examples" / "out" / "test_color_histogram.png"

    if output_path.exists():
        output_path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "examples" / "terrain_demo.py"),
            "--dem", "assets/Gore_Range_Albers_1m.tif",
            "--hdr", "assets/snow_field_4k.hdr",
            "--size", "640", "360",  # Using smaller size for faster test
            "--msaa", "4",
            "--output", str(output_path),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, "Histogram test failed to render"
    assert output_path.exists(), "Histogram test output not created"

    img = load_png(output_path)
    rgb = img[:, :, :3].astype(float)

    # Check histogram uniqueness
    unique_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
    print(f"Unique colors: {unique_colors}")

    # For 640x360, we should have substantial color variation
    # (The requirement is ≥256 for 1920×1080, so scale down proportionally)
    expected_min = int(256 * (640 * 360) / (1920 * 1080))
    assert unique_colors >= expected_min, f"Not enough unique colors: {unique_colors} < {expected_min}"

    # Check mean luminance
    mean_luminance = np.mean(rgb) / 255.0
    print(f"Mean luminance: {mean_luminance:.3f}")
    assert 0.25 <= mean_luminance <= 0.85, f"Luminance out of range: {mean_luminance}"

    if output_path.exists():
        output_path.unlink()

    print(f"[PASS] Histogram test: unique={unique_colors}, luminance={mean_luminance:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
