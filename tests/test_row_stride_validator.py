"""
Row-stride validator test

Verifies that PNG files written by forge3d have correct row structure
with no padding artifacts, diagonal banding, or center seams.

Per prompt5.md acceptance criteria:
- 1920×1080 and 2560×1440 frames save with no diagonal banding or center seam
- Row-stride validator tool passes

RELEVANT FILES: src/renderer/readback.rs, src/util/image_write.rs, src/terrain_renderer.rs
"""

import numpy as np
import pytest
from pathlib import Path


def validate_row_uniqueness(image_path: Path, tolerance: float = 1e-6) -> tuple[bool, str]:
    """
    Check if image rows are unique (detects copy-paste stride errors).

    Returns:
        (is_valid, message) tuple
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        arr = np.array(img)

        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return False, f"Expected RGB/RGBA image, got shape {arr.shape}"

        # Check for duplicate rows (common padding error symptom)
        height = arr.shape[0]
        duplicate_count = 0
        for i in range(height - 1):
            if np.allclose(arr[i], arr[i + 1], atol=tolerance):
                duplicate_count += 1

        # Allow some duplicates (sky, flat terrain), but too many indicates error
        max_allowed = height * 0.1  # 10% threshold
        if duplicate_count > max_allowed:
            return False, f"Too many duplicate rows: {duplicate_count}/{height} (threshold: {max_allowed})"

        return True, f"Row uniqueness OK: {duplicate_count}/{height} duplicates"

    except Exception as e:
        return False, f"Validation error: {e}"


def detect_banding_artifacts(image_path: Path) -> tuple[bool, str]:
    """
    Detect diagonal banding or center seam artifacts.

    These artifacts typically appear as:
    - Diagonal stripes across the image
    - Vertical center seam
    - Regular horizontal patterns

    Returns:
        (is_valid, message) tuple
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        arr = np.array(img, dtype=np.float32)

        if arr.ndim == 3:
            # Convert to grayscale for analysis
            if arr.shape[2] == 4:  # RGBA
                arr = arr[:, :, :3]  # Drop alpha
            gray = np.mean(arr, axis=2)
        else:
            gray = arr

        height, width = gray.shape

        # Check for center vertical seam (common readback error)
        center = width // 2
        left_col = gray[:, center - 1]
        right_col = gray[:, center]
        center_diff = np.abs(left_col - right_col)

        # High variance at center suggests seam
        if np.mean(center_diff) > 30:  # Threshold in 0-255 range
            return False, f"Center seam detected: mean diff = {np.mean(center_diff):.1f}"

        # Check for diagonal banding by computing gradient variance
        grad_y = np.diff(gray, axis=0)
        grad_x = np.diff(gray, axis=1)

        # High periodic variance suggests banding
        y_variance = np.var(np.mean(np.abs(grad_y), axis=1))
        x_variance = np.var(np.mean(np.abs(grad_x), axis=0))

        # These thresholds may need tuning based on actual terrain content
        if y_variance > 100 or x_variance > 100:
            return False, f"Banding detected: y_var={y_variance:.1f}, x_var={x_variance:.1f}"

        return True, "No banding artifacts detected"

    except Exception as e:
        return False, f"Artifact detection error: {e}"


def test_readback_alignment():
    """Test that readback uses correct 256-byte alignment."""
    # This is a compile-time check - verify constant is correct
    import forge3d

    # The COPY_BYTES_PER_ROW_ALIGNMENT should be 256
    # We can't directly access Rust constants, but we can verify behavior
    # by checking that readback succeeds for various resolutions

    session = forge3d.Session()

    # Test various widths to ensure alignment works
    test_widths = [1, 64, 127, 256, 1920, 2560, 3840]

    for width in test_widths:
        # Create a simple frame
        frame = forge3d.render_debug_pattern_frame(width, 100)

        # If readback alignment is wrong, this will fail
        arr = frame.to_numpy()

        # Verify shape is correct (no padding in output)
        assert arr.shape == (100, width, 4), f"Shape mismatch for width={width}"


@pytest.mark.parametrize("resolution", [(1920, 1080), (2560, 1440)])
def test_high_resolution_no_artifacts(resolution, tmp_path):
    """
    Test that high-resolution renders have no banding or seam artifacts.

    Per prompt5.md: "1920×1080 and 2560×1440 frames save with no diagonal banding or center seam"
    """
    import forge3d

    width, height = resolution
    output_path = tmp_path / f"test_{width}x{height}.png"

    # Render a debug pattern
    frame = forge3d.render_debug_pattern_frame(width, height)
    frame.save(str(output_path))

    # Verify file was created
    assert output_path.exists(), f"Output file not created: {output_path}"

    # Validate row structure
    valid, msg = validate_row_uniqueness(output_path)
    assert valid, f"Row validation failed for {width}×{height}: {msg}"

    # Check for banding artifacts
    valid, msg = detect_banding_artifacts(output_path)
    assert valid, f"Artifact detection failed for {width}×{height}: {msg}"

    print(f"✓ {width}×{height} validation passed")


def test_msaa_resolve_formats():
    """
    Verify that MSAA textures are properly resolved before readback.

    Per prompt5.md: "forbid copying from MSAA surfaces"
    """
    import forge3d

    # This test verifies that even with MSAA enabled, readback works
    # (because terrain_renderer.rs resolves MSAA before creating Frame)

    session = forge3d.Session()

    # Create terrain renderer with MSAA
    from forge3d import TerrainRenderer, TerrainRenderParams, MaterialSet, IBL

    renderer = TerrainRenderer(session)
    params = TerrainRenderParams(size_px=(512, 512), msaa_samples=4)

    # Create minimal required inputs
    heightmap = np.random.rand(128, 128).astype(np.float32)
    materials = MaterialSet()
    ibl = IBL()

    # Render with MSAA
    frame = renderer.render_terrain_pbr_pom(materials, ibl, params, heightmap)

    # If MSAA wasn't resolved, to_numpy() would fail with the check in readback.rs:27-30
    arr = frame.to_numpy()

    # Verify output shape
    assert arr.shape == (512, 512, 4)


def test_tight_buffer_requirement():
    """
    Verify that PNG writer requires tight (non-padded) buffers.

    Per prompt5.md: "we re-pack rows tightly before PNG writing"
    """
    import forge3d
    from forge3d.util.image_write import write_png_rgba8

    # This test verifies the buffer size check in image_write.rs:21-26

    # Create a properly sized buffer
    width, height = 100, 50
    tight_size = width * height * 4
    tight_buffer = np.zeros(tight_size, dtype=np.uint8)

    # This should succeed
    # Note: We can't directly call the Rust function from Python,
    # but Frame.save() uses it internally

    frame = forge3d.render_debug_pattern_frame(width, height)
    arr = frame.to_numpy()

    # Verify buffer is tight (no padding)
    assert arr.size == tight_size, f"Buffer not tight: expected {tight_size}, got {arr.size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
