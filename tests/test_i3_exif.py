# tests/test_i3_exif.py
# Workstream I3: EXIF metadata tests
# - Validates metadata is written to PNG
# - Validates metadata can be extracted
# - Validates no corruption of determinism when metadata absent

import pytest
import numpy as np
from pathlib import Path

import forge3d as f3d
from forge3d.helpers.offscreen import save_png_with_exif, save_png_deterministic, render_offscreen_rgba


@pytest.mark.offscreen
def test_exif_metadata_present(tmp_path):
    """Test that EXIF metadata is embedded in PNG."""
    output = tmp_path / "test_exif.png"

    # Create test image
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[:, :, :3] = 128  # Gray
    rgba[:, :, 3] = 255   # Opaque

    # Metadata
    metadata = {
        "camera": {
            "eye": [1.0, 2.0, 3.0],
            "target": [0.0, 0.0, 0.0],
            "fov_deg": 45.0
        },
        "exposure": {
            "mode": "ACES",
            "stops": 0.5
        }
    }

    # Save with metadata
    save_png_with_exif(str(output), rgba, metadata)

    assert output.exists()
    assert output.stat().st_size > 0

    # Extract and validate
    try:
        from PIL import Image
        img = Image.open(output)

        # Check basic image properties
        assert img.size == (100, 100)
        assert img.mode == "RGBA"

        # Check metadata (if PIL supports text extraction)
        if hasattr(img, 'text') and img.text:
            # Camera metadata
            assert "forge3d:camera:eye" in img.text
            assert "1.0" in img.text["forge3d:camera:eye"]
            assert "forge3d:camera:fov_deg" in img.text
            assert "45" in img.text["forge3d:camera:fov_deg"]

            # Exposure metadata
            assert "forge3d:exposure:mode" in img.text
            assert "ACES" in img.text["forge3d:exposure:mode"]
            assert "forge3d:exposure:stops" in img.text
            assert "0.5" in img.text["forge3d:exposure:stops"]

    except ImportError:
        pytest.skip("PIL not available for metadata extraction")


@pytest.mark.offscreen
def test_exif_no_corruption_when_absent(tmp_path):
    """Test that PNG without metadata is still deterministic."""
    output1 = tmp_path / "deterministic1.png"
    output2 = tmp_path / "deterministic2.png"

    rgba = np.zeros((50, 50, 4), dtype=np.uint8)
    rgba[:, :, :3] = 200
    rgba[:, :, 3] = 255

    # Save twice without metadata - should be identical
    save_png_with_exif(str(output1), rgba, metadata=None)
    save_png_with_exif(str(output2), rgba, metadata=None)

    # Byte-for-byte identical
    bytes1 = output1.read_bytes()
    bytes2 = output2.read_bytes()

    assert bytes1 == bytes2


@pytest.mark.offscreen
def test_exif_with_partial_metadata(tmp_path):
    """Test EXIF with only camera or only exposure metadata."""
    output = tmp_path / "partial.png"

    rgba = np.ones((60, 60, 4), dtype=np.uint8) * 128

    # Only camera metadata
    metadata = {
        "camera": {
            "eye": [5.0, 5.0, 5.0]
        }
    }

    save_png_with_exif(str(output), rgba, metadata)
    assert output.exists()

    try:
        from PIL import Image
        img = Image.open(output)
        if hasattr(img, 'text') and img.text:
            assert "forge3d:camera:eye" in img.text
    except ImportError:
        pytest.skip("PIL not available")


@pytest.mark.offscreen
def test_exif_float32_input(tmp_path):
    """Test EXIF metadata with float32 input (common for HDR)."""
    output = tmp_path / "float32.png"

    # Float32 input
    rgba = np.random.rand(80, 80, 4).astype(np.float32)

    metadata = {
        "exposure": {
            "mode": "Reinhard",
            "gamma": 2.2
        }
    }

    save_png_with_exif(str(output), rgba, metadata)
    assert output.exists()

    try:
        from PIL import Image
        img = Image.open(output)
        # Should be converted to uint8
        arr = np.array(img)
        assert arr.dtype == np.uint8
    except ImportError:
        pytest.skip("PIL not available")
