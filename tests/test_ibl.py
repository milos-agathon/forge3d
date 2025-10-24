"""Tests for IBL class (Task 2.4)."""
import pytest
import forge3d as f3d
import os
import tempfile
import numpy as np


def create_test_hdr(path, width=64, height=32):
    """Create a simple test HDR file for testing."""
    # Create a simple HDR header and minimal data
    with open(path, 'wb') as f:
        # Write Radiance HDR header
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n")
        f.write(b"\n")
        f.write(f"-Y {height} +X {width}\n".encode())

        # Write uncompressed scanlines (old-style)
        # Each pixel is 4 bytes (RGBE)
        for y in range(height):
            for x in range(width):
                # Create a simple gradient
                r = int((x / width) * 255)
                g = int((y / height) * 255)
                b = 128
                e = 128  # Mid-range exponent
                f.write(bytes([r, g, b, e]))


@pytest.fixture
def test_hdr_file():
    """Create a temporary test HDR file."""
    with tempfile.NamedTemporaryFile(suffix=".hdr", delete=False) as tmp:
        create_test_hdr(tmp.name)
        yield tmp.name
    # Cleanup
    if os.path.exists(tmp.name):
        os.unlink(tmp.name)


def test_ibl_creation(test_hdr_file):
    """Test basic IBL creation from HDR file."""
    ibl = f3d.IBL.from_hdr(test_hdr_file, intensity=1.0)
    assert ibl is not None
    assert ibl.intensity == 1.0
    assert ibl.rotation_deg == 0.0


def test_ibl_with_custom_params(test_hdr_file):
    """Test IBL with custom intensity and rotation."""
    ibl = f3d.IBL.from_hdr(
        test_hdr_file,
        intensity=2.5,
        rotate_deg=45.0,
        quality="medium"
    )
    assert ibl is not None
    assert ibl.intensity == 2.5
    assert ibl.rotation_deg == 45.0
    assert ibl.quality == "medium"


def test_ibl_quality_levels(test_hdr_file):
    """Test all IBL quality levels."""
    for quality in ["low", "medium", "high", "ultra"]:
        ibl = f3d.IBL.from_hdr(test_hdr_file, quality=quality)
        assert ibl is not None
        assert ibl.quality == quality


def test_ibl_invalid_quality(test_hdr_file):
    """Test that invalid quality raises ValueError."""
    with pytest.raises(ValueError, match="Invalid quality level"):
        f3d.IBL.from_hdr(test_hdr_file, quality="invalid")


def test_ibl_invalid_intensity(test_hdr_file):
    """Test that negative intensity raises ValueError."""
    with pytest.raises(ValueError, match="intensity must be >= 0"):
        f3d.IBL.from_hdr(test_hdr_file, intensity=-1.0)


def test_ibl_missing_file():
    """Test that missing file raises IOError."""
    with pytest.raises(IOError):
        f3d.IBL.from_hdr("nonexistent_file.hdr")


def test_ibl_properties(test_hdr_file):
    """Test IBL property getters and setters."""
    ibl = f3d.IBL.from_hdr(test_hdr_file, intensity=1.0, rotate_deg=30.0)

    # Test getters
    assert ibl.path == test_hdr_file
    assert ibl.intensity == 1.0
    assert ibl.rotation_deg == 30.0

    # Test setters (use property assignment in Python)
    ibl.intensity = 2.0
    assert ibl.intensity == 2.0

    ibl.rotation_deg = 90.0
    assert ibl.rotation_deg == 90.0


def test_ibl_intensity_setter_validation(test_hdr_file):
    """Test that setting negative intensity raises ValueError."""
    ibl = f3d.IBL.from_hdr(test_hdr_file)

    with pytest.raises(ValueError, match="intensity must be >= 0"):
        ibl.intensity = -1.0


def test_ibl_dimensions(test_hdr_file):
    """Test IBL dimensions property."""
    ibl = f3d.IBL.from_hdr(test_hdr_file)

    dims = ibl.dimensions
    assert dims is not None
    assert dims[0] == 64  # width
    assert dims[1] == 32  # height


def test_ibl_repr(test_hdr_file):
    """Test IBL string representation."""
    ibl = f3d.IBL.from_hdr(test_hdr_file, intensity=1.5, rotate_deg=45.0)
    repr_str = repr(ibl)

    assert "IBL" in repr_str
    assert "path=" in repr_str
    assert "intensity=" in repr_str
    assert "rotation_deg=" in repr_str
    assert "quality=" in repr_str
