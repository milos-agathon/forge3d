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


# ------------------------------
# Milestone 4 — Unit tests (4.1)
# ------------------------------

def _import_m4_generate():
    import importlib.util, sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    m4_path = repo_root / "examples" / "m4_generate.py"
    if not m4_path.exists():
        raise ImportError(f"m4_generate.py not found at {m4_path}")
    spec = importlib.util.spec_from_file_location("m4_generate", str(m4_path))
    if spec is None or spec.loader is None:
        raise ImportError("Failed to import examples/m4_generate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules['m4_generate'] = mod
    spec.loader.exec_module(mod)
    return mod


def test_p4_lut_bounds_random_samples():
    """LUT bounds: random UV samples produce [0,1] with no NaNs."""
    m4 = _import_m4_generate()
    lut = m4.compute_dfg_lut(128, m4.DFG_LUT_SAMPLES)
    H, W, C = lut.shape
    assert C == 2
    # Sample 256 random points
    rng = np.random.default_rng(123)
    xs = rng.integers(0, W, size=256)
    ys = rng.integers(0, H, size=256)
    samples = lut[ys, xs]
    assert np.isfinite(samples).all()
    assert (samples >= 0.0).all() and (samples <= 1.0).all()


def test_p4_cache_roundtrip_metric():
    """Cache round-trip determinism using CPU reference path (backend-agnostic).

    We use examples/m4_generate.py to compute small artifacts twice and
    assert determinism at the array level. Production path remains GPU-first.
    """
    import tempfile
    m4 = _import_m4_generate()
    with tempfile.TemporaryDirectory() as _td:
        hdr, _ = m4.load_hdr_environment(m4.HDR_DEFAULT, force_synthetic=True)
        base = 128
        irr = 32
        brdf = 128

        # First compute
        faces1, _ = m4.equirect_to_cubemap(hdr, base)
        pre1, _geoms1, _ = m4.compute_prefilter_chain(
            hdr, base, m4.PREFILTER_SAMPLES_TOP, m4.PREFILTER_SAMPLES_BOTTOM
        )
        irr1 = m4.build_irradiance_cubemap(hdr, irr, m4.IRRADIANCE_SAMPLES)
        lut1 = m4.compute_dfg_lut(brdf, m4.DFG_LUT_SAMPLES)

        # Second compute
        faces2, _ = m4.equirect_to_cubemap(hdr, base)
        pre2, _geoms2, _ = m4.compute_prefilter_chain(
            hdr, base, m4.PREFILTER_SAMPLES_TOP, m4.PREFILTER_SAMPLES_BOTTOM
        )
        irr2 = m4.build_irradiance_cubemap(hdr, irr, m4.IRRADIANCE_SAMPLES)
        lut2 = m4.compute_dfg_lut(brdf, m4.DFG_LUT_SAMPLES)

        # Hashes must match across runs
        h1 = m4.hash_prefilter_levels(pre1)
        h2 = m4.hash_prefilter_levels(pre2)
        assert h1 == h2
        assert np.allclose(faces1, faces2)
        assert np.allclose(irr1, irr2)
        assert np.allclose(lut1, lut2)


def test_p4_quality_mapping_sizes_and_mips():
    """Quality mapping: Low/Medium/High → exact sizes/mips via Scene API."""
    # Scene is GPU-backed; skip gracefully if native module or GPU not available
    if not hasattr(f3d, "Scene"):
        pytest.skip("forge3d native Scene unavailable")
    try:
        if hasattr(f3d, "has_gpu") and not f3d.has_gpu():
            pytest.skip("No GPU available for Scene-based IBL test")
    except Exception:
        pass

    scene = f3d.Scene(128, 128, grid=8)
    scene.enable_ibl('low')

    # Map expected sizes per quality from src/core/ibl.rs
    expected = {
        'low':   {'irr': 64,  'spec': 128,  'mips': 5},
        'medium':{'irr': 128, 'spec': 256,  'mips': 6},
        'high':  {'irr': 256, 'spec': 512,  'mips': 7},
    }

    # Load a tiny synthetic environment
    m4 = _import_m4_generate()
    env = m4.generate_synthetic_environment(64, 32)
    scene.load_environment_map(env.flatten().tolist(), env.shape[1], env.shape[0])
    scene.generate_ibl_textures()

    for quality, exp in expected.items():
        scene.set_ibl_quality(quality)
        scene.generate_ibl_textures()
        irr_info, spec_info, _ = scene.get_ibl_texture_info()
        assert f"{exp['irr']}x{exp['irr']}" in irr_info
        assert f"{exp['spec']}x{exp['spec']}" in spec_info
        assert f"{exp['mips']} mips" in spec_info
