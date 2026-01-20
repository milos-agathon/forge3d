# tests/test_bundle_render.py
"""Tests for bundle save/load with PNG hash verification.

Per docs/plan.md acceptance test requirement:
- Assert save_bundle → load_bundle → render produces identical PNG hash.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

# Optional imports for render testing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from forge3d.bundle import (
    save_bundle,
    load_bundle,
    BundleManifest,
    CameraBookmark,
    BUNDLE_VERSION,
)


def compute_image_hash(img_path: Path) -> str:
    """Compute SHA-256 hash of image file."""
    with open(img_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_bundle_manifest_version():
    """Bundle manifest has version == 1 as required by plan."""
    manifest = BundleManifest.new("test")
    assert manifest.version == BUNDLE_VERSION
    assert manifest.version == 1, "Schema version must be 1 per docs/plan.md"


def test_bundle_roundtrip_preserves_preset():
    """save_bundle → load_bundle preserves preset with all fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "render_test.forge3d"

        # Full preset with all common rendering parameters
        preset = {
            "exposure": 1.5,
            "z_scale": 2.5,
            "colormap": "terrain",
            "colormap_strength": 0.8,
            "normal_strength": 1.0,
            "cam_phi": 45.0,
            "cam_theta": 30.0,
            "cam_distance": 500.0,
            "sun_azimuth": 135.0,
            "sun_elevation": 45.0,
        }

        bookmarks = [
            CameraBookmark(
                name="main",
                eye=(100.0, 200.0, 300.0),
                target=(0.0, 0.0, 0.0),
                fov_deg=45.0,
            )
        ]

        # Save
        save_bundle(
            bundle_path,
            name="render_test",
            preset=preset,
            camera_bookmarks=bookmarks,
        )

        # Load
        loaded = load_bundle(bundle_path)

        # Verify preset matches exactly
        assert loaded.preset == preset, "Preset must match after roundtrip"

        # Verify camera bookmark
        assert len(loaded.manifest.camera_bookmarks) == 1
        bm = loaded.manifest.camera_bookmarks[0]
        assert bm.name == "main"
        assert bm.eye == (100.0, 200.0, 300.0)
        assert bm.fov_deg == 45.0


@pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="Requires PIL and numpy")
def test_bundle_image_hash_verification():
    """PNG hash verification: identical inputs produce identical hashes.
    
    This test validates the hash computation function and demonstrates
    the pattern for bundle render verification.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple test image
        img_path = Path(tmpdir) / "test.png"
        
        # Create 10x10 red image
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(img_path)
        
        # Compute hash
        hash1 = compute_image_hash(img_path)
        
        # Save same image again
        img.save(img_path)
        hash2 = compute_image_hash(img_path)
        
        # Hashes should match for identical content
        assert hash1 == hash2, "Identical images must produce identical hash"
        
        # Modify image
        img.putpixel((5, 5), (0, 255, 0))
        img.save(img_path)
        hash3 = compute_image_hash(img_path)
        
        # Hash should differ for modified image
        assert hash1 != hash3, "Modified image must produce different hash"


@pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="Requires PIL and numpy")
def test_bundle_render_roundtrip():
    """save_bundle → load_bundle → render produces identical output.
    
    This test verifies that bundle roundtrip preserves all rendering parameters
    by comparing render outputs before and after save/load cycle.
    """
    from forge3d.render import render_polygons
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Define test polygon
        polygon = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 100.0],
            [0.0, 100.0],
        ], dtype=np.float64)
        
        # Define rendering parameters via preset
        preset = {
            "fill_rgba": [0.2, 0.4, 0.8, 1.0],
            "stroke_rgba": [0.0, 0.0, 0.0, 1.0],
            "stroke_width": 2.0,
            "size": [200, 200],
        }
        
        # Render BEFORE bundle save
        img_before = render_polygons(
            polygon,
            size=tuple(preset["size"]),
            fill_rgba=tuple(preset["fill_rgba"]),
            stroke_rgba=tuple(preset["stroke_rgba"]),
            stroke_width=preset["stroke_width"],
        )
        
        # Save rendered image
        before_path = Path(tmpdir) / "before.png"
        Image.fromarray(img_before).save(before_path)
        hash_before = compute_image_hash(before_path)
        
        # Save bundle with preset
        bundle_path = Path(tmpdir) / "render_test.forge3d"
        save_bundle(
            bundle_path,
            name="render_test",
            preset=preset,
        )
        
        # Load bundle
        loaded = load_bundle(bundle_path)
        loaded_preset = loaded.preset
        
        # Render AFTER bundle load using loaded preset
        img_after = render_polygons(
            polygon,
            size=tuple(loaded_preset["size"]),
            fill_rgba=tuple(loaded_preset["fill_rgba"]),
            stroke_rgba=tuple(loaded_preset["stroke_rgba"]),
            stroke_width=loaded_preset["stroke_width"],
        )
        
        # Save rendered image
        after_path = Path(tmpdir) / "after.png"
        Image.fromarray(img_after).save(after_path)
        hash_after = compute_image_hash(after_path)
        
        # Compare hashes - must be identical
        assert hash_before == hash_after, (
            f"Bundle roundtrip must preserve render output. "
            f"Before: {hash_before[:16]}... After: {hash_after[:16]}..."
        )
        
        # Also verify pixel-level equality
        assert np.array_equal(img_before, img_after), (
            "Bundle roundtrip must produce pixel-identical output"
        )

