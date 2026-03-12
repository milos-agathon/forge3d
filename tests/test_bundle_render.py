"""Tests for bundle save/load flows tied to viewer-driven rendering presets."""

import hashlib
import tempfile
from pathlib import Path

import pytest

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from forge3d.bundle import BUNDLE_VERSION, BundleManifest, CameraBookmark, load_bundle, save_bundle

pytestmark = pytest.mark.usefixtures("pro_license")


def compute_image_hash(img_path: Path) -> str:
    """Compute a SHA-256 hash for an image file."""
    with open(img_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_bundle_manifest_version():
    """Bundle manifest version remains locked to schema version 1."""
    manifest = BundleManifest.new("test")
    assert manifest.version == BUNDLE_VERSION
    assert manifest.version == 1


def test_bundle_roundtrip_preserves_preset():
    """save_bundle -> load_bundle preserves a viewer-oriented preset payload."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "render_test.forge3d"

        preset = {
            "terrain_path": "terrain/dem.tif",
            "snapshot": {"width": 1920, "height": 1080},
            "camera": {"phi": 225.0, "theta": 35.0, "distance": 1.0},
            "sun": {"azimuth": 315.0, "elevation": 35.0},
        }
        bookmarks = [
            CameraBookmark(
                name="main",
                eye=(100.0, 200.0, 300.0),
                target=(0.0, 0.0, 0.0),
                fov_deg=45.0,
            )
        ]

        save_bundle(
            bundle_path,
            name="render_test",
            preset=preset,
            camera_bookmarks=bookmarks,
        )
        loaded = load_bundle(bundle_path)

        assert loaded.preset == preset
        assert len(loaded.manifest.camera_bookmarks) == 1
        assert loaded.manifest.camera_bookmarks[0].name == "main"


@pytest.mark.skipif(not HAS_PIL, reason="Requires Pillow")
def test_bundle_image_hash_verification():
    """Identical PNG bytes hash identically; changed bytes do not."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"

        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(img_path)
        hash1 = compute_image_hash(img_path)

        img.save(img_path)
        hash2 = compute_image_hash(img_path)
        assert hash1 == hash2

        img.putpixel((5, 5), (0, 255, 0))
        img.save(img_path)
        hash3 = compute_image_hash(img_path)
        assert hash1 != hash3
