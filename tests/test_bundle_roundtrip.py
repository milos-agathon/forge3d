# tests/test_bundle_roundtrip.py
"""Tests for scene bundle (.forge3d) save/load roundtrip."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forge3d.bundle import (
    save_bundle,
    load_bundle,
    is_bundle,
    BundleManifest,
    CameraBookmark,
    TerrainMeta,
    BUNDLE_VERSION,
)


def test_manifest_roundtrip():
    """BundleManifest serializes and deserializes without data loss."""
    manifest = BundleManifest.new("test_bundle")
    manifest.description = "Test description"
    manifest.checksums["terrain/dem.tif"] = "abc123def456"
    manifest.terrain = TerrainMeta(
        dem_path="terrain/dem.tif",
        crs="EPSG:32610",
        domain=(0.0, 1000.0),
        colormap="viridis",
    )
    manifest.camera_bookmarks = [
        CameraBookmark(
            name="default",
            eye=(100.0, 200.0, 300.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_deg=45.0,
        )
    ]
    manifest.preset = {"exposure": 1.5, "z_scale": 2.0}

    # Serialize
    data = manifest.to_dict()
    json_str = json.dumps(data)

    # Deserialize
    loaded_data = json.loads(json_str)
    loaded = BundleManifest.from_dict(loaded_data)

    assert loaded.version == BUNDLE_VERSION
    assert loaded.name == "test_bundle"
    assert loaded.description == "Test description"
    assert loaded.checksums["terrain/dem.tif"] == "abc123def456"
    assert loaded.terrain is not None
    assert loaded.terrain.dem_path == "terrain/dem.tif"
    assert loaded.terrain.crs == "EPSG:32610"
    assert loaded.terrain.domain == (0.0, 1000.0)
    assert loaded.terrain.colormap == "viridis"
    assert len(loaded.camera_bookmarks) == 1
    assert loaded.camera_bookmarks[0].name == "default"
    assert loaded.camera_bookmarks[0].eye == (100.0, 200.0, 300.0)
    assert loaded.preset == {"exposure": 1.5, "z_scale": 2.0}


def test_save_load_bundle_minimal():
    """save_bundle -> load_bundle roundtrip with minimal data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "test.forge3d"

        # Save minimal bundle
        result_path = save_bundle(
            bundle_path,
            name="minimal_test",
            preset={"exposure": 1.0, "z_scale": 2.5},
        )

        assert result_path.exists()
        assert is_bundle(result_path)
        assert (result_path / "manifest.json").exists()

        # Load bundle
        loaded = load_bundle(result_path)

        assert loaded.manifest.name == "minimal_test"
        assert loaded.manifest.version == BUNDLE_VERSION
        assert loaded.preset == {"exposure": 1.0, "z_scale": 2.5}


def test_save_load_bundle_with_preset():
    """save_bundle -> load_bundle preserves preset configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "preset_test.forge3d"

        preset = {
            "exposure": 1.5,
            "z_scale": 3.0,
            "colormap": "viridis",
            "colormap_strength": 0.7,
            "normal_strength": 1.2,
            "cam_phi": 45.0,
            "cam_theta": 30.0,
        }

        save_bundle(bundle_path, name="preset_test", preset=preset)
        loaded = load_bundle(bundle_path)

        assert loaded.preset == preset


def test_save_load_bundle_with_camera_bookmarks():
    """save_bundle -> load_bundle preserves camera bookmarks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "camera_test.forge3d"

        bookmarks = [
            CameraBookmark(
                name="top_down",
                eye=(0.0, 1000.0, 0.0),
                target=(0.0, 0.0, 0.0),
            ),
            CameraBookmark(
                name="oblique",
                eye=(500.0, 500.0, 500.0),
                target=(0.0, 100.0, 0.0),
                fov_deg=60.0,
            ),
        ]

        save_bundle(bundle_path, name="camera_test", camera_bookmarks=bookmarks)
        loaded = load_bundle(bundle_path)

        assert len(loaded.manifest.camera_bookmarks) == 2
        assert loaded.manifest.camera_bookmarks[0].name == "top_down"
        assert loaded.manifest.camera_bookmarks[1].name == "oblique"
        assert loaded.manifest.camera_bookmarks[1].fov_deg == 60.0


def test_is_bundle_false_for_nonexistent():
    """is_bundle returns False for non-existent paths."""
    assert not is_bundle(Path("/nonexistent/path"))


def test_is_bundle_false_for_file():
    """is_bundle returns False for regular files."""
    with tempfile.NamedTemporaryFile() as f:
        assert not is_bundle(Path(f.name))


def test_is_bundle_false_for_dir_without_manifest():
    """is_bundle returns False for directories without manifest.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert not is_bundle(Path(tmpdir))


def test_load_bundle_rejects_invalid():
    """load_bundle raises for invalid bundle directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Not a valid bundle"):
            load_bundle(Path(tmpdir))


def test_manifest_version_check():
    """BundleManifest rejects future versions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = Path(tmpdir) / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({"version": 999, "name": "future", "created_at": "2025-01-01"}, f)

        with pytest.raises(ValueError, match="version 999"):
            BundleManifest.load(manifest_path)


def test_checksum_verification():
    """load_bundle verifies file checksums."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "checksum_test.forge3d"

        # Create bundle with preset
        save_bundle(bundle_path, name="checksum_test", preset={"test": True})

        # Corrupt the preset file
        preset_file = bundle_path / "render" / "preset.json"
        with open(preset_file, "w") as f:
            f.write('{"corrupted": true}')

        # Load should fail checksum verification
        with pytest.raises(ValueError, match="Checksum mismatch"):
            load_bundle(bundle_path, verify_checksums=True)

        # Load without verification should succeed
        loaded = load_bundle(bundle_path, verify_checksums=False)
        assert loaded.preset == {"corrupted": True}
