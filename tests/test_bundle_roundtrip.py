# tests/test_bundle_roundtrip.py
"""Tests for scene bundle (.forge3d) save/load roundtrip."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from forge3d.bundle import (
    BundleManifest,
    BUNDLE_VERSION,
    CameraBookmark,
    RasterOverlaySpec,
    ReviewLayer,
    SceneBaseState,
    SceneState,
    SceneVariant,
    TerrainMeta,
    is_bundle,
    load_bundle,
    save_bundle,
)

pytestmark = pytest.mark.usefixtures("pro_license")


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


def test_scene_state_v2_roundtrip_preserves_variants_and_assets(tmp_path: Path):
    """TV16 scene/state.json preserves review layers, variants, and copied assets."""
    bundle_path = tmp_path / "tv16_bundle.forge3d"
    raster_path = tmp_path / "base_overlay.png"
    hdr_path = tmp_path / "studio.hdr"
    raster_path.write_bytes(b"base-overlay")
    hdr_path.write_bytes(b"studio-hdr")

    scene_state = SceneState(
        base=SceneBaseState(
            preset={"exposure": 1.25, "hdr_path": str(hdr_path)},
            camera_bookmarks=[
                CameraBookmark(
                    name="overview",
                    eye=(100.0, 200.0, 300.0),
                    target=(0.0, 0.0, 0.0),
                )
            ],
            raster_overlays=[
                RasterOverlaySpec(
                    name="ortho",
                    path=str(raster_path),
                    extent=(0.0, 0.0, 10.0, 10.0),
                    opacity=0.75,
                    z_order=2,
                )
            ],
            vector_overlays=[
                {
                    "name": "base-triangles",
                    "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "indices": [0, 1, 2],
                }
            ],
            labels=[{"text": "Summit", "world_pos": [1.0, 2.0, 3.0]}],
            scatter_batches=[{"name": "base-scatter", "levels": [], "transforms": []}],
        ),
        review_layers=[
            ReviewLayer(
                id="annotations",
                labels=[{"text": "Review note", "kind": "callout", "anchor": [2.0, 3.0, 4.0]}],
            ),
            ReviewLayer(
                id="imagery",
                raster_overlays=[
                    RasterOverlaySpec(name="secondary", path=str(raster_path), opacity=0.4)
                ],
            ),
        ],
        variants=[
            SceneVariant(
                id="review",
                active_layer_ids=["annotations"],
                preset={"exposure": 3.0},
            ),
            SceneVariant(id="imagery", active_layer_ids=["imagery"]),
        ],
        active_variant_id="review",
    )

    save_bundle(bundle_path, name="tv16_bundle", scene_state=scene_state)

    scene_file = bundle_path / "scene" / "state.json"
    assert scene_file.exists()
    with scene_file.open(encoding="utf-8") as handle:
        raw_state = json.load(handle)

    assert raw_state["base_state"]["raster_overlays"][0]["path"].startswith("assets/overlays/")
    assert raw_state["base_state"]["preset"]["hdr_path"].startswith("assets/hdri/")

    loaded = load_bundle(bundle_path)
    assert loaded.get_active_variant_id() == "review"
    assert [variant.id for variant in loaded.list_variants()] == ["review", "imagery"]
    assert [layer.id for layer in loaded.list_review_layers()] == ["annotations", "imagery"]
    assert loaded.scene_state.base.raster_overlays[0].path == str((bundle_path / raw_state["base_state"]["raster_overlays"][0]["path"]).resolve())
    assert loaded.hdr_path == (bundle_path / raw_state["base_state"]["preset"]["hdr_path"]).resolve()

    effective = loaded.effective_scene_state()
    assert effective.preset == {"exposure": 3.0}
    assert len(effective.labels) == 2
    assert len(effective.raster_overlays) == 1


def test_effective_scene_state_respects_variant_layers_and_manual_overrides(tmp_path: Path):
    """Layer overrides apply on top of the active variant and clear when variants change."""
    bundle_path = tmp_path / "review_state.forge3d"
    scene_state = SceneState(
        base=SceneBaseState(labels=[{"text": "Base", "world_pos": [0.0, 0.0, 0.0]}]),
        review_layers=[
            ReviewLayer(id="roads", labels=[{"text": "Roads", "world_pos": [1.0, 0.0, 0.0]}]),
            ReviewLayer(id="contours", labels=[{"text": "Contours", "world_pos": [2.0, 0.0, 0.0]}]),
        ],
        variants=[
            SceneVariant(id="focus", active_layer_ids=["roads"]),
            SceneVariant(id="analysis", active_layer_ids=["contours"]),
        ],
        active_variant_id="focus",
    )
    save_bundle(bundle_path, name="review_state", scene_state=scene_state)

    loaded = load_bundle(bundle_path)
    assert [label["text"] for label in loaded.effective_scene_state().labels] == ["Base", "Roads"]

    loaded.set_review_layer_visible("contours", True)
    assert [label["text"] for label in loaded.effective_scene_state().labels] == ["Base", "Roads", "Contours"]

    loaded.set_review_layer_visible("roads", False)
    assert [label["text"] for label in loaded.effective_scene_state().labels] == ["Base", "Contours"]

    loaded.apply_variant("analysis")
    assert loaded.get_active_variant_id() == "analysis"
    assert [label["text"] for label in loaded.effective_scene_state().labels] == ["Base", "Contours"]


def test_loaded_bundle_rejects_unknown_variant_and_layer_mutations(tmp_path: Path):
    """Bad layer and variant IDs fail explicitly."""
    bundle_path = tmp_path / "invalid_ops.forge3d"
    save_bundle(
        bundle_path,
        name="invalid_ops",
        scene_state=SceneState(
            review_layers=[ReviewLayer(id="notes")],
            variants=[SceneVariant(id="default", active_layer_ids=["notes"])],
            active_variant_id="default",
        ),
    )

    loaded = load_bundle(bundle_path)
    with pytest.raises(KeyError, match="Unknown scene variant"):
        loaded.apply_variant("missing")
    with pytest.raises(KeyError, match="Unknown review layer"):
        loaded.set_review_layer_visible("missing", True)


def test_load_v1_bundle_synthesizes_empty_review_registry(tmp_path: Path):
    """Version 1 bundles load into an empty TV16 registry with base mirrors preserved."""
    bundle_path = tmp_path / "legacy.forge3d"
    (bundle_path / "overlays").mkdir(parents=True)
    (bundle_path / "camera").mkdir()
    (bundle_path / "render").mkdir()
    (bundle_path / "assets" / "hdri").mkdir(parents=True)

    hdr_path = bundle_path / "assets" / "hdri" / "legacy.hdr"
    hdr_path.write_bytes(b"legacy-hdr")

    with (bundle_path / "overlays" / "vectors.geojson").open("w", encoding="utf-8") as handle:
        json.dump([{"vertices": [[0.0, 0.0, 0.0]], "indices": [0]}], handle)
    with (bundle_path / "overlays" / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump([{"text": "Legacy", "world_pos": [0.0, 0.0, 0.0]}], handle)
    with (bundle_path / "camera" / "bookmarks.json").open("w", encoding="utf-8") as handle:
        json.dump(
            [
                CameraBookmark(
                    name="legacy",
                    eye=(1.0, 2.0, 3.0),
                    target=(0.0, 0.0, 0.0),
                ).to_dict()
            ],
            handle,
            indent=2,
        )
    with (bundle_path / "render" / "preset.json").open("w", encoding="utf-8") as handle:
        json.dump({"exposure": 2.0}, handle, indent=2)

    manifest = {
        "version": 1,
        "name": "legacy",
        "created_at": "2026-03-16T00:00:00+00:00",
        "checksums": {
            "overlays/vectors.geojson": "ignored",
            "overlays/labels.json": "ignored",
            "camera/bookmarks.json": "ignored",
            "render/preset.json": "ignored",
        },
        "camera_bookmarks": [
            CameraBookmark(
                name="legacy",
                eye=(1.0, 2.0, 3.0),
                target=(0.0, 0.0, 0.0),
            ).to_dict()
        ],
        "preset": {"exposure": 2.0},
    }
    with (bundle_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    loaded = load_bundle(bundle_path, verify_checksums=False)
    assert loaded.scene_state.active_variant_id is None
    assert loaded.scene_state.review_layers == []
    assert loaded.scene_state.variants == []
    assert loaded.scene_state.base.preset is not None
    assert loaded.scene_state.base.preset["exposure"] == 2.0
    assert loaded.scene_state.base.preset["hdr_path"] == str(hdr_path.resolve())
    assert len(loaded.scene_state.base.camera_bookmarks) == 1
    assert loaded.scene_state.base.labels[0]["kind"] == "point"
