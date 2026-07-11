from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
box = pytest.importorskip("shapely.geometry").box


def _load_example_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "population_ghsl" / "romania_builtup_cover_3d.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("romania_builtup_cover_3d", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_romania_example_uses_romania_constants():
    module = _load_example_module()

    assert module.COUNTRY_A3 == "ROU"
    assert module.COUNTRY_NAME == "Romania"
    assert module.OUT_DIR.name == "romania_builtup_cover"
    assert module.CACHE_DIR.name == "romania_builtup_cover"
    assert module.DEM_ZOOM == 10
    assert module.BUILTUP_PATH.name == "GHS_BUILT_S_NRES_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
    assert module.OVERLAY_CACHE_NAME == "romania_builtup_overlay_v15.png"
    assert module.TITLE_LINES == ["Built-up areas", "ROMANIA"]
    assert any("built-up surface" in line for line in module.CAPTION_LINES)
    assert any("30 arcsec" in line for line in module.CAPTION_LINES)


def test_romania_caption_matches_reference_output_text():
    module = _load_example_module()

    assert module.CAPTION_LINES[0] == "©2024 Milos Popovic (https://milospopovic.net)"


def test_romania_example_does_not_patch_render_with_external_reference_image():
    path = Path(__file__).resolve().parents[1] / "examples" / "population_ghsl" / "romania_builtup_cover_3d.py"
    source = path.read_text(encoding="utf-8")

    assert "REFERENCE_IMAGE" not in source
    assert "3d-builtup-romania-light.png" not in source
    assert "_apply_reference_terrain_component" not in source


def test_requested_batch_presets_cover_target_regions():
    module = _load_example_module()

    assert module.REQUESTED_BATCH == (
        "italy",
        "germany",
        "france",
        "uk_ireland",
        "mainland_usa",
        "japan",
        "switzerland",
        "poland",
        "africa",
        "brazil",
        "argentina",
    )
    presets = module.REGION_PRESETS
    assert presets["uk_ireland"].country_a3 == ("GBR", "IRL")
    assert presets["mainland_usa"].bbox == pytest.approx((-125.0, 24.0, -66.0, 50.0))
    assert presets["france"].bbox == pytest.approx((-6.0, 41.0, 10.5, 52.0))
    assert presets["africa"].continent == "Africa"
    assert presets["mainland_usa"].target_crs == "EPSG:3857"
    assert presets["japan"].target_crs == "EPSG:3857"
    assert presets["brazil"].target_crs == "EPSG:3857"
    assert presets["argentina"].target_crs == "EPSG:3857"


def test_country_geometry_selects_romania(tmp_path):
    module = _load_example_module()
    boundary_path = tmp_path / "countries.geojson"
    countries = gpd.GeoDataFrame(
        {
            "ADM0_A3": ["ROU", "BGR"],
            "ADMIN": ["Romania", "Bulgaria"],
            "geometry": [box(20.0, 43.0, 30.0, 49.0), box(22.0, 41.0, 29.0, 44.0)],
        },
        crs="EPSG:4326",
    )
    countries.to_file(boundary_path, driver="GeoJSON")

    geometry = module._country_geometry(boundary_path, "EPSG:4326")

    assert geometry.bounds == pytest.approx((20.0, 43.0, 30.0, 49.0))


def test_terrain_style_constants_match_reference_art_direction():
    module = _load_example_module()

    assert module.HDR_URL == "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/brown_photostudio_02_4k.hdr"
    assert module.HDR.name == "brown_photostudio_02_4k.hdr"
    assert module.SNAPSHOT_SIZE == (4000, 4000)
    assert tuple(module.BUILTUP_COLOR.astype(int)) == (255, 211, 1)
    assert module.BUILTUP_SHADE_FLOOR == pytest.approx(0.70)
    assert module.BUILTUP_SHADE_GAIN == pytest.approx(0.30)
    np.testing.assert_array_equal(
        module.TERRAIN_PALETTE.astype(int),
        np.array(
            [
                [17, 40, 54],
                [31, 71, 98],
                [119, 157, 182],
            ],
            dtype=np.uint8,
        ),
    )
    assert module.CAMERA["zscale"] == pytest.approx(0.145)
    assert module.TERRAIN["sun_elevation"] == pytest.approx(24.0)
    assert module.TERRAIN["sun_intensity"] == pytest.approx(1.95)
    assert module.TERRAIN["ambient"] == pytest.approx(0.58)
    assert module.TERRAIN["shadow"] == pytest.approx(0.42)
    assert module.PBR["exposure"] == pytest.approx(1.08)
    assert module.PBR["ibl_intensity"] == pytest.approx(1.3)
    assert module.PBR["hdr_rotate_deg"] == pytest.approx(225.0)
    assert module.PBR["normal_strength"] == pytest.approx(1.10)
    assert module.RELIEF_TERRAIN["sun_elevation"] == pytest.approx(18.0)
    assert module.RELIEF_TERRAIN["sun_intensity"] == pytest.approx(3.70)
    assert module.RELIEF_TERRAIN["ambient"] == pytest.approx(0.36)
    assert module.RELIEF_TERRAIN["shadow"] == pytest.approx(0.78)
    assert module.RELIEF_PBR["normal_strength"] == pytest.approx(1.85)
    assert module.RELIEF_PBR["height_ao"]["strength"] == pytest.approx(0.42)
    assert module.RELIEF_PBR["sun_visibility"]["mode"] == "hard"
    assert module.TERRAIN_CAST_SHADOW["enabled"] is True
    assert module.TERRAIN_CAST_SHADOW["zscale"] == pytest.approx(20.0)
    assert module.TERRAIN_CAST_SHADOW["darkness"] == pytest.approx(0.72)
    assert module.LAYOUT["map_target_width"] == pytest.approx(1.01)
    assert module.LAYOUT["map_target_height"] == pytest.approx(0.735)
    assert module.LAYOUT["map_scale_x"] == pytest.approx(1.0)
    assert module.LAYOUT["map_scale_y"] == pytest.approx(1.0)
    assert module.LAYOUT["map_x"] == pytest.approx(0.004)
    assert module.LAYOUT["map_y"] == pytest.approx(0.152)
    assert module.LAYOUT["title_y"] == pytest.approx(0.018)
    assert module.COMPOSITE["terrain_gap_filter_size"] == 55


def test_layout_resize_can_match_reference_footprint_aspect(monkeypatch):
    module = _load_example_module()
    monkeypatch.setitem(module.LAYOUT, "map_target_width", 0.50)
    monkeypatch.setitem(module.LAYOUT, "map_target_height", 0.50)
    monkeypatch.setitem(module.LAYOUT, "map_scale_x", 0.90)
    monkeypatch.setitem(module.LAYOUT, "map_scale_y", 1.10)
    subject = module.Image.new("RGBA", (100, 100), (0, 0, 0, 255))

    resized = module._resize_subject_to_layout(subject, (200, 200))

    assert resized.size == (90, 110)


def test_compose_snapshot_keeps_background_outside_subject_white(tmp_path, monkeypatch):
    module = _load_example_module()
    monkeypatch.setattr(module, "SNAPSHOT_SIZE", (120, 120))
    monkeypatch.setattr(module, "TITLE_LINES", ["", ""])
    monkeypatch.setattr(module, "CAPTION_LINES", ())
    monkeypatch.setitem(module.LAYOUT, "map_target_width", 0.60)
    monkeypatch.setitem(module.LAYOUT, "map_target_height", 0.60)
    monkeypatch.setitem(module.LAYOUT, "map_scale_x", 1.0)
    monkeypatch.setitem(module.LAYOUT, "map_scale_y", 1.0)
    monkeypatch.setitem(module.LAYOUT, "map_x", 0.20)
    monkeypatch.setitem(module.LAYOUT, "map_y", 0.20)

    raw = np.full((32, 32, 4), 255, dtype=np.uint8)
    raw[10:22, 10:22, :3] = module.TERRAIN_PALETTE[1]
    raw_image = module.Image.fromarray(raw, mode="RGBA")
    subject = module._resize_subject_to_layout(module._crop_subject(raw_image), module.SNAPSHOT_SIZE)
    map_x = round(module.SNAPSHOT_SIZE[0] * module.LAYOUT["map_x"])
    map_y = round(module.SNAPSHOT_SIZE[1] * module.LAYOUT["map_y"])
    subject_mask = module.Image.new("L", module.SNAPSHOT_SIZE, 0)
    subject_mask.paste(subject.getchannel("A"), (map_x, map_y))

    output_path = tmp_path / "composed.png"
    module._compose_snapshot(raw_image, output_path)

    out = np.asarray(module.Image.open(output_path).convert("RGB"), dtype=np.uint8)
    outside_subject = np.asarray(subject_mask, dtype=np.uint8) == 0
    assert np.all(out[outside_subject] == 255)


def test_dem_cast_shadow_darkens_pixels_behind_a_ridge():
    module = _load_example_module()
    dem = np.zeros((7, 9), dtype=np.float32)
    dem[:, 4] = 1000.0
    valid = np.ones_like(dem, dtype=bool)

    shadow = module._dem_cast_shadow_multiplier(
        dem,
        valid,
        sun_azimuth=90.0,
        sun_elevation=18.0,
        max_steps=5,
        zscale=20.0,
        darkness=0.72,
    )

    assert shadow[3, 2] < 0.45
    assert shadow[3, 6] > 0.95
    assert shadow[3, 4] > 0.95


def test_dem_cast_shadow_uses_rayshader_zscale_clearance():
    module = _load_example_module()
    dem = np.zeros((5, 7), dtype=np.float32)
    dem[:, 3] = 5.0
    valid = np.ones_like(dem, dtype=bool)

    shadow = module._dem_cast_shadow_multiplier(
        dem,
        valid,
        sun_azimuth=90.0,
        sun_elevation=18.0,
        max_steps=4,
        zscale=20.0,
        darkness=0.72,
    )

    assert shadow[2, 2] > 0.95


def test_builtup_rgba_preserves_yellow_visibility_under_cast_shadow(monkeypatch):
    module = _load_example_module()
    builtup = np.array([[1.0]], dtype=np.float32)
    dem = np.array([[100.0]], dtype=np.float32)
    valid = np.array([[True]])
    monkeypatch.setattr(module.base_viewer, "_height_shade_from_dem", lambda heightmap: np.ones_like(heightmap, dtype=np.float32))
    monkeypatch.setattr(module, "_reference_cast_shadow", lambda dem_data, valid_mask: np.full_like(dem_data, 0.1, dtype=np.float32))

    rgba = module._builtup_rgba(builtup, dem, valid)

    np.testing.assert_array_equal(rgba[0, 0, :3], module.BUILTUP_COLOR)


def test_terrain_base_rgb_uses_reference_height_palette(monkeypatch):
    module = _load_example_module()
    dem = np.array(
        [
            [0.0, 50.0, 100.0],
            [150.0, 200.0, 250.0],
            [300.0, 350.0, 400.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(dem, dtype=bool)
    monkeypatch.setattr(module, "_reference_cast_shadow", lambda dem_data, valid_mask: np.ones_like(dem_data, dtype=np.float32))

    rgb = module._terrain_base_rgb(dem, valid)

    np.testing.assert_array_equal(rgb[0, 0], np.array([17, 40, 54], dtype=np.uint8))
    np.testing.assert_array_equal(rgb[-1, -1], np.array([119, 157, 182], dtype=np.uint8))
    assert int(rgb[-1, -1].mean()) > int(rgb[0, 0].mean()) + 100


def test_reference_terrain_tone_maps_blue_luminance_to_target_distribution():
    module = _load_example_module()
    width = 128
    ramp = np.linspace(0.08, 0.92, width, dtype=np.float32)
    rgb = np.zeros((8, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.round(40.0 * ramp[None, :]).astype(np.uint8)
    rgb[:, :, 1] = np.round(95.0 * ramp[None, :]).astype(np.uint8)
    rgb[:, :, 2] = np.round(150.0 * ramp[None, :]).astype(np.uint8)
    rgba = np.dstack([rgb, np.full((8, width), 255, dtype=np.uint8)])

    toned = module._apply_reference_terrain_tone(module.Image.fromarray(rgba, mode="RGBA"))
    toned_rgb = np.asarray(toned.convert("RGB"), dtype=np.float32)
    luminance = (
        toned_rgb[:, :, 0] * 0.2126
        + toned_rgb[:, :, 1] * 0.7152
        + toned_rgb[:, :, 2] * 0.0722
    )

    expected = np.asarray(module.COMPOSITE["terrain_luma_targets"], dtype=np.float32)
    actual = np.percentile(
        luminance,
        np.asarray(module.COMPOSITE["terrain_luma_quantiles"], dtype=np.float32),
    )
    np.testing.assert_allclose(actual[[0, 3, 4, 5]], expected[[0, 3, 4, 5]], atol=4.0)
    assert np.all(np.diff(actual) > 0.0)


def test_reference_terrain_tone_compresses_pale_terrain_highlights():
    module = _load_example_module()
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = (220, 225, 225)
    rgba[:, :, 3] = 255
    rgba[0, 0, :3] = module.BUILTUP_COLOR

    toned = module._apply_reference_terrain_tone(module.Image.fromarray(rgba, mode="RGBA"))
    arr = np.asarray(toned.convert("RGB"), dtype=np.float32)
    highlight_luma = arr[2, 2] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    assert highlight_luma < 170.0
    np.testing.assert_array_equal(arr[0, 0], module.BUILTUP_COLOR.astype(np.float32))


def test_reference_terrain_tone_tints_neutral_shadow_terrain_blue():
    module = _load_example_module()
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = (92, 94, 96)
    rgba[:, :, 3] = 255
    rgba[0, 0, :3] = module.BUILTUP_COLOR

    toned = module._apply_reference_terrain_tone(module.Image.fromarray(rgba, mode="RGBA"))
    arr = np.asarray(toned.convert("RGB"), dtype=np.float32)

    assert arr[2, 2, 2] > arr[2, 2, 0] + 20.0
    assert arr[0, 0, 2] < 20.0


def test_crop_subject_makes_pale_interior_terrain_highlights_opaque():
    module = _load_example_module()
    rgba = np.full((11, 11, 4), 255, dtype=np.uint8)
    rgba[2:9, 2:9, :3] = module.TERRAIN_PALETTE[1].astype(np.uint8)
    rgba[5, 5, :3] = (245, 245, 246)

    subject = module._crop_subject(module.Image.fromarray(rgba, mode="RGBA"))
    alpha = np.asarray(subject.getchannel("A"), dtype=np.uint8)
    center = (alpha.shape[0] // 2, alpha.shape[1] // 2)

    assert alpha[center] >= 240


def test_final_canvas_terrain_tone_protects_text_and_builtup():
    module = _load_example_module()
    image = module.Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[1, 1, :3] = (30, 73, 96)  # title-like text above the map band
    arr[4:6, 1:7, :3] = (220, 225, 225)  # pale terrain highlights
    arr[6, 1, :3] = (150, 150, 150)  # neutral map artifact
    arr[6, 2, :3] = (230, 230, 230)  # bright neutral map artifact
    arr[6, 3, :3] = (120, 210, 252)  # over-bright cyan terrain artifact
    arr[4, 3, :3] = module.BUILTUP_COLOR

    toned = module._apply_final_canvas_terrain_tone(module.Image.fromarray(arr, mode="RGBA"))
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)
    terrain_luma = out[5, 5] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    bright_artifact_luma = out[6, 2] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    cyan_artifact_luma = out[6, 3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    assert terrain_luma < 170.0
    assert bright_artifact_luma < 170.0
    assert cyan_artifact_luma < 170.0
    np.testing.assert_array_equal(out[1, 1], np.array([30, 73, 96], dtype=np.float32))
    assert out[6, 1, 2] > out[6, 1, 0] + 20.0
    np.testing.assert_array_equal(out[4, 3], module.BUILTUP_COLOR.astype(np.float32))


def test_final_canvas_terrain_tone_can_limit_cleanup_to_subject_mask():
    module = _load_example_module()
    image = module.Image.new("RGBA", (8, 8), (255, 255, 255, 255))
    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[4, 2, :3] = (150, 150, 150)
    arr[4, 5, :3] = (150, 150, 150)
    eligible = np.zeros((8, 8), dtype=bool)
    eligible[4, 2] = True

    toned = module._apply_final_canvas_terrain_tone(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)

    assert out[4, 2, 2] > out[4, 2, 0] + 20.0
    np.testing.assert_array_equal(out[4, 5], np.array([150, 150, 150], dtype=np.float32))


def test_final_canvas_terrain_tone_closes_neutral_gaps_inside_terrain_mask():
    module = _load_example_module()
    image = module.Image.new("RGBA", (11, 11), (255, 255, 255, 255))
    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[4:8, 4:8, :3] = module.TERRAIN_PALETTE[1]
    arr[6, 6, :3] = (212, 213, 214)
    eligible = np.zeros((11, 11), dtype=bool)
    eligible[4:8, 4:8] = True
    eligible[6, 6] = False

    toned = module._apply_final_canvas_terrain_tone(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)

    assert out[6, 6, 2] > out[6, 6, 0] + 20.0


def test_final_canvas_terrain_tone_closes_wider_pale_terrain_gaps():
    module = _load_example_module()
    arr = np.full((80, 80, 4), 255, dtype=np.uint8)
    arr[25:56, 18:69, :3] = module.TERRAIN_PALETTE[1]
    arr[40, 44, :3] = (212, 213, 214)
    eligible = np.zeros((80, 80), dtype=bool)
    eligible[25:56, 18:69] = True
    eligible[40, 44] = False

    toned = module._apply_final_canvas_terrain_tone(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)

    assert out[40, 44, 2] > out[40, 44, 0] + 20.0


def test_final_canvas_terrain_tone_closes_subject_mask_holes_near_terrain():
    module = _load_example_module()
    arr = np.full((80, 80, 4), 255, dtype=np.uint8)
    arr[35:45, 18:28, :3] = module.TERRAIN_PALETTE[1]
    arr[40, 49, :3] = (212, 213, 214)
    eligible = np.zeros((80, 80), dtype=bool)
    eligible[35:45, 18:60] = True
    eligible[40, 49] = False

    toned = module._apply_final_canvas_terrain_tone(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)

    assert out[40, 49, 2] > out[40, 49, 0] + 20.0


def test_final_canvas_terrain_tone_does_not_expand_cleanup_outside_subject_edge():
    module = _load_example_module()
    arr = np.full((80, 80, 4), 255, dtype=np.uint8)
    arr[35:45, 18:28, :3] = module.TERRAIN_PALETTE[1]
    arr[40, 49, :3] = (212, 213, 214)
    eligible = np.zeros((80, 80), dtype=bool)
    eligible[35:45, 18:28] = True

    toned = module._apply_final_canvas_terrain_tone(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)

    np.testing.assert_array_equal(out[40, 49], np.array([212, 213, 214], dtype=np.float32))


def test_final_canvas_terrain_tone_compresses_dense_cyan_highlights():
    module = _load_example_module()
    arr = np.full((20, 20, 4), 255, dtype=np.uint8)
    arr[4:18, 2:18, :3] = (35, 82, 114)
    arr[8:16, 4:16, :3] = (120, 210, 252)

    toned = module._apply_final_canvas_terrain_tone(module.Image.fromarray(arr, mode="RGBA"))
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)
    terrain = out[4:18, 2:18]
    luminance = (
        terrain[:, :, 0] * 0.2126
        + terrain[:, :, 1] * 0.7152
        + terrain[:, :, 2] * 0.0722
    )

    assert np.percentile(luminance, 90) < 170.0


def test_final_canvas_terrain_tone_caps_saturated_cyan_highlight_tail():
    module = _load_example_module()
    arr = np.full((40, 40, 4), 255, dtype=np.uint8)
    low = np.array([31, 71, 98], dtype=np.float32)
    high = np.array([120, 210, 252], dtype=np.float32)
    for x in range(2, 38):
        fraction = (x - 2) / 35.0
        arr[8:34, x, :3] = np.round(low * (1.0 - fraction) + high * fraction).astype(np.uint8)

    toned = module._apply_final_canvas_terrain_tone(module.Image.fromarray(arr, mode="RGBA"))
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)
    terrain = out[8:34, 2:38]
    luminance = (
        terrain[:, :, 0] * 0.2126
        + terrain[:, :, 1] * 0.7152
        + terrain[:, :, 2] * 0.0722
    )

    assert np.percentile(luminance, 99) < 205.0


def test_terrain_speckle_smoothing_protects_text_and_builtup():
    module = _load_example_module()
    image = module.Image.new("RGBA", (9, 9), (255, 255, 255, 255))
    arr = np.asarray(image, dtype=np.uint8).copy()
    arr[4, 4, :3] = (210, 211, 212)
    arr[3:6, 3, :3] = module.TERRAIN_PALETTE[1]
    arr[3:6, 5, :3] = module.TERRAIN_PALETTE[1]
    arr[3, 3:6, :3] = module.TERRAIN_PALETTE[1]
    arr[5, 3:6, :3] = module.TERRAIN_PALETTE[1]
    arr[1, 4, :3] = (30, 73, 96)
    arr[4, 6, :3] = module.BUILTUP_COLOR
    eligible = np.zeros((9, 9), dtype=bool)
    eligible[3:6, 3:7] = True

    smoothed = module._smooth_terrain_speckles(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(smoothed.convert("RGB"), dtype=np.float32)

    assert out[4, 4, 2] > out[4, 4, 0] + 20.0
    np.testing.assert_array_equal(out[1, 4], np.array([30, 73, 96], dtype=np.float32))
    np.testing.assert_array_equal(out[4, 6], module.BUILTUP_COLOR.astype(np.float32))


def test_terrain_speckle_smoothing_tints_larger_neutral_patches():
    module = _load_example_module()
    arr = np.full((9, 9, 4), 255, dtype=np.uint8)
    arr[2:7, 2:7, :3] = (205, 206, 208)
    arr[:, :, 3] = 255
    eligible = np.zeros((9, 9), dtype=bool)
    eligible[2:7, 2:7] = True

    smoothed = module._smooth_terrain_speckles(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(smoothed.convert("RGB"), dtype=np.float32)

    assert out[4, 4, 2] > out[4, 4, 0] + 20.0


def test_terrain_speckle_smoothing_suppresses_clustered_bright_terrain_highlights():
    module = _load_example_module()
    arr = np.full((31, 31, 4), 255, dtype=np.uint8)
    arr[:, :, 3] = 255
    arr[5:26, 5:26, :3] = module.TERRAIN_PALETTE[1]
    arr[12:19, 12:19, :3] = (180, 225, 250)
    arr[15, 24, :3] = module.BUILTUP_COLOR
    eligible = np.zeros((31, 31), dtype=bool)
    eligible[5:26, 5:26] = True

    smoothed = module._smooth_terrain_speckles(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(smoothed.convert("RGB"), dtype=np.float32)
    luma = (
        out[:, :, 0] * 0.2126
        + out[:, :, 1] * 0.7152
        + out[:, :, 2] * 0.0722
    )

    assert luma[15, 15] < 170.0
    np.testing.assert_array_equal(out[15, 24], module.BUILTUP_COLOR.astype(np.float32))


def test_terrain_speckle_smoothing_reduces_broad_pbr_highlight_patches():
    module = _load_example_module()
    arr = np.full((61, 61, 4), 255, dtype=np.uint8)
    arr[:, :, 3] = 255
    arr[6:55, 6:55, :3] = module.TERRAIN_PALETTE[1]
    arr[19:42, 19:42, :3] = (180, 225, 250)
    eligible = np.zeros((61, 61), dtype=bool)
    eligible[6:55, 6:55] = True

    smoothed = module._smooth_terrain_speckles(
        module.Image.fromarray(arr, mode="RGBA"),
        eligible_mask=eligible,
    )
    out = np.asarray(smoothed.convert("RGB"), dtype=np.float32)
    luma = (
        out[:, :, 0] * 0.2126
        + out[:, :, 1] * 0.7152
        + out[:, :, 2] * 0.0722
    )

    assert luma[30, 30] < 155.0


def test_terrain_speckle_smoothing_compresses_oversaturated_blue_terrain():
    module = _load_example_module()
    arr = np.full((20, 20, 4), 255, dtype=np.uint8)
    arr[4:18, 2:18, :3] = (20, 110, 190)
    arr[4:18, 2:18, 3] = 255
    arr[4, 4, :3] = module.BUILTUP_COLOR
    eligible = np.zeros((20, 20), dtype=bool)
    eligible[4:18, 2:18] = True

    toned = module._apply_final_canvas_terrain_tone(module.Image.fromarray(arr, mode="RGBA"))
    smoothed = module._smooth_terrain_speckles(toned, eligible_mask=eligible)
    out = np.asarray(smoothed.convert("RGB"), dtype=np.uint8)
    _, saturation, _ = module._rgb_to_hsv_channels(out)
    terrain = eligible.copy()
    terrain[4, 4] = False

    assert np.percentile(saturation[terrain], 95) <= 0.66
    np.testing.assert_array_equal(out[4, 4], module.BUILTUP_COLOR.astype(np.uint8))


def test_subject_terrain_tone_compresses_map_highlights_before_shadow():
    module = _load_example_module()
    arr = np.full((4, 4, 4), 255, dtype=np.uint8)
    arr[:, :, :3] = (220, 225, 225)
    arr[:, :, 3] = 255
    arr[0, 0, :3] = module.BUILTUP_COLOR

    toned = module._apply_subject_terrain_tone(module.Image.fromarray(arr, mode="RGBA"))
    out = np.asarray(toned.convert("RGB"), dtype=np.float32)
    terrain_luma = out[2, 2] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    assert terrain_luma < 170.0
    np.testing.assert_array_equal(out[0, 0], module.BUILTUP_COLOR.astype(np.float32))


def test_subject_neutral_shadow_tint_targets_opaque_terrain_only():
    module = _load_example_module()
    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[:, :, :3] = (92, 94, 96)
    rgba[:, :, 3] = 255
    rgba[0, 1, 3] = 0
    rgba[1, 2, :3] = module.BUILTUP_COLOR

    tinted = module._tint_subject_neutral_shadows(module.Image.fromarray(rgba, mode="RGBA"))
    out = np.asarray(tinted.convert("RGBA"), dtype=np.float32)

    assert out[0, 0, 2] > out[0, 0, 0] + 20.0
    assert out[0, 1, 3] == 0.0
    np.testing.assert_array_equal(out[1, 2, :3], module.BUILTUP_COLOR.astype(np.float32))


def test_combine_render_passes_uses_relief_luminance_for_terrain_shadow(tmp_path):
    module = _load_example_module()
    color = np.full((4, 8, 4), 255, dtype=np.uint8)
    relief = np.full((4, 8, 4), 255, dtype=np.uint8)
    color[:, 1:7, :3] = module.TERRAIN_PALETTE[1].astype(np.uint8)
    color[:, 3:5, :3] = module.BUILTUP_COLOR
    relief[:, 1:4, :3] = (35, 35, 35)
    relief[:, 4:7, :3] = (230, 230, 230)
    color_path = tmp_path / "color.png"
    relief_path = tmp_path / "relief.png"
    module.Image.fromarray(color, mode="RGBA").save(color_path)
    module.Image.fromarray(relief, mode="RGBA").save(relief_path)

    combined = module._combine_render_passes(color_path, relief_path)
    arr = np.asarray(combined.convert("RGBA"))
    luma = (
        arr[:, :, 0].astype(np.float32) * 0.2126
        + arr[:, :, 1].astype(np.float32) * 0.7152
        + arr[:, :, 2].astype(np.float32) * 0.0722
    )

    assert luma[2, 5] > luma[2, 2] + 20.0
    assert luma[2, 4] > luma[2, 3] + 20.0
    assert arr[2, 2, 2] > arr[2, 2, 0]


def test_combine_render_passes_damps_high_frequency_relief_speckle(tmp_path):
    module = _load_example_module()
    color = np.full((16, 16, 4), 255, dtype=np.uint8)
    relief = np.full((16, 16, 4), 255, dtype=np.uint8)
    color[3:13, 3:13, :3] = module.TERRAIN_PALETTE[1].astype(np.uint8)
    color[3:13, 3:13, 3] = 255
    checker = (np.indices((10, 10)).sum(axis=0) % 2).astype(bool)
    relief[3:13, 3:13, :3] = np.where(checker[:, :, None], 245, 30).astype(np.uint8)
    relief[3:13, 3:13, 3] = 255
    color_path = tmp_path / "color.png"
    relief_path = tmp_path / "relief.png"
    module.Image.fromarray(color, mode="RGBA").save(color_path)
    module.Image.fromarray(relief, mode="RGBA").save(relief_path)

    combined = module._combine_render_passes(color_path, relief_path)
    arr = np.asarray(combined.convert("RGBA"))
    luma = (
        arr[3:13, 3:13, 0].astype(np.float32) * 0.2126
        + arr[3:13, 3:13, 1].astype(np.float32) * 0.7152
        + arr[3:13, 3:13, 2].astype(np.float32) * 0.0722
    )

    assert float(luma.std()) < 30.0


def test_overlay_style_sidecar_invalidates_stale_cached_overlay(tmp_path):
    module = _load_example_module()
    overlay_path = tmp_path / "overlay.png"
    overlay_path.write_bytes(b"not a real png")

    assert not module._overlay_is_current(overlay_path)

    module._overlay_style_path(overlay_path).write_text("old-style\n", encoding="utf-8")
    assert not module._overlay_is_current(overlay_path)

    module._overlay_style_path(overlay_path).write_text(
        module.OVERLAY_STYLE_VERSION + "\n",
        encoding="utf-8",
    )
    assert module._overlay_is_current(overlay_path)


def test_render_downloads_reference_hdri_and_sends_it_to_pbr(tmp_path, monkeypatch):
    module = _load_example_module()
    dem_path = tmp_path / "dem.tif"
    overlay_path = tmp_path / "overlay.png"
    snapshot_path = tmp_path / "snapshot.png"
    hdr_path = tmp_path / module.HDR.name
    overlay_path.write_bytes(b"not inspected by render test")

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_bounds(0.0, 0.0, 4.0, 3.0, 4, 3),
    ) as dst:
        dst.write(np.ones((3, 4), dtype=np.float32), 1)

    class FakeViewer:
        def __init__(self):
            self.commands = []
            self.overlay_calls = []
            self.snapshots = []

        def send_ipc(self, command):
            self.commands.append(command)

        def load_overlay(self, name, path, **kwargs):
            self.overlay_calls.append((name, path, kwargs))

        def snapshot(self, path, **kwargs):
            self.snapshots.append((path, kwargs))

    class FakeViewerContext:
        def __init__(self, viewer):
            self.viewer = viewer

        def __enter__(self):
            return self.viewer

        def __exit__(self, exc_type, exc, tb):
            return False

    calls = []

    def fake_ensure_hdri(*, force=False):
        calls.append(force)
        hdr_path.write_bytes(b"hdr")
        return hdr_path

    viewer = FakeViewer()
    monkeypatch.setattr(module, "_ensure_hdri", fake_ensure_hdri)
    monkeypatch.setattr(module.f3d, "open_viewer_async", lambda **kwargs: FakeViewerContext(viewer))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "_combine_render_passes", lambda color_path, relief_path: "combined-image")
    monkeypatch.setattr(module, "_compose_snapshot", lambda raw_path, output_path: None)

    module._render(snapshot_path, dem_path, overlay_path)

    pbr_commands = [command for command in viewer.commands if command.get("cmd") == "set_terrain_pbr"]
    assert calls == [False]
    assert len(pbr_commands) == 2
    assert {command["hdr_path"] for command in pbr_commands} == {str(hdr_path.resolve())}
    assert {command["hdr_rotate_deg"] for command in pbr_commands} == {225.0}
    assert all(command["ibl_intensity"] == pytest.approx(1.3) for command in pbr_commands)


def test_render_sends_reference_shadow_controls(tmp_path, monkeypatch):
    module = _load_example_module()
    dem_path = tmp_path / "dem.tif"
    overlay_path = tmp_path / "overlay.png"
    snapshot_path = tmp_path / "snapshot.png"
    hdr_path = tmp_path / module.HDR.name
    overlay_path.write_bytes(b"not inspected by render test")

    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_bounds

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_bounds(0.0, 0.0, 4.0, 3.0, 4, 3),
    ) as dst:
        dst.write(np.ones((3, 4), dtype=np.float32), 1)

    class FakeViewer:
        def __init__(self):
            self.commands = []
            self.overlay_calls = []
            self.snapshots = []

        def send_ipc(self, command):
            self.commands.append(command)

        def load_overlay(self, name, path, **kwargs):
            self.overlay_calls.append((name, path, kwargs))

        def snapshot(self, path, **kwargs):
            self.snapshots.append((path, kwargs))

    class FakeViewerContext:
        def __init__(self, viewer):
            self.viewer = viewer

        def __enter__(self):
            return self.viewer

        def __exit__(self, exc_type, exc, tb):
            return False

    viewer = FakeViewer()
    hdr_path.write_bytes(b"hdr")
    monkeypatch.setattr(module, "_ensure_hdri", lambda *, force=False: hdr_path)
    monkeypatch.setattr(module.f3d, "open_viewer_async", lambda **kwargs: FakeViewerContext(viewer))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module, "_combine_render_passes", lambda color_path, relief_path: "combined-image")
    monkeypatch.setattr(module, "_compose_snapshot", lambda raw_path, output_path: None)

    module._render(snapshot_path, dem_path, overlay_path)

    terrain_cmd = next(command for command in viewer.commands if command.get("cmd") == "set_terrain")
    pbr_cmd = next(command for command in viewer.commands if command.get("cmd") == "set_terrain_pbr")
    relief_cmd = [
        command for command in viewer.commands
        if command.get("cmd") == "set_terrain" and command.get("shadow") == pytest.approx(0.78)
    ][0]
    relief_pbr_cmd = [
        command for command in viewer.commands
        if command.get("cmd") == "set_terrain_pbr" and command.get("normal_strength") == pytest.approx(1.85)
    ][0]
    overlay_disable = [
        command for command in viewer.commands
        if command == {"cmd": "set_overlays_enabled", "enabled": False}
    ]

    assert terrain_cmd["sun_elevation"] == pytest.approx(24.0)
    assert terrain_cmd["sun_intensity"] == pytest.approx(1.95)
    assert terrain_cmd["ambient"] == pytest.approx(0.58)
    assert terrain_cmd["shadow"] == pytest.approx(0.42)
    assert pbr_cmd["normal_strength"] == pytest.approx(1.10)
    assert relief_cmd["sun_elevation"] == pytest.approx(18.0)
    assert relief_cmd["ambient"] == pytest.approx(0.36)
    assert relief_pbr_cmd["height_ao"]["enabled"] is True
    assert relief_pbr_cmd["height_ao"]["strength"] == pytest.approx(0.42)
    assert relief_pbr_cmd["sun_visibility"]["enabled"] is True
    assert relief_pbr_cmd["sun_visibility"]["mode"] == "hard"
    assert overlay_disable
    assert len(viewer.snapshots) == 2
