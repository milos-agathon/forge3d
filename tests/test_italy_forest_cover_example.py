from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")
shapely_geometry = pytest.importorskip("shapely.geometry")
MultiPolygon = shapely_geometry.MultiPolygon
box = shapely_geometry.box
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "examples" / "forest_cover_copernicus" / "italy_forest_cover_3d.py"


def _load_example_module():
    examples_dir = str(MODULE_PATH.parent)
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    spec = importlib.util.spec_from_file_location("italy_forest_cover_3d", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legend_bar_uses_same_cover_palette_as_overlay():
    module = _load_example_module()
    size = (1000, 1000)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    module._draw_vertical_moss_legend(draw, size)

    x, y, width, height = module._legend_box(size)
    for cover in (0.0, 25.0, 50.0, 75.0):
        offset = min(height - 1, round((1.0 - cover / 100.0) * height))
        sample_y = y + offset
        represented_cover = np.asarray([[100.0 * (1.0 - offset / max(height - 1, 1))]], dtype=np.float32)
        expected = module._forest_rgba(represented_cover, np.ones_like(represented_cover, dtype=bool))[0, 0, :3]
        expected = np.asarray(module._apply_final_rgb_lighting(expected), dtype=np.int16)
        actual = np.asarray(canvas.getpixel((x + width // 2, sample_y))[:3], dtype=np.int16)
        assert np.max(np.abs(actual - expected)) <= 2


def test_layout_is_portrait_with_tighter_map_and_title_placement():
    module = _load_example_module()

    assert module.COMPOSE_CANVAS_SIZE[1] > module.COMPOSE_CANVAS_SIZE[0]
    assert module.COMPOSE_CANVAS_SIZE[1] < 7900
    assert module.SNAPSHOT_SIZE == module.COMPOSE_CANVAS_SIZE
    assert module.MAP_TARGET_LAND_BBOX[0] < 0.10
    assert module.MAP_TARGET_LAND_BBOX[1] > module.CAPTION_Y + 0.02
    assert module.MAP_TARGET_LAND_BBOX[2] >= 0.80
    assert module.MAP_TARGET_LAND_BBOX[3] >= 0.84
    assert module.TITLE_CENTER_X > 0.72
    assert module.TITLE_COUNTRY_BBOX[2] - module.TITLE_COUNTRY_BBOX[0] > 0.30
    assert module.TITLE_FOREST_MAP_BBOX[3] - module.TITLE_FOREST_MAP_BBOX[1] <= 0.020
    assert module.TITLE_COUNTRY_BBOX[3] - module.TITLE_COUNTRY_BBOX[1] <= 0.062
    assert module.CAPTION_Y - module.SUBTITLE_Y <= 0.024
    assert module.CAPTION_FONT_SCALE > module.SUBTITLE_FONT_SCALE


def test_russia_layout_is_landscape_with_top_text_and_wide_map():
    module = _load_example_module()

    module._configure_region(module.REGION_PRESETS["russia"])

    assert module.COMPOSE_CANVAS_SIZE == (5700, 4300)
    assert module.RENDER_SNAPSHOT_SIZE == (5700, 4300)
    assert module.COMPOSE_CANVAS_SIZE[0] > module.COMPOSE_CANVAS_SIZE[1]
    assert module.RENDER_SNAPSHOT_SIZE[0] > module.RENDER_SNAPSHOT_SIZE[1]
    assert module.SNAPSHOT_SIZE == module.COMPOSE_CANVAS_SIZE
    assert module.MAP_TARGET_LAND_BBOX[0] <= 0.030
    assert module.MAP_TARGET_LAND_BBOX[1] > module.CAPTION_Y + 0.040
    assert module.MAP_TARGET_LAND_BBOX[2] >= 0.970
    assert module.MAP_TARGET_LAND_BBOX[3] >= 0.940
    assert module.TITLE_CENTER_X == pytest.approx(0.500)
    assert module.TITLE_FOREST_MAP_BBOX[1] < module.TITLE_COUNTRY_BBOX[1] < module.SUBTITLE_Y
    assert module.CAPTION_Y < module.MAP_TARGET_LAND_BBOX[1]


def test_russia_legend_is_horizontal_between_source_and_map():
    module = _load_example_module()

    module._configure_region(module.REGION_PRESETS["russia"])
    x, y, width, height = module._legend_box(module.COMPOSE_CANVAS_SIZE)

    assert module.LEGEND_ORIENTATION == "horizontal"
    assert width > height * 18
    assert module.LEGEND_TITLE_FONT_SCALE >= 0.0104
    assert module.LEGEND_LABEL_FONT_SCALE >= 0.0094
    assert y / module.COMPOSE_CANVAS_SIZE[1] > module.CAPTION_Y + 0.050
    assert (y + height) / module.COMPOSE_CANVAS_SIZE[1] < module.MAP_TARGET_LAND_BBOX[1]
    assert module.MAP_TARGET_LAND_BBOX[1] - ((y + height) / module.COMPOSE_CANVAS_SIZE[1]) > 0.060


def test_russia_title_is_reduced_to_leave_room_for_legend_stack():
    module = _load_example_module()

    module._configure_region(module.REGION_PRESETS["russia"])

    assert module.TITLE_FOREST_MAP_BBOX[2] - module.TITLE_FOREST_MAP_BBOX[0] <= 0.120
    assert module.TITLE_FOREST_MAP_BBOX[3] - module.TITLE_FOREST_MAP_BBOX[1] <= 0.017
    assert module.TITLE_COUNTRY_BBOX[2] - module.TITLE_COUNTRY_BBOX[0] <= 0.330
    assert module.TITLE_COUNTRY_BBOX[3] - module.TITLE_COUNTRY_BBOX[1] <= 0.044


def _text_bbox(module, size, text, position, font_scale):
    probe = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(probe)
    return draw.textbbox(
        position,
        text,
        font=module._load_font(round(size[0] * font_scale), bold=True),
        anchor="ma",
    )


def test_russia_top_stack_text_does_not_overlap():
    module = _load_example_module()

    module._configure_region(module.REGION_PRESETS["russia"])
    size = module.COMPOSE_CANVAS_SIZE
    width, height = size
    title_center_x = round(width * module.TITLE_CENTER_X)
    x, y, legend_w, legend_h = module._legend_box(size)

    source_bbox = _text_bbox(
        module,
        size,
        "COPERNICUS LAND COVER 2019",
        (title_center_x, round(height * module.CAPTION_Y)),
        module.CAPTION_FONT_SCALE,
    )
    legend_title_bbox = _text_bbox(
        module,
        size,
        module.LEGEND_TITLE,
        (x + legend_w // 2, y - round(height * module.LEGEND_TITLE_GAP)),
        module.LEGEND_TITLE_FONT_SCALE,
    )
    label_y = y + legend_h + round(height * module.LEGEND_LABEL_GAP)
    label_boxes = [
        _text_bbox(
            module,
            size,
            label,
            (x + round(fraction * (legend_w - 1)), label_y),
            module.LEGEND_LABEL_FONT_SCALE,
        )
        for label, fraction in reversed(module.LEGEND_TICKS)
    ]
    map_top = round(height * module.MAP_TARGET_LAND_BBOX[1])

    assert source_bbox[3] + 16 <= legend_title_bbox[1]
    assert legend_title_bbox[3] + 8 <= y
    assert y + legend_h + 8 <= min(box[1] for box in label_boxes)
    assert max(box[3] for box in label_boxes) + 24 <= map_top


def test_horizontal_legend_uses_palette_left_to_right():
    module = _load_example_module()
    module._configure_region(module.REGION_PRESETS["russia"])
    canvas = Image.new("RGBA", module.COMPOSE_CANVAS_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    module._draw_horizontal_moss_legend(draw, module.COMPOSE_CANVAS_SIZE)

    x, y, width, height = module._legend_box(module.COMPOSE_CANVAS_SIZE)
    for cover in (0.0, 50.0, 100.0):
        offset = min(width - 1, round((cover / 100.0) * (width - 1)))
        sample_x = x + offset
        expected = module._forest_rgb_for_cover(np.asarray(cover, dtype=np.float32)).astype(np.uint8)
        expected = np.asarray(module._apply_final_rgb_lighting(expected), dtype=np.int16)
        actual = np.asarray(canvas.getpixel((sample_x, y + height // 2))[:3], dtype=np.int16)
        assert np.max(np.abs(actual - expected)) <= 2


def test_russia_layout_uses_horizontal_camera_orientation():
    module = _load_example_module()

    module._configure_region(module.REGION_PRESETS["russia"])

    assert module.TERRAIN["phi"] == pytest.approx(90.0)


def test_legend_is_moved_up_in_portrait_layout():
    module = _load_example_module()

    x, y, _, height = module._legend_box(module.COMPOSE_CANVAS_SIZE)
    assert x / module.COMPOSE_CANVAS_SIZE[0] >= 0.86
    assert y / module.COMPOSE_CANVAS_SIZE[1] <= 0.64
    assert (y + height) / module.COMPOSE_CANVAS_SIZE[1] <= 0.80


def test_legend_includes_title_and_maximum_label():
    module = _load_example_module()

    assert module.LEGEND_TITLE == "% of cover"
    labels = [label for label, _ in module.LEGEND_TICKS]
    assert labels == ["100", "75", "50", "25", "0"]


def test_forest_palette_uses_green_fes_slice():
    module = _load_example_module()

    assert module.FOREST_PALETTE_HEX[0] == "#E8E6ED"
    assert module.FOREST_PALETTE_HEX[-1] == "#024026"
    assert module._forest_rgb_for_cover(np.asarray(100.0, dtype=np.float32)).astype(np.uint8).tolist() == [2, 64, 38]


def test_exaggeration_and_lightness_are_increased_but_bounded():
    module = _load_example_module()

    assert 86.0 <= module.SURFACE_FOREST_HEIGHT <= 96.0
    assert 1.62 <= module.CAMERA["zscale"] <= 1.80
    assert 1.55 <= module.FINAL_MAP_BRIGHTNESS <= 1.70
    assert module.FINAL_MAP_WHITE_MIX <= 0.07


def test_final_map_brightness_lifts_subject_colors_without_washing_them_out():
    module = _load_example_module()
    image = Image.new("RGBA", (3, 1), (*module.MAP_BACKGROUND_RGB, 255))
    image.putpixel((1, 0), (90, 120, 60, 255))

    toned = module._apply_final_map_brightness(image)

    assert toned.getpixel((1, 0))[:3] == module._apply_final_rgb_lighting((90, 120, 60))
    assert 130 <= toned.getpixel((1, 0))[1] <= 190
    assert max(toned.getpixel((1, 0))[:3]) <= 205
    assert max(toned.getpixel((1, 0))[:3]) - min(toned.getpixel((1, 0))[:3]) >= 55


def test_requested_forest_batch_presets_cover_target_regions():
    module = _load_example_module()

    assert module.REQUESTED_BATCH == (
        "france",
        "germany",
        "poland",
        "turkey",
        "southeast_europe",
        "mainland_usa",
        "iberia",
        "russia",
        "africa",
        "europe",
    )
    presets = module.REGION_PRESETS
    assert presets["france"].bbox == pytest.approx((-6.0, 41.0, 10.5, 52.0))
    assert presets["mainland_usa"].bbox == pytest.approx((-125.0, 24.0, -66.0, 50.0))
    assert presets["mainland_usa"].target_crs == "EPSG:5070"
    assert presets["africa"].continent == "Africa"
    assert presets["russia"].target_crs == module.RUSSIA_TARGET_CRS
    assert presets["russia"].boundary_source == "gisco"
    assert presets["europe"].continent == "Europe"


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _text_and_legend_reserved_boxes(module, size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    width, height = size
    legend_x, legend_y, legend_w, legend_h = module._legend_box(size)
    return [
        module._fraction_bbox_to_pixels(module.TITLE_FOREST_MAP_BBOX, size),
        module._fraction_bbox_to_pixels(module.TITLE_COUNTRY_BBOX, size),
        (
            round(width * (module.TITLE_CENTER_X - 0.17)),
            round(height * (module.SUBTITLE_Y - 0.018)),
            round(width * (module.TITLE_CENTER_X + 0.17)),
            round(height * (module.CAPTION_Y + 0.018)),
        ),
        (
            legend_x - round(width * 0.018),
            legend_y - round(height * 0.050),
            legend_x + legend_w + round(width * 0.075),
            legend_y + legend_h + round(height * 0.025),
        ),
    ]


def test_reference_layout_keeps_fitted_map_out_of_title_caption_and_legend():
    module = _load_example_module()
    size = module.COMPOSE_CANVAS_SIZE
    subject = Image.new("RGBA", (500, 300), (40, 90, 45, 255))

    fitted, origin = module._fit_subject_to_reference_bbox(subject, size)
    local_bbox = module._alpha_bbox(fitted)
    assert local_bbox is not None
    fitted_bbox = (
        local_bbox[0] + origin[0],
        local_bbox[1] + origin[1],
        local_bbox[2] + origin[0],
        local_bbox[3] + origin[1],
    )

    for reserved in _text_and_legend_reserved_boxes(module, size):
        assert not _boxes_intersect(fitted_bbox, reserved)


def test_reference_layout_preserves_subject_aspect_ratio_and_centers_fit():
    module = _load_example_module()
    size = module.COMPOSE_CANVAS_SIZE
    subject = Image.new("RGBA", (1000, 420), (40, 90, 45, 255))

    fitted, origin = module._fit_subject_to_reference_bbox(subject, size)
    local_bbox = module._alpha_bbox(fitted)
    assert local_bbox is not None
    fitted_bbox = (
        local_bbox[0] + origin[0],
        local_bbox[1] + origin[1],
        local_bbox[2] + origin[0],
        local_bbox[3] + origin[1],
    )
    target_bbox = module._fraction_bbox_to_pixels(module.MAP_TARGET_LAND_BBOX, size)
    source_aspect = subject.width / subject.height
    fitted_aspect = (fitted_bbox[2] - fitted_bbox[0]) / (fitted_bbox[3] - fitted_bbox[1])
    target_center = ((target_bbox[0] + target_bbox[2]) / 2.0, (target_bbox[1] + target_bbox[3]) / 2.0)
    fitted_center = ((fitted_bbox[0] + fitted_bbox[2]) / 2.0, (fitted_bbox[1] + fitted_bbox[3]) / 2.0)

    assert fitted_aspect == pytest.approx(source_aspect, rel=0.02)
    assert abs(fitted_center[0] - target_center[0]) <= 2.0
    assert abs(fitted_center[1] - target_center[1]) <= 2.0


def test_only_european_regions_use_epsg_3035():
    module = _load_example_module()
    europe_regions = {"italy", "france", "germany", "poland", "southeast_europe", "iberia", "europe"}
    own_crs_regions = {"turkey", "mainland_usa", "russia", "africa"}

    assert {key for key, preset in module.REGION_PRESETS.items() if preset.target_crs == "EPSG:3035"} == europe_regions
    for key in own_crs_regions:
        assert module.REGION_PRESETS[key].target_crs != "EPSG:3035"


def test_russia_geometry_uses_dateline_safe_projection(tmp_path):
    module = _load_example_module()
    module._configure_region(module.REGION_PRESETS["russia"])
    boundary_path = tmp_path / "countries.geojson"
    russia = MultiPolygon(
        [
            box(30.0, 55.0, 40.0, 60.0),
            box(170.0, 60.0, 180.0, 65.0),
            box(-180.0, 60.0, -170.0, 65.0),
        ]
    )
    countries = gpd.GeoDataFrame(
        {
            "ADM0_A3": ["RUS", "USA"],
            "ADMIN": ["Russia", "United States of America"],
            "geometry": [russia, box(-125.0, 25.0, -66.0, 50.0)],
        },
        crs="EPSG:4326",
    )
    countries.to_file(boundary_path, driver="GeoJSON")

    geometry = module._country_geometry(boundary_path, module.TARGET_CRS)
    minx, _, maxx, _ = geometry.bounds

    assert (maxx - minx) < 15_000_000.0


def test_russia_boundary_download_uses_gisco_country_regions(tmp_path, monkeypatch):
    module = _load_example_module()
    module._configure_region(module.REGION_PRESETS["russia"])
    calls = []

    def fake_download(url, dest, *, force):
        calls.append((url, dest, force))
        return dest

    monkeypatch.setattr(module, "_download", fake_download)

    boundary_path = module._ensure_boundary_file(tmp_path, force=False)

    assert boundary_path.name == "CNTR_RG_01M_2024_4326.geojson"
    assert calls == [(module.GISCO_COUNTRIES_2024_URL, boundary_path, False)]


def test_russia_geometry_can_select_gisco_iso3_schema(tmp_path):
    module = _load_example_module()
    module._configure_region(module.REGION_PRESETS["russia"])
    boundary_path = tmp_path / "gisco_countries.geojson"
    russia = MultiPolygon(
        [
            box(30.0, 55.0, 40.0, 60.0),
            box(170.0, 60.0, 180.0, 65.0),
            box(-180.0, 60.0, -170.0, 65.0),
        ]
    )
    countries = gpd.GeoDataFrame(
        {
            "ISO3_CODE": ["RUS", "USA"],
            "NAME_ENGL": ["Russian Federation", "United States"],
            "geometry": [russia, box(-125.0, 25.0, -66.0, 50.0)],
        },
        crs="EPSG:4326",
    )
    countries.to_file(boundary_path, driver="GeoJSON")

    geometry = module._country_geometry(boundary_path, module.TARGET_CRS)
    minx, _, maxx, _ = geometry.bounds

    assert (maxx - minx) < 15_000_000.0


def test_tree_cover_tile_selection_intersects_region_geometry(tmp_path, monkeypatch):
    module = _load_example_module()
    module._configure_region(module.REGION_PRESETS["russia"])
    for name in (
        "E020N60_PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
        "W100N40_PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
        "W180N60_PROBAV_LC100_global_v3.0.1_2019-nrt_Tree-CoverFraction-layer_EPSG-4326.tif",
    ):
        (tmp_path / name).write_bytes(b"")
    russia = MultiPolygon(
        [
            box(30.0, 55.0, 40.0, 56.0),
            box(-179.0, 55.0, -170.0, 56.0),
        ]
    )
    monkeypatch.setattr(module, "FOREST_DATA_DIR", tmp_path)

    selected = [path.name[:7] for path in module._tree_cover_tifs(russia)]

    assert selected == ["E020N60", "W180N60"]


def test_final_map_brightness_preserves_highlight_detail():
    module = _load_example_module()
    image = Image.new("RGBA", (3, 1), (*module.MAP_BACKGROUND_RGB, 255))
    image.putpixel((1, 0), (230, 225, 210, 255))

    toned = module._apply_final_map_brightness(image)

    assert toned.getpixel((1, 0))[:3] == module._apply_final_rgb_lighting((230, 225, 210))
    assert max(toned.getpixel((1, 0))[:3]) <= 250
    assert toned.getpixel((1, 0))[0] > 230


def test_final_map_brightness_keeps_dark_forest_tones_visible():
    module = _load_example_module()
    toned = module._apply_final_rgb_lighting((40, 70, 35))

    assert 55 <= min(toned) <= 90
    assert max(toned) <= 115
    assert max(toned) - min(toned) >= 40


def test_palette_chroma_restore_keeps_relief_luminance():
    module = _load_example_module()
    source = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    source.putpixel((0, 0), (27, 89, 98, 255))
    source.putpixel((1, 0), (1, 25, 89, 255))
    relief_toned = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    relief_toned.putpixel((0, 0), (126, 120, 101, 255))
    relief_toned.putpixel((1, 0), (78, 83, 89, 255))
    mask = np.ones((1, 2), dtype=bool)

    restored = module._restore_palette_chroma_from_source(source, relief_toned, mask)
    restored_rgb = np.asarray(restored, dtype=np.float32)[0, :, :3]
    relief_rgb = np.asarray(relief_toned, dtype=np.float32)[0, :, :3]

    restored_hue, restored_saturation, _ = module._rgb_to_hsv_channels(restored_rgb.reshape(1, 2, 3).astype(np.uint8))
    source_hue, source_saturation, _ = module._rgb_to_hsv_channels(
        np.asarray(source, dtype=np.uint8)[:, :, :3]
    )
    _, relief_saturation, _ = module._rgb_to_hsv_channels(relief_rgb.reshape(1, 2, 3).astype(np.uint8))
    restored_luma = restored_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    relief_luma = relief_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    assert np.all(np.abs(restored_luma - relief_luma) <= 9.0)
    assert np.all(restored_saturation[0] > relief_saturation[0] + 0.18)
    assert np.all(np.abs(restored_hue[0] - source_hue[0]) <= 10.0)
    assert np.all(restored_saturation[0] >= source_saturation[0] * 0.70)


def test_clean_palette_relief_composite_preserves_palette_hue_and_uses_relief():
    module = _load_example_module()
    palette = Image.new("RGBA", (5, 5), (0, 0, 0, 0))
    relief = Image.new("RGBA", (5, 5), (0, 0, 0, 255))
    for y in range(1, 4):
        for x in range(1, 4):
            palette.putpixel((x, y), (88, 104, 42, 255))
            value = 60 + x * 35 + y * 8
            relief.putpixel((x, y), (value, value, value, 255))

    composed = module._compose_clean_palette_with_relief(palette, relief)
    composed_rgb = np.asarray(composed, dtype=np.uint8)[:, :, :3]
    palette_rgb = np.asarray(palette, dtype=np.uint8)[:, :, :3]
    composed_hue, composed_saturation, _ = module._rgb_to_hsv_channels(composed_rgb)
    palette_hue, palette_saturation, _ = module._rgb_to_hsv_channels(palette_rgb)
    composed_luma = composed_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    palette_luma = palette_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    active = np.asarray(palette, dtype=np.uint8)[:, :, 3] > 0

    assert np.max(np.abs(composed_hue[active] - palette_hue[active])) <= 10.0
    assert np.median(composed_saturation[active]) >= float(np.median(palette_saturation[active])) * 0.70
    assert float(composed_luma[active].max() - composed_luma[active].min()) >= 8.0
    assert not np.allclose(composed_luma[active], palette_luma[active])


def test_clean_palette_relief_composite_uses_mask_source_for_subject_alpha():
    module = _load_example_module()
    palette = Image.new("RGBA", (5, 5), (88, 104, 42, 255))
    relief = Image.new("RGBA", (5, 5), (120, 120, 120, 255))
    mask_source = Image.new("RGBA", (5, 5), (250, 250, 250, 255))
    for y in range(1, 4):
        for x in range(1, 4):
            mask_source.putpixel((x, y), (80, 90, 50, 255))

    composed = module._compose_clean_palette_with_relief(palette, relief, mask_source)
    alpha = np.asarray(composed, dtype=np.uint8)[:, :, 3]

    assert alpha[0, 0] == 0
    assert alpha[2, 2] > 0


def test_shadow_defaults_are_not_overcranked():
    module = _load_example_module()

    assert module.COMPOSITE["directional_shadow_strength"] == 0.50
    assert module.COMPOSITE["directional_highlight_strength"] == 0.24
    assert module.MAP_WARM_SHADOW_RGBA[3] == 14
    assert module.MAP_CONTACT_BLUE_RGBA[3] == 22


def test_reference_contact_glows_stay_subtle_for_light_map():
    module = _load_example_module()

    assert min(module.MAP_BACKGROUND_RGB) >= 250
    assert module.MAP_BLUE_GLOW_RGBA[3] <= 16
    assert module.MAP_LAVENDER_GLOW_RGBA[3] <= 14
    assert module.MAP_WARM_SHADOW_RGBA[3] <= 14
    assert module.MAP_CONTACT_BLUE_RGBA[3] <= 22
