from pathlib import Path
import importlib.util
import inspect
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO / "examples" / "population_spike_worldpop" / "poland_population_contour_3d.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("poland_population_contour_3d", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_poland_contour_3d_example_contract() -> None:
    module = _load_example()

    assert module.DATA_PATH.name == "pol_pd_2020_1km_UNadj.tif"
    assert module.FINAL_PATH.name == "poland_population_contours_3d.png"
    assert module.CONTOUR_LABELS == (
        "<25",
        "25-50",
        "50-100",
        "100-200",
        "200-500",
        "500-1k",
        "1k-2.5k",
        "2.5k-5k",
        ">5k",
    )
    assert module.VINTAGE_COLOURS == (
        "#b4c79d",
        "#c8cf8f",
        "#e3d685",
        "#e8c374",
        "#dfb39b",
        "#d88b83",
        "#b85c62",
        "#954957",
        "#733745",
    )
    assert module.TARGET_CRS == "EPSG:3035"
    assert module.SNAP_SIZE >= 8192
    assert module.FINAL_WIDTH >= 8192
    assert module.FINAL_HEIGHT < module.FINAL_WIDTH
    assert module.WINDOW_SIZE >= 2200
    assert module.RENDER_SUPERSAMPLE == 8
    assert module.CAMERA_RADIUS >= 3000.0 * module.RENDER_SUPERSAMPLE
    assert module.CAMERA_ZSCALE >= 0.088
    assert module.CONTOUR_LINE_WIDTH == 0
    assert module.CONTOUR_RENDER_SHADOW == 0.0
    assert module.CONTOUR_RENDER_NORMAL_STRENGTH == 0.0
    assert module.CONTOUR_RENDER_HEIGHT_AO_STRENGTH == 0.0
    assert module.MAP_MARGIN <= 80
    assert module.MAP_TOP <= 700
    assert module.MAP_BOTTOM <= 120
    assert 180 <= module.TITLE_FONT_SIZE <= 210
    assert module.TITLE_Y >= 60
    assert module.SUBTITLE_Y >= module.TITLE_Y + 200
    assert module.CAPTION_Y <= module.FINAL_HEIGHT - 120
    assert module.CAPTION_FONT_SIZE >= 64
    assert module.LEGEND_Y >= 60
    assert module.LEGEND_SWATCH_W >= 300
    assert module.LEGEND_SWATCH_H >= 108
    assert module.LEGEND_LABEL_FONT_SIZE >= 78
    assert module.LEGEND_TITLE_FONT_SIZE >= 90


def test_contour_indices_follow_r_breaks() -> None:
    module = _load_example()
    data = np.array([[0.0, 24.9, 25.0, 50.0, 100.0], [200.0, 500.0, 1000.0, 2500.0, 5000.0]], dtype=np.float32)
    valid = np.ones_like(data, dtype=bool)

    indices = module.contour_indices(data, valid)

    assert indices.tolist() == [[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]]


def test_contour_overlay_uses_alpha_mask_and_original_colors() -> None:
    module = _load_example()
    data = np.array([[12.0, 75.0, 6000.0]], dtype=np.float32)
    valid = np.array([[True, False, True]])

    overlay = module.contour_overlay_rgba(data, valid)

    assert overlay.shape == (1, 3, 4)
    assert overlay[0, 0].tolist() == [180, 199, 157, 255]
    assert overlay[0, 1, 3] == 0
    assert overlay[0, 2].tolist() == [115, 55, 69, 255]


def test_contour_overlay_uses_class_fills_without_dark_strokes() -> None:
    module = _load_example()
    data = np.zeros((31, 31), dtype=np.float32)
    data[4:27, 4:27] = 75.0
    data[12:23, 12:23] = 6000.0
    valid = np.ones_like(data, dtype=bool)

    overlay = module.contour_overlay_rgba(data, valid)
    mid_fill = list(module.hex_to_rgb(module.VINTAGE_COLOURS[2])) + [255]
    high_fill = list(module.hex_to_rgb(module.VINTAGE_COLOURS[-1])) + [255]

    assert overlay[7, 15].tolist() == mid_fill
    assert overlay[17, 17].tolist() == high_fill
    assert overlay[7, 15, :3].tolist() != overlay[17, 17, :3].tolist()
    assert overlay[7, 15, :3].tolist() != [38, 35, 32]


def test_contour_edges_are_subtle_not_bleeding() -> None:
    module = _load_example()

    assert module.CONTOUR_LINE_WIDTH == 0
    assert module.CONTOUR_LINE_DARKEN >= 0.90


def test_high_density_palette_is_muted_and_connected_to_pinks() -> None:
    module = _load_example()
    palette = [module.hex_to_rgb(color) for color in module.VINTAGE_COLOURS]

    assert palette[6][0] <= 184
    assert palette[7][0] <= 154
    assert palette[8][0] <= 124
    assert max(abs(a - b) for a, b in zip(palette[5], palette[6])) <= 62
    assert max(abs(a - b) for a, b in zip(palette[6], palette[7])) <= 48


def test_contour_lines_are_thickened_inside_own_band() -> None:
    module = _load_example()
    data = np.zeros((21, 21), dtype=np.float32)
    data[5:16, 5:16] = 75.0
    valid = np.ones_like(data, dtype=bool)

    overlay = module.contour_overlay_rgba(data, valid)
    line = list(module.darken_rgb(module.hex_to_rgb(module.VINTAGE_COLOURS[2]))) + [255]

    assert overlay[5, 10].tolist() != line
    assert overlay[7, 10].tolist() != line
    assert overlay[8, 10].tolist() != line
    assert overlay[10, 10].tolist() != line
    assert overlay[4, 10].tolist() != line


def test_overlay_generalizes_tiny_orphan_fragments() -> None:
    module = _load_example()
    data = np.zeros((31, 31), dtype=np.float32)
    data[15, 15] = 6000.0
    valid = np.ones_like(data, dtype=bool)

    overlay = module.contour_overlay_rgba(data, valid)

    assert overlay[15, 15, :3].tolist() == list(module.hex_to_rgb(module.VINTAGE_COLOURS[0]))


def test_overlay_fills_internal_invalid_holes_as_low_class() -> None:
    module = _load_example()
    data = np.zeros((9, 9), dtype=np.float32)
    valid = np.ones_like(data, dtype=bool)
    valid[4, 4] = False

    overlay = module.contour_overlay_rgba(data, valid)

    assert overlay[4, 4].tolist() == list(module.hex_to_rgb(module.VINTAGE_COLOURS[0])) + [255]


def test_cleanup_does_not_oversaturate_legend_red() -> None:
    module = _load_example()
    red = np.array(module.hex_to_rgb(module.VINTAGE_COLOURS[6]), dtype=np.uint8)
    raw = red.reshape(1, 1, 3)

    cleaned = module.cleanup_snapshot(raw)[0, 0].astype(int)

    assert cleaned[0] <= int(red[0]) + 8
    assert cleaned[1] >= int(red[1]) - 8
    assert cleaned[2] >= int(red[2]) - 8


def test_cleanup_adds_soft_relief_without_dark_walls() -> None:
    module = _load_example()
    raw = np.tile(np.array(module.hex_to_rgb(module.VINTAGE_COLOURS[1]), dtype=np.uint8), (41, 41, 1))
    raw[10:31, 10:31] = np.array(module.hex_to_rgb(module.VINTAGE_COLOURS[7]), dtype=np.uint8)

    cleaned = module.cleanup_snapshot(raw)
    lum = (
        0.2126 * cleaned[:, :, 0].astype(np.float32)
        + 0.7152 * cleaned[:, :, 1].astype(np.float32)
        + 0.0722 * cleaned[:, :, 2].astype(np.float32)
    )
    plateau = lum[10:31, 10:31]

    assert plateau.max() - plateau.min() >= 6.0
    assert plateau.min() >= module.MIN_DETAIL_LUMA


def test_cleanup_removes_one_pixel_cap_speckles() -> None:
    module = _load_example()
    high = np.array(module.hex_to_rgb(module.VINTAGE_COLOURS[7]), dtype=np.uint8)
    raw = np.tile(high, (15, 15, 1))
    raw[7, 7] = np.array(module.hex_to_rgb(module.VINTAGE_COLOURS[2]), dtype=np.uint8)

    cleaned = module.cleanup_snapshot(raw)

    assert np.linalg.norm(cleaned[7, 7].astype(np.int16) - high.astype(np.int16)) < 18


def test_legend_canvas_fits_larger_text() -> None:
    module = _load_example()
    legend = module.make_contour_legend()

    assert legend.height >= (
        module.LEGEND_TITLE_FONT_SIZE
        + module.LEGEND_SWATCH_H
        + module.LEGEND_LABEL_FONT_SIZE
        + 40
    )


def test_height_dem_is_mostly_stepped_not_spiky() -> None:
    module = _load_example()
    data = np.array([[10.0, 30.0, 75.0], [150.0, 300.0, 900.0], [1500.0, 3000.0, 6000.0]], dtype=np.float32)
    valid = np.ones_like(data, dtype=bool)

    old_radius = module.SMOOTH_RADIUS
    old_supersample = module.RENDER_SUPERSAMPLE
    try:
        module.SMOOTH_RADIUS = 0.0
        module.RENDER_SUPERSAMPLE = 1
        height = module.build_contour_height_dem(data, 6000.0, valid)
        expected_bands = module.contour_indices(data, valid).astype(np.float32) / (len(module.CONTOUR_LABELS) - 1) * 6000.0
    finally:
        module.SMOOTH_RADIUS = old_radius
        module.RENDER_SUPERSAMPLE = old_supersample

    assert np.max(np.abs(height - expected_bands)) <= 6000.0 * 0.08


def test_smooth_density_keeps_invalid_cells_zero() -> None:
    module = _load_example()
    data = np.zeros((5, 5), dtype=np.float32)
    data[2, 2] = 100.0
    valid = np.ones_like(data, dtype=bool)
    valid[0, 0] = False

    old_radius = module.SMOOTH_RADIUS
    try:
        module.SMOOTH_RADIUS = 1.0
        smoothed = module.smooth_density(data, valid)
    finally:
        module.SMOOTH_RADIUS = old_radius

    assert smoothed.shape == data.shape
    assert smoothed[0, 0] == 0.0
    assert 0.0 < smoothed[2, 2] < 100.0


def test_prepare_render_density_supersamples_data_and_mask() -> None:
    module = _load_example()
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    valid = np.array([[True, False, True], [True, True, False]])

    up_data, up_valid = module.prepare_render_density(data, valid)

    assert up_data.shape == (2 * module.RENDER_SUPERSAMPLE, 3 * module.RENDER_SUPERSAMPLE)
    assert up_valid.shape == up_data.shape
    assert up_valid[0, module.RENDER_SUPERSAMPLE] == valid[0, 1]


def test_cleanup_lifts_black_shadow_fragments_without_touching_background() -> None:
    module = _load_example()
    raw = np.array([[[63, 0, 108], [12, 18, 8], [120, 130, 80]]], dtype=np.uint8)

    cleaned = module.cleanup_snapshot(raw)

    assert cleaned[0, 0].tolist() == list(module.RENDER_BG_RGB)
    assert cleaned[0, 1].mean() > raw[0, 1].mean()
    assert cleaned[0, 1].tolist() != [12, 18, 8]


def test_configure_base_module_wires_shared_renderer() -> None:
    module = _load_example()
    base_names = (
        "DATA_PATH",
        "OUTPUT_DIR",
        "CLEAN_DEM",
        "OVERLAY_PATH",
        "SNAPSHOT_PATH",
        "FINAL_PATH",
        "generate_magma_overlay",
        "compose_final_plate",
    )
    shade_names = (
        "OUTPUT_DIR",
        "CLEAN_DEM",
        "OVERLAY_PATH",
        "SNAPSHOT_PATH",
        "FINAL_PATH",
        "WINDOW_SIZE",
        "SNAP_SIZE",
        "BG_COLOR",
        "CAMERA_PHI",
        "CAMERA_THETA",
        "CAMERA_RADIUS",
        "CAMERA_FOV",
        "CAMERA_ZSCALE",
        "build_height_dem",
        "cleanup_snapshot",
    )
    old_base = {name: getattr(module.base, name) for name in base_names}
    old_shade = {name: getattr(module.height_shade, name) for name in shade_names}

    try:
        module.configure_base_module()

        assert module.base.DATA_PATH == module.DATA_PATH
        assert module.base.FINAL_PATH == module.FINAL_PATH
        assert module.base.generate_magma_overlay is module.generate_contour_overlay
        assert module.height_shade.build_height_dem is module.build_contour_height_dem
        assert module.height_shade.CAMERA_ZSCALE == module.CAMERA_ZSCALE
    finally:
        for name, value in old_base.items():
            setattr(module.base, name, value)
        for name, value in old_shade.items():
            setattr(module.height_shade, name, value)


def test_shared_height_shade_renderer_preserves_overlay_colors() -> None:
    module = _load_example()

    source = inspect.getsource(module.height_shade.render_height_shade_map)

    assert "set_overlay_preserve_colors" in source
    assert '"preserve_colors": True' in source
    assert '"zscale": CAMERA_ZSCALE' in source
    assert '"shadow": SHADOW' in source
    assert '"normal_strength": NORMAL_STRENGTH' in source
    assert '"strength": HEIGHT_AO_STRENGTH' in source
