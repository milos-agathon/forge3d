from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyproj")
pytest.importorskip("shapely")
from PIL import Image
from shapely.geometry import Polygon


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples"
    module_path = example_dir / "osm_city_demo.py"
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    spec = importlib.util.spec_from_file_location("osm_city_demo", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


osm_city_demo = _load_example_module()


def test_utm_epsg_for_copenhagen():
    assert osm_city_demo.utm_epsg_for_lon_lat(12.56553, 55.67594) == 32633


def test_parse_osm_numeric_handles_units_and_commas():
    assert osm_city_demo.parse_osm_numeric("12,5 m") == 12.5
    assert abs(osm_city_demo.parse_osm_numeric("100 ft") - 30.48) < 1e-6


def test_infer_building_height_prefers_explicit_height_and_clamps():
    assert osm_city_demo.infer_building_height({"height": "120"}) == 80.0
    assert osm_city_demo.infer_building_height({"height": "9.5"}) == 9.5


def test_infer_building_height_falls_back_to_levels_and_default():
    assert abs(osm_city_demo.infer_building_height({"building:levels": "3"}) - 9.6) < 1e-6
    assert osm_city_demo.infer_building_height({}) == 12.0


def test_building_bin_thresholds():
    assert osm_city_demo.building_bin(15.0) == "low"
    assert osm_city_demo.building_bin(15.01) == "mid"
    assert osm_city_demo.building_bin(30.0) == "mid"
    assert osm_city_demo.building_bin(30.01) == "high"
    assert osm_city_demo.building_bin(45.0) == "high"
    assert osm_city_demo.building_bin(45.01) == "landmark"


def test_prepare_polygonal_geom_falls_back_when_simplify_collapses_area():
    polygon = Polygon(
        [
            (864.0292510336149, 502.2748162522912),
            (861.2202524687164, 508.1652366388589),
            (861.213240924757, 508.179629817605),
            (864.5529150093207, 502.5029105292633),
            (864.0292510336149, 502.2748162522912),
        ]
    )
    prepared = osm_city_demo.prepare_polygonal_geom(polygon, simplify_tolerance=0.75)
    assert prepared is not None
    assert prepared.area > 1.0
    assert prepared.equals(polygon)


def test_merge_surface_geometry_preserves_polygon_holes():
    polygon = Polygon(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        holes=[[(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)]],
    )
    merged = osm_city_demo.merge_surface_geometry([{"geometry": polygon, "tags": {}}])
    assert merged is not None
    parts = osm_city_demo._polygon_parts(merged)
    assert len(parts) == 1
    assert len(parts[0].interiors) == 1


def test_alpha_bounds_returns_padded_subject_extent():
    alpha = osm_city_demo.np.zeros((12, 14), dtype=osm_city_demo.np.uint8)
    alpha[3:8, 4:10] = 255
    assert osm_city_demo.alpha_bounds(alpha, threshold=10, pad=2) == (2, 1, 12, 10)


def test_crop_subject_uses_alpha_channel_bounds():
    image = Image.new("RGBA", (100, 80), (0, 0, 0, 0))
    block = Image.new("RGBA", (24, 18), (255, 200, 40, 255))
    image.alpha_composite(block, dest=(34, 28))
    cropped = osm_city_demo.crop_subject(image, threshold=10, pad_ratio=0.0, min_pad=0)
    assert cropped.size == (24, 18)


def test_make_poster_background_shape_and_gradient_progression():
    background = osm_city_demo.make_poster_background(96, 96)
    assert background.mode == "RGBA"
    assert background.size == (96, 96)
    arr = osm_city_demo.np.asarray(background, dtype=osm_city_demo.np.uint8)
    top_luma = float(arr[4, 48, :3].mean())
    bottom_luma = float(arr[90, 48, :3].mean())
    assert bottom_luma > top_luma


def test_apply_depth_grade_rgba_moves_far_color_toward_background():
    near = osm_city_demo.apply_depth_grade_rgba(
        (220, 180, 80, 255),
        radial_t=0.0,
    )
    far = osm_city_demo.apply_depth_grade_rgba(
        (220, 180, 80, 255),
        radial_t=1.0,
    )
    near_luma = sum(near[:3])
    far_luma = sum(far[:3])
    near_spread = max(near[:3]) - min(near[:3])
    far_spread = max(far[:3]) - min(far[:3])
    assert far_luma > near_luma
    assert far_spread < near_spread
