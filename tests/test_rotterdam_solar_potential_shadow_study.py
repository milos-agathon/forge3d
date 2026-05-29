from __future__ import annotations

import datetime as dt
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyproj")
pytest.importorskip("shapely")


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples"
    module_path = example_dir / "rotterdam_solar_potential_shadow_study.py"
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    spec = importlib.util.spec_from_file_location("rotterdam_solar", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rotterdam = _load_example_module()


def _wfs_payload(ctx, *, side_m: float = 24.0, offset_x: float = 0.0) -> dict:
    cx, cy = ctx.center_xy
    half = side_m * 0.5
    coords = [
        [cx + offset_x - half, cy - half],
        [cx + offset_x + half, cy - half],
        [cx + offset_x + half, cy + half],
        [cx + offset_x - half, cy + half],
        [cx + offset_x - half, cy - half],
    ]
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{ctx.epsg}"}},
        "features": [
            {
                "type": "Feature",
                "id": "lod22.test",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "fid": 1,
                    "identificatie": "NL.IMBAG.Pand.test",
                    "b3_h_maaiveld": 2.0,
                    "b3_h_50p": 15.0,
                    "b3_h_70p": 15.2,
                    "b3_h_min": 12.0,
                    "b3_h_max": 18.0,
                    "b3_hellingshoek": 25.0,
                    "b3_azimut": 180.0,
                    "b3_dak_type": "slanted",
                    "b3_is_glas_dak": False,
                    "b3_kwaliteitsindicator": True,
                },
            }
        ],
    }


def test_clock_time_parser_accepts_minutes_and_seconds():
    assert rotterdam.parse_clock_time("09:30") == dt.time(9, 30)
    assert rotterdam.parse_clock_time("21:05:07") == dt.time(21, 5, 7)


def test_3dbag_roof_surfaces_clip_to_radius_and_preserve_slope():
    ctx = rotterdam.build_metric_context(rotterdam.DEFAULT_LON, rotterdam.DEFAULT_LAT, 15.0)
    surfaces = rotterdam.roof_surfaces_from_3dbag(_wfs_payload(ctx, side_m=40.0), ctx)

    assert len(surfaces) == 1
    surface = surfaces[0]
    assert surface.area_m2 < 40.0 * 40.0
    assert surface.geometry.within(rotterdam.Point(0.0, 0.0).buffer(15.1))
    assert surface.slope_deg == 25.0
    assert np.linalg.norm(surface.gradient_xy) > 0.1


def test_mesh_uses_real_roof_height_variation_not_flat_footprint_extrusion():
    ctx = rotterdam.build_metric_context(rotterdam.DEFAULT_LON, rotterdam.DEFAULT_LAT, 80.0)
    surfaces = rotterdam.roof_surfaces_from_3dbag(_wfs_payload(ctx, side_m=32.0), ctx)
    sun = rotterdam.sun_state_for_local_time(
        rotterdam.DEFAULT_LAT,
        rotterdam.DEFAULT_LON,
        dt.date(2026, 6, 21),
        dt.time(13, 0),
        progress=0.5,
    )
    rotterdam.evaluate_roof_surfaces(
        surfaces,
        assumptions=rotterdam.SolarAssumptions(min_usable_roof_area_m2=5.0),
        irradiance_kwh_m2=1030.0,
        sun=sun,
    )
    layers, stats = rotterdam.build_roof_mesh_layers(surfaces)
    roof_layers = [layer for layer in layers if layer.rgba != rotterdam.ROOF_COLORS["wall"]]
    roof_y = np.concatenate([layer.positions[:, 1] for layer in roof_layers])

    assert stats.roof_triangles > 0
    assert stats.wall_triangles > 0
    assert float(roof_y.max() - roof_y.min()) > 0.5


def test_solar_classification_marks_small_and_south_roofs():
    ctx = rotterdam.build_metric_context(rotterdam.DEFAULT_LON, rotterdam.DEFAULT_LAT, 80.0)
    large = rotterdam.roof_surfaces_from_3dbag(_wfs_payload(ctx, side_m=32.0), ctx)[0]
    small = rotterdam.roof_surfaces_from_3dbag(_wfs_payload(ctx, side_m=6.0, offset_x=45.0), ctx)[0]
    surfaces = [large, small]
    sun = rotterdam.sun_state_for_local_time(
        rotterdam.DEFAULT_LAT,
        rotterdam.DEFAULT_LON,
        dt.date(2026, 6, 21),
        dt.time(13, 0),
        progress=0.5,
    )

    rotterdam.evaluate_roof_surfaces(
        surfaces,
        assumptions=rotterdam.SolarAssumptions(min_usable_roof_area_m2=50.0),
        irradiance_kwh_m2=1100.0,
        sun=sun,
    )

    assert large.category in {"high", "medium"}
    assert large.annual_kwh > 0.0
    assert small.category == "constrained"
    assert small.constraint_reason == "small roof surface"


def test_lod12_fallback_adds_only_missing_buildings():
    ctx = rotterdam.build_metric_context(rotterdam.DEFAULT_LON, rotterdam.DEFAULT_LAT, 120.0)
    lod22 = rotterdam.roof_surfaces_from_3dbag(_wfs_payload(ctx, side_m=20.0), ctx)
    same_building = rotterdam.roof_surfaces_from_3dbag(_wfs_payload(ctx, side_m=18.0, offset_x=30.0), ctx)
    missing_building_payload = _wfs_payload(ctx, side_m=16.0, offset_x=-35.0)
    missing_building_payload["features"][0]["properties"]["identificatie"] = "NL.IMBAG.Pand.fallback"
    missing_building = rotterdam.roof_surfaces_from_3dbag(
        missing_building_payload,
        ctx,
        source_lod="lod12",
    )

    merged = rotterdam.add_lod12_fallback_surfaces(lod22, same_building + missing_building)

    assert len(merged) == len(lod22) + 1
    assert merged[-1].building_id == "NL.IMBAG.Pand.fallback"
    assert merged[-1].source_lod == "lod12"


def test_incomplete_3dbag_cache_is_not_treated_as_complete():
    complete = {"numberMatched": 12000, "numberReturned": 12000, "features": [{}] * 3}
    incomplete = {"numberMatched": 12000, "numberReturned": 10000, "features": [{}] * 3}

    assert rotterdam.cached_payload_complete(complete, max_features=60000)
    assert not rotterdam.cached_payload_complete(incomplete, max_features=60000)


def test_output_layout_reserves_non_map_panel_space():
    wide = rotterdam.compute_output_layout(1280, 720)
    compact = rotterdam.compute_output_layout(480, 270)

    assert wide.panel_side == "left"
    assert wide.map_width + wide.panel_width == wide.final_width
    assert wide.map_height == wide.final_height
    assert wide.map_width / wide.final_width >= 0.75
    assert compact.panel_side == "bottom"
    assert compact.map_height + compact.panel_height == compact.final_height


def test_osm_water_lines_are_drawn_even_when_water_polygons_exist(monkeypatch, tmp_path):
    ctx = rotterdam.build_metric_context(rotterdam.DEFAULT_LON, rotterdam.DEFAULT_LAT, 1200.0)

    def point_from_offset(x: float, y: float) -> dict[str, float]:
        lon, lat = ctx.to_wgs84.transform(ctx.center_xy[0] + x, ctx.center_xy[1] + y)
        return {"lon": float(lon), "lat": float(lat)}

    water_polygon = {
        "type": "way",
        "tags": {"natural": "water"},
        "geometry": [
            point_from_offset(260.0, -10.0),
            point_from_offset(300.0, -10.0),
            point_from_offset(300.0, 30.0),
            point_from_offset(260.0, 30.0),
            point_from_offset(260.0, -10.0),
        ],
    }
    river_line = {
        "type": "way",
        "tags": {"waterway": "river", "name": "Nieuwe Maas"},
        "geometry": [
            point_from_offset(-320.0, 0.0),
            point_from_offset(320.0, 0.0),
        ],
    }

    def fake_fetch(query, cache_path, refresh, *, timeout_seconds):
        _ = (query, refresh, timeout_seconds)
        if "water_ways" in cache_path.name:
            return [water_polygon]
        if "waterway_lines" in cache_path.name:
            return [river_line]
        return []

    monkeypatch.setattr(rotterdam, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rotterdam, "fetch_overpass_query_cached", fake_fetch)

    surfaces = rotterdam.build_osm_context_surfaces(
        ctx,
        lon=rotterdam.DEFAULT_LON,
        lat=rotterdam.DEFAULT_LAT,
        radius_m=1200.0,
        refresh_osm=True,
        skip_osm=False,
        osm_timeout_seconds=1.0,
    )
    water_surfaces = [
        surface for surface in surfaces if surface.rgba == rotterdam.CONTEXT_COLORS["water"]
    ]
    water_areas = sorted(surface.geometry.area for surface in water_surfaces)

    assert len(water_surfaces) == 2
    assert water_areas[-1] > 150_000.0


def test_osm_context_uses_way_only_queries_for_parks_and_water(monkeypatch, tmp_path):
    ctx = rotterdam.build_metric_context(rotterdam.DEFAULT_LON, rotterdam.DEFAULT_LAT, 1200.0)
    seen: list[tuple[str, str]] = []

    def fake_fetch(query, cache_path, refresh, *, timeout_seconds):
        _ = (refresh, timeout_seconds)
        seen.append((cache_path.name, query))
        return []

    monkeypatch.setattr(rotterdam, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(rotterdam, "fetch_overpass_query_cached", fake_fetch)

    surfaces = rotterdam.build_osm_context_surfaces(
        ctx,
        lon=rotterdam.DEFAULT_LON,
        lat=rotterdam.DEFAULT_LAT,
        radius_m=1200.0,
        refresh_osm=True,
        skip_osm=False,
        osm_timeout_seconds=1.0,
    )

    assert len(surfaces) == 1
    light_names = {
        name
        for name, _ in seen
        if any(marker in name for marker in ("parks", "green", "water", "river"))
    }
    assert light_names
    for name, query in seen:
        if name in light_names:
            assert "relation" not in query
