from __future__ import annotations

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
    module_path = example_dir / "helsinki_transit_daycycle.py"
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    spec = importlib.util.spec_from_file_location("helsinki_transit_daycycle", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helsinki = _load_example_module()


def test_parse_gtfs_time_seconds_allows_after_midnight():
    assert helsinki.parse_gtfs_time_seconds("04:05:06") == 14706
    assert helsinki.parse_gtfs_time_seconds("25:02:03") == 90123
    assert helsinki.parse_gtfs_time_seconds("") is None


def test_trip_position_interpolates_along_cumulative_distance():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    trip = helsinki.TransitTrip(
        trip_id="t",
        route_id="r",
        route_type=3,
        route_name="3",
        route_rgb=(0, 128, 255),
        start_seconds=0,
        end_seconds=20,
        points_xy=points,
        cumulative_m=helsinki.cumulative_distances(points),
    )

    assert np.allclose(helsinki.trip_position_at_seconds(trip, 0), [0.0, 0.0])
    assert np.allclose(helsinki.trip_position_at_seconds(trip, 10), [10.0, 0.0])
    assert np.allclose(helsinki.trip_position_at_seconds(trip, 15), [10.0, 5.0])
    assert helsinki.trip_position_at_seconds(trip, 21) is None


def test_official_building_height_uses_floor_count_before_area_estimate():
    assert helsinki.official_building_height({"i_kerrlkm": "5", "i_kerrosala": "100"}, 100.0) == 16.0
    assert helsinki.official_building_height({"i_kerrosala": "900"}, 300.0) == 9.0
    assert helsinki.official_building_height({"tyyppi": "Asuinrakennus"}, 200.0) == 12.0


def test_traffic_daily_vehicle_count_prefers_official_kavl_total():
    assert helsinki.traffic_daily_vehicle_count({"syksyn_kavl": 21700, "autot": 20448}) == 21700
    assert helsinki.traffic_daily_vehicle_count({"ha": 100, "pa": 20, "ka": 5}) == 125


def test_traffic_hour_share_has_commuter_peak_shape():
    assert helsinki.traffic_hour_share(8 * 3600) > helsinki.traffic_hour_share(3 * 3600)
    assert helsinki.traffic_hour_share(16 * 3600) > helsinki.traffic_hour_share(22 * 3600)


def test_active_car_flow_positions_are_count_scaled_and_capped():
    points = np.array([[0.0, 0.0], [120.0, 0.0]], dtype=np.float32)
    segment = helsinki.RoadTrafficSegment(
        segment_id="traffic-1",
        street_name="TESTIKATU",
        daily_vehicles=1_000_000.0,
        heavy_share=0.0,
        speed_mps=10.0,
        points_xy=points,
        cumulative_m=helsinki.cumulative_distances(points),
    )

    active = helsinki.active_car_flow_positions(
        [segment],
        seconds=8 * 3600,
        radius_m=200.0,
        max_particles=12,
    )

    assert 1 <= len(active) <= 12
    assert all(rgb == helsinki.CAR_FLOW_RGB for _, rgb, _ in active)
