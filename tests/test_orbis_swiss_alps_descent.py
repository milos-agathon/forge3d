from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO_ROOT / "examples" / "orbis_swiss_alps_descent.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("orbis_swiss_alps_descent", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    examples_dir = str(EXAMPLE_PATH.parent)
    added_examples_dir = examples_dir not in sys.path
    previous_forge3d = sys.modules.get("forge3d")
    previous_module = sys.modules.get(spec.name)
    if added_examples_dir:
        sys.path.insert(0, examples_dir)
    sys.modules["forge3d"] = types.ModuleType("forge3d")
    sys.modules[spec.name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
        if previous_forge3d is None:
            sys.modules.pop("forge3d", None)
        else:
            sys.modules["forge3d"] = previous_forge3d
        if added_examples_dir:
            sys.path.remove(examples_dir)
    return module


@pytest.fixture(scope="module")
def example():
    return _load_module()


def test_descent_opens_on_the_whole_earth_and_ends_exactly(example):
    path = example.descent_waypoints(360, end_altitude_m=2500.0)
    assert len(path) == 360
    assert all(len(w) == 5 for w in path)
    lon, lat, altitude, heading, pitch = path[0]
    assert (lon, lat) == example.START_TARGET
    assert altitude == example.START_ALTITUDE_M
    assert (heading, pitch) == (0.0, 0.0)
    lon, lat, altitude, heading, pitch = path[-1]
    assert (lon, lat) == pytest.approx(example.FINAL_TARGET)
    assert altitude == 2500.0
    assert heading == pytest.approx(180.0) and pitch == pytest.approx(example.FINAL_PITCH_DEG)


def test_descent_is_monotonic_and_swings_through_east(example):
    path = example.descent_waypoints(240)
    altitudes = [w[2] for w in path]
    pitches = [w[4] for w in path]
    headings = [w[3] for w in path]
    assert all(a > b for a, b in zip(altitudes, altitudes[1:]))
    assert all(a <= b + 1e-9 for a, b in zip(pitches, pitches[1:]))
    assert all(a <= b + 1e-9 for a, b in zip(headings, headings[1:]))
    # The turn happens while tilted, so it reads as an orbit, not a spin.
    first_turn = next(i for i, h in enumerate(headings) if h > 1.0)
    assert pitches[first_turn] > 20.0


def test_every_camera_stays_within_the_globe_render_bound(example):
    for _lon, _lat, altitude, _heading, pitch in example.descent_waypoints(360):
        assert altitude / math.cos(math.radians(pitch)) <= example.MAX_CAMERA_TARGET_DISTANCE_M


def test_targets_stay_inside_the_swiss_dem(example):
    for lon, lat, *_ in example.descent_waypoints(120):
        assert 5.96 < lon < 10.49 and 45.82 < lat < 47.81


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 1}, "frame_count"),
        ({"frame_count": 10, "end_altitude_m": 9_000_000.0}, "end_altitude_m"),
        ({"frame_count": 10, "knee_m": 0.0}, "knee"),
    ],
)
def test_descent_rejects_invalid_inputs(example, kwargs, message):
    frames = kwargs.pop("frame_count")
    with pytest.raises(ValueError, match=message):
        example.descent_waypoints(frames, **kwargs)


def test_fes_palette_splits_exactly_at_sea_level(example):
    stops = example.fes_stops()
    assert len(stops) == 34
    assert (stops[0][0], stops[-1][0]) == example.ELEVATION_DOMAIN
    assert stops[16] == (-example.SEA_LEVEL_EPSILON_M, "#f1f1f1")  # shallowest fes grey
    assert stops[17] == (0.0, "#024026")  # first fes land colour
    assert [s[0] for s in stops] == sorted(s[0] for s in stops)
    shallow, land = example.fes_rgb(np.array([-1.0, 1.0]))
    # No grey-green blend on continental shelves: grey just below 0 m, green above.
    assert shallow[0] == shallow[1] == shallow[2] and shallow[0] > 230
    assert land[1] > land[0]


def test_terrain_stops_are_the_fes_palette_decoded_to_linear(example):
    display = example.fes_stops()
    linear = example.terrain_colormap_stops()
    assert [s[0] for s in linear] == [s[0] for s in display]
    assert linear[0][1] == "#010101"  # #0d0d0d decoded
    assert linear[-1][1] == "#d8d8f8"  # #ededfc decoded
    for (_, d), (_, l) in zip(display, linear):
        assert all(int(l[i:i + 2], 16) <= int(d[i:i + 2], 16) for i in (1, 3, 5))


def test_earth_texture_bake_is_north_up_fes_with_flat_ground_unshaded(example):
    height, width = 32, 64
    z = np.full((height, width), -3000.0)
    z[: height // 2, : width // 2] = 800.0  # north-west quadrant is land
    texture = example.bake_earth_texture(z)
    assert texture.shape == (height, width, 3) and texture.dtype == np.uint8
    land = texture[4, 4].astype(int)
    sea = texture[height - 4, width - 4].astype(int)
    assert land[1] > land[0] and land[1] > land[2]  # fes green on land
    assert sea[0] == sea[1] == sea[2]  # fes grey at sea
    expected = example.fes_rgb(np.array([800.0]))[0]
    assert np.abs(land - expected).max() <= 1  # flat ground keeps its colour


@pytest.mark.parametrize(
    ("altitude", "text"),
    [(7_500_000.0, "7,500 km"), (2_345.0, "2.3 km"), (512.4, "512 m"), (3.25, "3.2 m")],
)
def test_format_altitude(example, altitude, text):
    assert example.format_altitude(altitude) == text
