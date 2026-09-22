from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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


def test_descent_starts_at_iss_and_ends_exactly(example):
    path = example.descent_waypoints(300, end_altitude_m=2500.0)
    assert len(path) == 300
    assert all(len(w) == 5 for w in path)
    assert path[0][2] == example.START_ALTITUDE_M
    assert path[-1][2] == 2500.0


def test_descent_is_monotonic_and_tilts_up(example):
    path = example.descent_waypoints(120)
    altitudes = [w[2] for w in path]
    pitches = [w[4] for w in path]
    assert all(a > b for a, b in zip(altitudes, altitudes[1:]))
    assert all(a <= b for a, b in zip(pitches, pitches[1:]))
    assert pitches[0] == 55.0 and pitches[-1] == 77.0


def test_every_waypoint_targets_the_jungfrau_wall(example):
    for lon, lat, _altitude, heading, _pitch in example.descent_waypoints(120):
        assert (lon, lat) == example.AIM_LONLAT
        assert heading == 180.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 1}, "frame_count"),
        ({"frame_count": 10, "end_altitude_m": 500_000.0}, "end_altitude_m"),
        ({"frame_count": 10, "end_pitch_deg": 95.0}, "pitch"),
        ({"frame_count": 10, "knee_m": 0.0}, "knee"),
    ],
)
def test_descent_rejects_invalid_inputs(example, kwargs, message):
    frames = kwargs.pop("frame_count")
    with pytest.raises(ValueError, match=message):
        example.descent_waypoints(frames, **kwargs)


@pytest.mark.parametrize(
    ("altitude", "text"),
    [(408_000.0, "408 km"), (2_345.0, "2.3 km"), (512.4, "512 m"), (3.25, "3.2 m")],
)
def test_format_altitude(example, altitude, text):
    assert example.format_altitude(altitude) == text


def test_palette_is_ordered_and_spans_the_domain(example):
    stops = example.ALPINE_STOPS
    assert [v for v, _ in stops] == sorted(v for v, _ in stops)
    assert (stops[0][0], stops[-1][0]) == example.ELEVATION_DOMAIN
